//! The stabilizer (Clifford) backend.
//!
//! Clifford circuits — those built from `H`, `S`, `CNOT`, and the Paulis — are
//! efficiently classically simulable by the Gottesman–Knill theorem. Rather than
//! store `2^n` amplitudes, this backend tracks the state's *stabilizer group*: a
//! set of `n` Pauli operators that fix the state (`P|psi> = |psi>`). That needs
//! only `O(n^2)` bits and `O(n^2)` work per gate, so Clifford circuits on
//! hundreds of qubits are routine — far past the statevector backend's reach.
//!
//! The implementation is the Aaronson–Gottesman tableau (the "CHP" algorithm,
//! <https://arxiv.org/abs/quant-ph/0406196>). The tableau holds `2n` Pauli rows
//! — `n` *stabilizer* generators and `n` *destabilizer* generators — plus one
//! scratch row. Each row is a Pauli string encoded by bits `x_j`, `z_j` (the
//! Pauli on qubit `j` is `I`, `X`, `Z`, or `Y` for `(x,z) =` `00`, `10`, `01`,
//! `11`) and a phase bit `r` (`0` → `+1`, `1` → `-1`).
//!
//! Only Clifford gates are accepted; a non-Clifford gate (e.g. `T`, a generic
//! rotation, or a Toffoli) makes [`StabilizerBackend::run`] return
//! [`crate::Error::NonClifford`].

use super::{Backend, drive};
use crate::circuit::Circuit;
use crate::error::Error;
use crate::gate::{Gate1, Gate2};
use crate::rng::Rng;

/// A Pauli operator on `n` qubits, with an overall `+1`/`-1` sign.
///
/// Returned by [`StabilizerBackend`] as the generators of the final state's
/// stabilizer group. Each generator `P` satisfies `P|psi> = |psi>`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PauliString {
    // per qubit: the (x, z) bit pair, same encoding as the tableau.
    x: Vec<bool>,
    z: Vec<bool>,
    // true if the overall sign is -1.
    neg: bool,
}

impl PauliString {
    /// The number of qubits this operator acts on.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.x.len()
    }

    /// The sign of the operator: `false` for `+1`, `true` for `-1`.
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.neg
    }

    /// The single-qubit Pauli on qubit `q`, as one of `'I'`, `'X'`, `'Y'`, `'Z'`.
    ///
    /// # Panics
    ///
    /// Panics if `q` is out of range.
    #[must_use]
    pub fn pauli(&self, q: usize) -> char {
        match (self.x[q], self.z[q]) {
            (false, false) => 'I',
            (true, false) => 'X',
            (false, true) => 'Z',
            (true, true) => 'Y',
        }
    }

    /// The expectation value `<psi|P|psi>` of this Pauli operator on `state`.
    ///
    /// For any state stabilized by `P` (`P|psi> = |psi>`) this is exactly `+1`,
    /// which is what makes it a cross-check against another backend: run a
    /// Clifford circuit on both, then confirm the statevector has expectation
    /// `+1` for every generator the tableau reports.
    ///
    /// # Panics
    ///
    /// Panics if `state` has a different qubit count than this operator.
    #[must_use]
    pub fn expectation(&self, state: &crate::State) -> f64 {
        assert_eq!(
            state.num_qubits(),
            self.num_qubits(),
            "qubit count mismatch"
        );
        // apply P to a copy of the state, then take the real part of the overlap
        // <psi|(P|psi>). P is Hermitian, so the expectation is real.
        let applied = self.apply_to(state);
        state.overlap(&applied).re
    }

    // applies this Pauli operator to `state`, returning the result P|psi>.
    fn apply_to(&self, state: &crate::State) -> crate::State {
        use crate::Complex64;
        let n = self.num_qubits();
        let dim = 1usize << n;
        let src = state.amplitudes();
        let mut out = vec![Complex64::ZERO; dim];
        // global factor: each Y contributes a factor of i = sqrt(-1) since we
        // encode Y via X then Z (Y = iXZ). track the power of i across qubits.
        let y_count = (0..n).filter(|&q| self.x[q] && self.z[q]).count();
        // i^(y_count): cycles 1, i, -1, -i.
        let phase = match y_count % 4 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            _ => Complex64::new(0.0, -1.0),
        };
        let sign = if self.neg { -1.0 } else { 1.0 };
        let factor = phase * sign;
        for (j, &amp) in src.iter().enumerate() {
            // X part flips the bits where x is set; Z part flips sign per set z bit.
            let mut target = j;
            let mut zsign = 1.0;
            for q in 0..n {
                if self.x[q] {
                    target ^= 1 << q;
                }
                if self.z[q] && (j >> q) & 1 == 1 {
                    zsign = -zsign;
                }
            }
            out[target] += amp * factor * zsign;
        }
        crate::State::from_amplitudes(out).expect("dim is a power of two")
    }
}

impl std::fmt::Display for PauliString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", if self.neg { '-' } else { '+' })?;
        for q in 0..self.num_qubits() {
            write!(f, "{}", self.pauli(q))?;
        }
        Ok(())
    }
}

// the Clifford primitives every accepted gate decomposes into. expressed on the
// backend's own qubit indices.
enum Prim {
    H(usize),
    S(usize),
    // S^dagger; S applied three times, but tracked directly for clarity.
    Sdg(usize),
    X(usize),
    Y(usize),
    Z(usize),
    Cnot(usize, usize),
}

/// Simulates a Clifford circuit via a stabilizer tableau.
///
/// # Examples
///
/// ```
/// use everett::{Circuit, StabilizerBackend};
///
/// // a 3-qubit GHZ circuit is Clifford.
/// let mut c = Circuit::new(3);
/// c.h(0).cnot(0, 1).cnot(1, 2);
/// let exec = StabilizerBackend::run(&c)?;
/// // the stabilizer group is generated by XXX, ZZI, IZZ (up to ordering/signs).
/// assert_eq!(exec.generators().len(), 3);
/// # Ok::<(), everett::Error>(())
/// ```
pub struct StabilizerBackend {
    n: usize,
    // tableau rows 0..n are destabilizers, n..2n are stabilizers, 2n is scratch.
    // x[row][col], z[row][col] are the Pauli bits; r[row] is the phase bit.
    x: Vec<Vec<bool>>,
    z: Vec<Vec<bool>>,
    r: Vec<bool>,
    rng: Rng,
    // first non-Clifford gate encountered, if any. poisons the run.
    error: Option<Error>,
}

/// The result of running a Clifford circuit on the stabilizer backend.
#[derive(Clone, Debug)]
pub struct StabilizerExecution {
    generators: Vec<PauliString>,
    classical: Vec<bool>,
}

impl StabilizerExecution {
    /// The `n` generators of the final state's stabilizer group.
    #[must_use]
    pub fn generators(&self) -> &[PauliString] {
        &self.generators
    }

    /// The final classical register, indexed by [`crate::ClassicalBit`].
    #[must_use]
    pub fn classical(&self) -> &[bool] {
        &self.classical
    }
}

impl StabilizerBackend {
    /// Runs `circuit` from `|0...0>` with a default RNG seed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NonClifford`] if the circuit uses a gate outside the
    /// Clifford group, or a validation error for a malformed circuit.
    pub fn run(circuit: &Circuit) -> crate::Result<StabilizerExecution> {
        Self::run_seeded(circuit, 0)
    }

    /// Runs `circuit` from `|0...0>` with the given RNG seed (for measurement).
    ///
    /// # Errors
    ///
    /// See [`Self::run`].
    pub fn run_seeded(circuit: &Circuit, seed: u64) -> crate::Result<StabilizerExecution> {
        circuit.validate()?;
        let n = circuit.num_qubits();
        let mut backend = Self::identity(n, seed);
        let classical = drive(&mut backend, circuit.ops(), circuit.num_classical());
        if let Some(err) = backend.error {
            return Err(err);
        }
        Ok(StabilizerExecution {
            generators: backend.stabilizer_generators(),
            classical,
        })
    }

    // the tableau for |0...0>: destabilizer i is X_i, stabilizer i is Z_i.
    fn identity(n: usize, seed: u64) -> Self {
        let rows = 2 * n + 1;
        let mut x = vec![vec![false; n]; rows];
        let mut z = vec![vec![false; n]; rows];
        for i in 0..n {
            x[i][i] = true; // destabilizers: X_i
            z[n + i][i] = true; // stabilizers: Z_i
        }
        Self {
            n,
            x,
            z,
            r: vec![false; rows],
            rng: Rng::seed_from_u64(seed),
            error: None,
        }
    }

    // --- Clifford primitive rules (Aaronson–Gottesman, section II) -----------

    fn prim_h(&mut self, q: usize) {
        for i in 0..2 * self.n {
            // r ^= x*z, then swap x and z for this qubit.
            self.r[i] ^= self.x[i][q] && self.z[i][q];
            let (xi, zi) = (self.x[i][q], self.z[i][q]);
            self.x[i][q] = zi;
            self.z[i][q] = xi;
        }
    }

    fn prim_s(&mut self, q: usize) {
        for i in 0..2 * self.n {
            // r ^= x*z, then z ^= x.
            self.r[i] ^= self.x[i][q] && self.z[i][q];
            self.z[i][q] ^= self.x[i][q];
        }
    }

    fn prim_cnot(&mut self, a: usize, b: usize) {
        for i in 0..2 * self.n {
            // r ^= x_a z_b (x_b ^ z_a ^ 1); x_b ^= x_a; z_a ^= z_b.
            self.r[i] ^= self.x[i][a] && self.z[i][b] && (self.x[i][b] ^ self.z[i][a] ^ true);
            self.x[i][b] ^= self.x[i][a];
            self.z[i][a] ^= self.z[i][b];
        }
    }

    // Pauli gates are S/H combinations but have trivial direct rules: X, Y, Z
    // only flip phase bits depending on the row's Pauli content.
    fn prim_x(&mut self, q: usize) {
        // X = H S S H. simplest correct route: phase flips where z_q is set.
        for i in 0..2 * self.n {
            self.r[i] ^= self.z[i][q];
        }
    }

    fn prim_z(&mut self, q: usize) {
        for i in 0..2 * self.n {
            self.r[i] ^= self.x[i][q];
        }
    }

    fn prim_y(&mut self, q: usize) {
        // Y = i X Z; the i is a global phase on the operator action, and the
        // stabilizer phase flips by both x_q and z_q.
        for i in 0..2 * self.n {
            self.r[i] ^= self.x[i][q] ^ self.z[i][q];
        }
    }

    fn apply_prim(&mut self, p: &Prim) {
        match *p {
            Prim::H(q) => self.prim_h(q),
            Prim::S(q) => self.prim_s(q),
            Prim::Sdg(q) => {
                // S^dagger = S^3 = Z S.
                self.prim_s(q);
                self.prim_z(q);
            }
            Prim::X(q) => self.prim_x(q),
            Prim::Y(q) => self.prim_y(q),
            Prim::Z(q) => self.prim_z(q),
            Prim::Cnot(a, b) => self.prim_cnot(a, b),
        }
    }

    // --- measurement (Aaronson–Gottesman, section II) ------------------------

    fn measure_qubit(&mut self, q: usize) -> bool {
        // is there a stabilizer that anticommutes with Z_q (i.e. has x_q set)?
        let p = (self.n..2 * self.n).find(|&i| self.x[i][q]);
        match p {
            Some(p) => self.measure_random(q, p),
            None => self.measure_determined(q),
        }
    }

    // random outcome: some stabilizer anticommutes with Z_q.
    fn measure_random(&mut self, q: usize, p: usize) -> bool {
        let n = self.n;
        // for every other row anticommuting with Z_q, rowsum with row p.
        for i in 0..2 * n {
            if i != p && self.x[i][q] {
                self.rowsum(i, p);
            }
        }
        // destabilizer p-n becomes old stabilizer p; stabilizer p becomes Z_q
        // with a random sign = the measurement outcome.
        self.copy_row(p - n, p);
        for j in 0..n {
            self.x[p][j] = false;
            self.z[p][j] = false;
        }
        let outcome = self.rng.next_u64() & 1 == 1;
        self.r[p] = outcome;
        self.z[p][q] = true;
        outcome
    }

    // deterministic outcome: Z_q is already in the stabilizer group.
    fn measure_determined(&mut self, q: usize) -> bool {
        let n = self.n;
        // use the scratch row 2n to accumulate the product of stabilizers whose
        // destabilizer anticommutes with Z_q; its phase is the outcome.
        let scratch = 2 * n;
        for j in 0..n {
            self.x[scratch][j] = false;
            self.z[scratch][j] = false;
        }
        self.r[scratch] = false;
        for i in 0..n {
            if self.x[i][q] {
                self.rowsum(scratch, n + i);
            }
        }
        self.r[scratch]
    }

    // h <- h * j (Pauli product), tracking the phase per Aaronson–Gottesman.
    fn rowsum(&mut self, h: usize, j: usize) {
        let n = self.n;
        let mut sum: i32 = 2 * i32::from(self.r[h]) + 2 * i32::from(self.r[j]);
        for q in 0..n {
            sum += Self::g(self.x[j][q], self.z[j][q], self.x[h][q], self.z[h][q]);
        }
        // sum mod 4 is 0 (-> +1) or 2 (-> -1); never odd for valid tableaux.
        self.r[h] = sum.rem_euclid(4) == 2;
        for q in 0..n {
            self.x[h][q] ^= self.x[j][q];
            self.z[h][q] ^= self.z[j][q];
        }
    }

    // the phase exponent g(x1,z1,x2,z2) in {-1,0,1} for multiplying two Paulis.
    // the four-bool signature mirrors the Aaronson–Gottesman definition exactly;
    // folding the bits into enums would obscure the correspondence to the paper.
    #[allow(clippy::fn_params_excessive_bools)]
    fn g(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
        match (x1, z1) {
            (false, false) => 0,
            (true, true) => i32::from(z2) - i32::from(x2), // Y
            (true, false) => i32::from(z2) * (2 * i32::from(x2) - 1), // X
            (false, true) => i32::from(x2) * (1 - 2 * i32::from(z2)), // Z
        }
    }

    fn copy_row(&mut self, dst: usize, src: usize) {
        self.x[dst] = self.x[src].clone();
        self.z[dst] = self.z[src].clone();
        self.r[dst] = self.r[src];
    }

    // the n stabilizer generators as Pauli strings.
    fn stabilizer_generators(&self) -> Vec<PauliString> {
        (self.n..2 * self.n)
            .map(|i| PauliString {
                x: self.x[i].clone(),
                z: self.z[i].clone(),
                neg: self.r[i],
            })
            .collect()
    }

    // record the first non-Clifford gate; later gates are then no-ops.
    fn poison(&mut self, gate: &'static str) {
        if self.error.is_none() {
            self.error = Some(Error::NonClifford { gate });
        }
    }
}

impl Backend for StabilizerBackend {
    fn apply_1q(&mut self, gate: &Gate1, target: usize) {
        if self.error.is_some() {
            return;
        }
        match clifford1(gate) {
            Some(prims) => {
                for p in &prims {
                    // single-qubit prims always reference qubit 0 of the decomposition;
                    // remap to the real target.
                    self.apply_prim(&remap1(p, target));
                }
            }
            None => self.poison("non-Clifford 1-qubit gate"),
        }
    }

    fn apply_2q(&mut self, gate: &Gate2, a: usize, b: usize) {
        if self.error.is_some() {
            return;
        }
        match clifford2(gate, a, b) {
            Some(prims) => {
                for p in &prims {
                    self.apply_prim(p);
                }
            }
            None => self.poison("non-Clifford 2-qubit gate"),
        }
    }

    fn apply_controlled(&mut self, controls: &[usize], gate: &Gate1, target: usize) {
        if self.error.is_some() {
            return;
        }
        // a single control with a Clifford target is Clifford iff the controlled
        // gate is (controlled-X/Y/Z). multi-control (Toffoli etc.) is not.
        if controls.len() == 1 {
            if let Some(prims) = controlled_clifford(gate, controls[0], target) {
                for p in &prims {
                    self.apply_prim(p);
                }
                return;
            }
        }
        self.poison("non-Clifford controlled gate");
    }

    fn measure(&mut self, qubit: usize) -> bool {
        if self.error.is_some() {
            return false;
        }
        self.measure_qubit(qubit)
    }
}

// --- Clifford detection ------------------------------------------------------
//
// gates are matched against the Clifford set up to global phase, so e.g.
// rz(pi/2) is recognized as S. matching uses the gate's matrix entries.

const EPS: f64 = 1e-9;

// single-qubit decompositions are written on a placeholder qubit (always 0);
// remap1 rewrites them onto the real target.
fn remap1(p: &Prim, target: usize) -> Prim {
    match *p {
        Prim::H(_) => Prim::H(target),
        Prim::S(_) => Prim::S(target),
        Prim::Sdg(_) => Prim::Sdg(target),
        Prim::X(_) => Prim::X(target),
        Prim::Y(_) => Prim::Y(target),
        Prim::Z(_) => Prim::Z(target),
        Prim::Cnot(_, _) => unreachable!("no 2-qubit prim in a 1-qubit decomposition"),
    }
}

// returns the Clifford primitive decomposition of a single-qubit gate, or None
// if it is not Clifford. matching is up to global phase.
fn clifford1(g: &Gate1) -> Option<Vec<Prim>> {
    if matrix_eq_phase(g, &Gate1::id()) {
        Some(vec![])
    } else if matrix_eq_phase(g, &Gate1::h()) {
        Some(vec![Prim::H(0)])
    } else if matrix_eq_phase(g, &Gate1::s()) {
        Some(vec![Prim::S(0)])
    } else if matrix_eq_phase(g, &Gate1::s().adjoint()) {
        Some(vec![Prim::Sdg(0)])
    } else if matrix_eq_phase(g, &Gate1::x()) {
        Some(vec![Prim::X(0)])
    } else if matrix_eq_phase(g, &Gate1::y()) {
        Some(vec![Prim::Y(0)])
    } else if matrix_eq_phase(g, &Gate1::z()) {
        Some(vec![Prim::Z(0)])
    } else {
        None
    }
}

// returns the decomposition of a two-qubit gate on operands (a, b).
fn clifford2(g: &Gate2, a: usize, b: usize) -> Option<Vec<Prim>> {
    if matrix2_eq_phase(g, &Gate2::cnot()) {
        Some(vec![Prim::Cnot(a, b)])
    } else if matrix2_eq_phase(g, &Gate2::cz()) {
        // CZ = (I ⊗ H) CNOT (I ⊗ H).
        Some(vec![Prim::H(b), Prim::Cnot(a, b), Prim::H(b)])
    } else if matrix2_eq_phase(g, &Gate2::swap()) {
        Some(vec![Prim::Cnot(a, b), Prim::Cnot(b, a), Prim::Cnot(a, b)])
    } else {
        None
    }
}

// a single-qubit gate `g` controlled by `control` on `target`, if Clifford.
fn controlled_clifford(g: &Gate1, control: usize, target: usize) -> Option<Vec<Prim>> {
    if matrix_eq_phase(g, &Gate1::x()) {
        Some(vec![Prim::Cnot(control, target)])
    } else if matrix_eq_phase(g, &Gate1::z()) {
        Some(vec![
            Prim::H(target),
            Prim::Cnot(control, target),
            Prim::H(target),
        ])
    } else if matrix_eq_phase(g, &Gate1::id()) {
        Some(vec![])
    } else {
        None
    }
}

// true if `a` equals `b` up to a global phase: a = e^{i*phi} b for some phi.
fn matrix_eq_phase(a: &Gate1, b: &Gate1) -> bool {
    phase_aligned(&a.m, &b.m)
}

fn matrix2_eq_phase(a: &Gate2, b: &Gate2) -> bool {
    phase_aligned(&a.m, &b.m)
}

// shared global-phase-insensitive matrix comparison: find the phase from the
// first significant entry of `a` relative to `b`, then check all entries agree.
fn phase_aligned(a: &[crate::Complex64], b: &[crate::Complex64]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    // locate an index where b is non-zero to fix the phase.
    let pivot = b.iter().position(|z| z.norm() > EPS);
    let Some(p) = pivot else {
        // b is all zeros; only equal if a is too.
        return a.iter().all(|z| z.norm() <= EPS);
    };
    if a[p].norm() <= EPS {
        return false;
    }
    // phase = a[p] / b[p], unit modulus if they match up to phase.
    let phase = a[p] * b[p].conj() / b[p].norm_sqr();
    if (phase.norm() - 1.0).abs() > EPS {
        return false;
    }
    // every entry of a must equal phase * b.
    a.iter()
        .zip(b.iter())
        .all(|(za, zb)| (*za - phase * *zb).norm() <= EPS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghz_has_n_generators() {
        let mut c = Circuit::new(4);
        c.h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3);
        let exec = StabilizerBackend::run(&c).unwrap();
        assert_eq!(exec.generators().len(), 4);
    }

    #[test]
    fn t_gate_is_rejected() {
        let mut c = Circuit::new(1);
        c.t(0);
        assert!(matches!(
            StabilizerBackend::run(&c),
            Err(Error::NonClifford { .. })
        ));
    }

    #[test]
    fn generic_rotation_is_rejected() {
        let mut c = Circuit::new(1);
        c.rx(0, 0.3);
        assert!(matches!(
            StabilizerBackend::run(&c),
            Err(Error::NonClifford { .. })
        ));
    }

    #[test]
    fn rz_half_pi_is_recognized_as_s() {
        // rz(pi/2) = S up to global phase; must be accepted.
        let mut c = Circuit::new(1);
        c.rz(0, std::f64::consts::FRAC_PI_2);
        assert!(StabilizerBackend::run(&c).is_ok());
    }

    #[test]
    fn bell_measurement_is_correlated() {
        // measure both halves of a Bell pair many times; outcomes must agree.
        for seed in 0..32 {
            let mut c = Circuit::with_classical(2, 2);
            c.h(0).cnot(0, 1).measure(0, 0).measure(1, 1);
            let exec = StabilizerBackend::run_seeded(&c, seed).unwrap();
            assert_eq!(exec.classical()[0], exec.classical()[1]);
        }
    }

    #[test]
    fn deterministic_measurement_of_zero() {
        let mut c = Circuit::with_classical(1, 1);
        c.measure(0, 0);
        let exec = StabilizerBackend::run(&c).unwrap();
        assert!(!exec.classical()[0]);
    }

    #[test]
    fn large_ghz_is_feasible() {
        // 200 qubits: utterly infeasible for the statevector backend.
        let n = 200;
        let mut c = Circuit::new(n);
        c.h(0);
        for k in 0..n - 1 {
            c.cnot(k, k + 1);
        }
        let exec = StabilizerBackend::run(&c).unwrap();
        assert_eq!(exec.generators().len(), n);
    }
}
