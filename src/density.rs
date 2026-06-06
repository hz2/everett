//! The density-matrix state type.
//!
//! A *pure* state `|psi>` corresponds to the rank-1 density matrix
//! `rho = |psi><psi|`. A *mixed* state is a convex combination of pure states:
//! `rho = sum_k p_k |psi_k><psi_k|`. Both live in the same `2^n x 2^n` matrix.
//!
//! The matrix is stored flattened in row-major order: element `(i, j)` is at
//! index `i * dim + j`, where `dim = 2^n`. Gates act as `rho -> U rho U†`;
//! noise channels act as `rho -> sum_k K_k rho K_k†`.

use crate::complex::Complex64;
use crate::error::{Error, Result};

/// The mixed state of an `n`-qubit system: a `2^n x 2^n` density matrix.
///
/// The diagonal entry `rho[j][j]` is the probability of observing computational
/// basis state `|j>`. Off-diagonal entries encode coherences. The trace is kept
/// equal to 1 by construction; unitary gates preserve it and noise channels
/// preserve it by definition of a valid quantum channel.
///
/// # Examples
///
/// ```
/// use everett::DensityMatrix;
///
/// // the zero state |0><0| is a valid density matrix.
/// let rho = DensityMatrix::zero(2);
/// assert_eq!(rho.num_qubits(), 2);
/// assert!((rho.trace() - 1.0).abs() < 1e-12);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct DensityMatrix {
    // row-major: element (i,j) at data[i * dim + j]. dim = 2^n.
    data: Vec<Complex64>,
    n: usize,
}

impl DensityMatrix {
    /// Creates the all-zeros pure state `|0...0><0...0|` for `n` qubits.
    ///
    /// # Panics
    ///
    /// Panics if `n >= 32` (the matrix would require more than 16 GB).
    #[must_use]
    pub fn zero(n: usize) -> Self {
        assert!(
            n < 32,
            "n={n} would require a {}-element matrix",
            1usize << (2 * n)
        );
        let dim = 1usize << n;
        let mut data = vec![Complex64::ZERO; dim * dim];
        data[0] = Complex64::ONE; // (0,0) entry = 1; all others 0.
        Self { data, n }
    }

    /// Builds a density matrix from a pure statevector `|psi>`.
    ///
    /// The result is `rho = |psi><psi|`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `amps.len()` is not a power of two.
    pub fn from_statevector(amps: &[Complex64]) -> Result<Self> {
        let dim = amps.len();
        if !dim.is_power_of_two() {
            return Err(Error::DimensionMismatch {
                len: dim,
                expected: dim.next_power_of_two(),
            });
        }
        let n = dim.trailing_zeros() as usize;
        let mut data = vec![Complex64::ZERO; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                // rho_{ij} = psi_i * conj(psi_j)
                data[i * dim + j] = amps[i] * amps[j].conj();
            }
        }
        Ok(Self { data, n })
    }

    /// The number of qubits `n`.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// The Hilbert-space dimension `2^n`.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        1usize << self.n
    }

    /// Read-only access to the flattened row-major matrix data.
    ///
    /// Element `(i, j)` is at `data[i * dim() + j]`.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[Complex64] {
        &self.data
    }

    /// Mutable access to the flattened matrix data. For use by backends.
    #[inline]
    pub(crate) fn data_mut(&mut self) -> &mut [Complex64] {
        &mut self.data
    }

    /// The element at row `i`, column `j`.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> Complex64 {
        self.data[i * self.dim() + j]
    }

    /// The trace `sum_i rho[i][i]`, which should stay near `1`.
    #[must_use]
    pub fn trace(&self) -> f64 {
        let dim = self.dim();
        (0..dim).map(|i| self.data[i * dim + i].re).sum()
    }

    /// The purity `Tr(rho^2)`, in `[1/dim, 1]`.
    ///
    /// A pure state has purity `1`; a maximally mixed state has purity `1/dim`.
    #[must_use]
    pub fn purity(&self) -> f64 {
        let dim = self.dim();
        let mut s = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                // rho^2[i][i] = sum_k rho[i][k] * rho[k][i]
                s += self.data[i * dim + j].norm_sqr();
            }
        }
        s
    }

    /// The probability of observing computational basis state `basis`.
    ///
    /// Equal to the diagonal element `rho[basis][basis]`.
    ///
    /// # Panics
    ///
    /// Panics if `basis >= dim()`.
    #[must_use]
    pub fn probability(&self, basis: usize) -> f64 {
        let dim = self.dim();
        assert!(
            basis < dim,
            "basis state {basis} out of range for {}-qubit register",
            self.n
        );
        self.data[basis * dim + basis].re
    }

    /// Applies a unitary gate: `rho -> U rho U†`.
    ///
    /// `u` is a `2^m x 2^m` unitary acting on `m` qubits; `targets` lists those
    /// qubit indices (length `m`, all distinct, all `< n`).
    pub(crate) fn apply_unitary(&mut self, u: &[Complex64], targets: &[usize]) {
        let m = targets.len();
        let gate_dim = 1usize << m;
        debug_assert_eq!(u.len(), gate_dim * gate_dim, "gate matrix size mismatch");
        debug_assert!(
            targets.iter().all(|&t| t < self.n),
            "target qubit out of range"
        );
        // all targets must be distinct.
        debug_assert!(
            (0..targets.len()).all(|i| (i + 1..targets.len()).all(|j| targets[i] != targets[j])),
            "duplicate target qubit"
        );
        let full_dim = self.dim();

        // scratch buffer for the output matrix.
        let mut out = vec![Complex64::ZERO; full_dim * full_dim];

        // apply U on the left: out = U rho.
        // for each (row, col) of the full matrix, the gate mixes a group of
        // `gate_dim` rows that share the same values on non-target bits.
        left_multiply(&self.data, &mut out, u, targets, self.n);

        // apply U† on the right: result = out * U† = (U rho) * U†.
        // equivalently, apply conj(U) to the columns (which are the rows of rho^T).
        // we transpose out, apply U to rows, then transpose back, or do it directly.
        right_multiply_adjoint(&out, &mut self.data, u, targets, self.n);
    }

    /// Applies a quantum channel (Kraus operators) to the state.
    ///
    /// `rho -> sum_k K_k rho K_k†` where the `K_k` are `2x2` matrices acting
    /// on `target`. The Kraus operators must satisfy `sum_k K_k† K_k = I` for
    /// trace preservation; this is not checked at runtime.
    pub(crate) fn apply_channel(&mut self, kraus: &[KrausOp], target: usize) {
        debug_assert!(
            target < self.n,
            "channel target qubit {target} out of range"
        );
        debug_assert!(!kraus.is_empty(), "Kraus operator list must be non-empty");
        let full_dim = self.dim();
        let mut acc = vec![Complex64::ZERO; full_dim * full_dim];

        for k in kraus {
            let u = &k.0;
            // compute K rho K† and add into acc.
            let mut tmp = vec![Complex64::ZERO; full_dim * full_dim];
            left_multiply(&self.data, &mut tmp, u, &[target], self.n);
            right_multiply_adjoint_add(&tmp, &mut acc, u, &[target], self.n);
        }

        self.data = acc;
    }

    /// Renormalizes so `Tr(rho) = 1`. Call after measurement if needed.
    pub(crate) fn renormalize(&mut self) {
        let t = self.trace();
        if t > 0.0 {
            let inv = 1.0 / t;
            for x in &mut self.data {
                *x *= inv;
            }
        }
    }

    /// The expectation value `Tr(O rho)` for a single-qubit Pauli observable.
    ///
    /// `observable` is one of `'X'`, `'Y'`, `'Z'`, or `'I'`.
    ///
    /// # Panics
    ///
    /// Panics if `qubit >= n` or `observable` is not a recognized Pauli.
    #[must_use]
    pub fn expectation_pauli(&self, qubit: usize, observable: char) -> f64 {
        assert!(qubit < self.n, "qubit {qubit} out of range");
        let dim = self.dim();
        let bit = 1usize << qubit;

        let mut result = Complex64::ZERO;
        for i in 0..dim {
            for j in 0..dim {
                // the Pauli matrix element P[i_q][j_q] depends only on the
                // qubit-`qubit` bits of i and j; other bits must agree.
                let i_other = i & !bit;
                let j_other = j & !bit;
                if i_other != j_other {
                    continue;
                }
                let iq = (i & bit) >> qubit; // 0 or 1
                let jq = (j & bit) >> qubit;
                let p = pauli_element(observable, iq, jq);
                if p.re != 0.0 || p.im != 0.0 {
                    result += p * self.data[j * dim + i]; // Tr(P rho) = sum_{i,j} P_{ij} rho_{ji}
                }
            }
        }
        result.re
    }

    /// Traces out all qubits except `qubit`, returning the reduced 2x2 density
    /// matrix of that qubit as four `Complex64` entries `[r00, r01, r10, r11]`.
    ///
    /// # Panics
    ///
    /// Panics if `qubit >= n`.
    #[must_use]
    pub fn partial_trace_1q(&self, qubit: usize) -> [Complex64; 4] {
        assert!(qubit < self.n, "qubit {qubit} out of range");
        let dim = self.dim();
        let bit = 1usize << qubit;
        let mut reduced = [Complex64::ZERO; 4];

        for i in 0..dim {
            for j in 0..dim {
                // only trace over indices that agree on the non-target bits.
                if (i & !bit) != (j & !bit) {
                    continue;
                }
                let iq = (i & bit) >> qubit;
                let jq = (j & bit) >> qubit;
                reduced[iq * 2 + jq] += self.data[i * dim + j];
            }
        }
        reduced
    }
}

// returns the (i, j) matrix element of the single-qubit Pauli.
fn pauli_element(op: char, i: usize, j: usize) -> Complex64 {
    let c = |re, im| Complex64::new(re, im);
    match op {
        'I' => {
            if i == j {
                Complex64::ONE
            } else {
                Complex64::ZERO
            }
        }
        'X' => {
            if i == j {
                Complex64::ZERO
            } else {
                Complex64::ONE
            }
        }
        'Y' => match (i, j) {
            (0, 1) => c(0.0, -1.0),
            (1, 0) => c(0.0, 1.0),
            _ => Complex64::ZERO,
        },
        'Z' => match (i, j) {
            (0, 0) => Complex64::ONE,
            (1, 1) => c(-1.0, 0.0),
            _ => Complex64::ZERO,
        },
        other => panic!("unknown Pauli '{other}'; expected I, X, Y, or Z"),
    }
}

// ─── matrix multiply helpers ─────────────────────────────────────────────────
//
// These operate on the full 2^n x 2^n density matrix but only mix amplitudes
// within the qubit subspace the gate acts on, exactly like the statevector
// kernel but applied to each row (left multiply) or column (right multiply).

/// Left-multiply: `out = U * rho`, where `U` acts on `targets`.
fn left_multiply(
    rho: &[Complex64],
    out: &mut [Complex64],
    u: &[Complex64],
    targets: &[usize],
    n: usize,
) {
    let full_dim = 1usize << n;
    let m = targets.len();
    let gate_dim = 1usize << m;

    // for each column `col` of rho, the left-multiply on `targets` mixes
    // `gate_dim` rows; we iterate over groups in the row index.
    for col in 0..full_dim {
        // groups of rows that share the same non-target bits.
        let row_groups = full_dim / gate_dim;
        for g in 0..row_groups {
            let rows = index_group(targets, n, g);
            // read the gate_dim inputs for this column.
            let inputs: Vec<Complex64> = rows.iter().map(|&r| rho[r * full_dim + col]).collect();
            // write the gate_dim outputs: out[rows[r]][col] = sum_c U[r][c] * inputs[c].
            for (r, &row) in rows.iter().enumerate() {
                let mut acc = Complex64::ZERO;
                for c in 0..gate_dim {
                    acc += u[r * gate_dim + c] * inputs[c];
                }
                out[row * full_dim + col] = acc;
            }
        }
    }
}

/// Right-multiply by U†: `rho_out = tmp * U†`, writing into `rho_out`.
fn right_multiply_adjoint(
    tmp: &[Complex64],
    rho_out: &mut [Complex64],
    u: &[Complex64],
    targets: &[usize],
    n: usize,
) {
    let full_dim = 1usize << n;
    let m = targets.len();
    let gate_dim = 1usize << m;

    for row in 0..full_dim {
        let col_groups = full_dim / gate_dim;
        for g in 0..col_groups {
            let cols = index_group(targets, n, g);
            let inputs: Vec<Complex64> = cols.iter().map(|&c| tmp[row * full_dim + c]).collect();
            for (c_idx, &col) in cols.iter().enumerate() {
                // (tmp * U†)[row][col] = sum_k tmp[row][cols[k]] * conj(U[c_idx][k])
                // = sum_k tmp[row][cols[k]] * U†[k][c_idx]
                let mut acc = Complex64::ZERO;
                for k in 0..gate_dim {
                    // U†[k][c_idx] = conj(U[c_idx][k])
                    acc += inputs[k] * u[c_idx * gate_dim + k].conj();
                }
                rho_out[row * full_dim + col] = acc;
            }
        }
    }
}

/// Like `right_multiply_adjoint` but accumulates into `acc` (for Kraus sums).
fn right_multiply_adjoint_add(
    tmp: &[Complex64],
    acc: &mut [Complex64],
    u: &[Complex64],
    targets: &[usize],
    n: usize,
) {
    let full_dim = 1usize << n;
    let m = targets.len();
    let gate_dim = 1usize << m;

    for row in 0..full_dim {
        let col_groups = full_dim / gate_dim;
        for g in 0..col_groups {
            let cols = index_group(targets, n, g);
            let inputs: Vec<Complex64> = cols.iter().map(|&c| tmp[row * full_dim + c]).collect();
            for (c_idx, &col) in cols.iter().enumerate() {
                let mut val = Complex64::ZERO;
                for k in 0..gate_dim {
                    val += inputs[k] * u[c_idx * gate_dim + k].conj();
                }
                acc[row * full_dim + col] += val;
            }
        }
    }
}

/// Returns the `gate_dim` row/column indices that belong to group `g` for a
/// gate acting on `targets` in an `n`-qubit register.
///
/// This is the density-matrix analogue of `index_quad`/`index_pair` from the
/// kernel: it enumerates the `2^m` indices whose non-target bits encode `g`.
fn index_group(targets: &[usize], _n: usize, g: usize) -> Vec<usize> {
    let m = targets.len();
    let gate_dim = 1usize << m;

    // start from the base index with all target bits cleared.
    // insert 0 bits at each target position (in sorted order, low to high).
    let mut sorted_targets = targets.to_vec();
    sorted_targets.sort_unstable();

    let mut base = g;
    for &t in &sorted_targets {
        // insert a 0 at position t in `base`.
        let low_mask = (1usize << t) - 1;
        base = ((base >> t) << (t + 1)) | (base & low_mask);
    }

    // enumerate all 2^m combinations of the target bits.
    (0..gate_dim)
        .map(|k| {
            let mut idx = base;
            for (bit_pos, &t) in targets.iter().enumerate() {
                if (k >> bit_pos) & 1 != 0 {
                    idx |= 1usize << t;
                }
            }
            idx
        })
        .collect()
}

// ─── Kraus operator ──────────────────────────────────────────────────────────

/// A single Kraus operator: a `2x2` complex matrix acting on one qubit.
///
/// Noise channels are collections of these; physically valid channels satisfy
/// `sum_k K_k† K_k = I`, which the noise constructors in [`crate::NoiseModel`] uphold.
#[derive(Clone, Debug)]
pub struct KrausOp(pub [Complex64; 4]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_state_has_trace_one() {
        for n in 1..=4 {
            let rho = DensityMatrix::zero(n);
            assert!((rho.trace() - 1.0).abs() < 1e-12, "n={n}");
        }
    }

    #[test]
    fn from_statevector_matches_zero_state() {
        // |0> = [1, 0], so rho = [[1,0],[0,0]].
        use crate::complex::Complex64;
        let amps = vec![Complex64::ONE, Complex64::ZERO];
        let rho = DensityMatrix::from_statevector(&amps).unwrap();
        assert!((rho.get(0, 0) - Complex64::ONE).norm() < 1e-12);
        assert!(rho.get(0, 1).norm() < 1e-12);
        assert!(rho.get(1, 0).norm() < 1e-12);
        assert!(rho.get(1, 1).norm() < 1e-12);
    }

    #[test]
    fn purity_of_pure_state_is_one() {
        let rho = DensityMatrix::zero(2);
        assert!((rho.purity() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn apply_unitary_x_gate_flips_zero_state() {
        // X|0><0|X† = |1><1|
        let mut rho = DensityMatrix::zero(1);
        let x = [
            Complex64::ZERO,
            Complex64::ONE,
            Complex64::ONE,
            Complex64::ZERO,
        ];
        rho.apply_unitary(&x, &[0]);
        assert!(rho.get(0, 0).norm() < 1e-12);
        assert!((rho.get(1, 1) - Complex64::ONE).norm() < 1e-12);
    }

    #[test]
    fn partial_trace_of_product_state() {
        // |+>|0> = (|0>+|1>)/sqrt(2) tensor |0>
        // tracing out qubit 1 (|0>) should leave qubit 0 in |+><+|.
        use crate::gate::Gate1;
        use crate::kernel::apply_1q;
        let mut amps = vec![
            Complex64::ONE,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
        ];
        apply_1q(&mut amps, 0, &Gate1::h());
        let rho = DensityMatrix::from_statevector(&amps).unwrap();
        let reduced = rho.partial_trace_1q(0);
        // r00 = r11 = 0.5, r01 = r10 = 0.5 (real).
        assert!((reduced[0].re - 0.5).abs() < 1e-12, "r00={}", reduced[0]);
        assert!((reduced[3].re - 0.5).abs() < 1e-12, "r11={}", reduced[3]);
        assert!((reduced[1].re - 0.5).abs() < 1e-12, "r01={}", reduced[1]);
    }

    #[test]
    fn index_group_2q_covers_all() {
        // two-qubit gate on targets [0, 1] in a 3-qubit register: 2 groups of 4.
        let mut seen = [false; 8];
        for g in 0..2 {
            for &idx in &index_group(&[0, 1], 3, g) {
                assert!(!seen[idx], "index {idx} seen twice");
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&b| b), "not all indices covered");
    }
}
