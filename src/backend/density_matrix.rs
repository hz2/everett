//! The density-matrix backend.
//!
//! Simulates a circuit as a mixed state, applying gate noise and readout
//! errors from a [`NoiseModel`]. With an ideal noise model this is identical
//! to the statevector backend (the state stays pure), but the representation
//! costs `O(4^n)` memory instead of `O(2^n)`, so this backend is best suited
//! for small qubit counts (n <= 12 or so) where noise effects are the point.

use super::{Backend, drive};
use crate::circuit::Circuit;
use crate::complex::Complex64;
use crate::density::DensityMatrix;
use crate::gate::{Gate1, Gate2};
use crate::noise::NoiseModel;
use crate::rng::Rng;

/// Simulates a circuit as a density matrix, optionally with noise.
///
/// # Examples
///
/// ```
/// use everett::{Circuit, DensityMatrixBackend, NoiseModel};
///
/// let mut c = Circuit::new(2);
/// c.h(0).cnot(0, 1);
///
/// // ideal (noiseless) run: purity should be 1.
/// let exec = DensityMatrixBackend::run(&c)?;
/// assert!((exec.density_matrix().purity() - 1.0).abs() < 1e-10);
///
/// // 2% depolarizing noise: purity drops below 1.
/// let noise = NoiseModel::uniform_depolarizing(0.02);
/// let noisy = DensityMatrixBackend::run_with_noise(&c, &noise)?;
/// assert!(noisy.density_matrix().purity() < 1.0);
/// # Ok::<(), everett::Error>(())
/// ```
pub struct DensityMatrixBackend {
    rho: DensityMatrix,
    rng: Rng,
    noise: NoiseModel,
}

/// The result of a density-matrix simulation.
#[derive(Clone, Debug)]
pub struct DensityMatrixExecution {
    rho: DensityMatrix,
    classical: Vec<bool>,
}

impl DensityMatrixExecution {
    /// The final density matrix.
    #[must_use]
    pub fn density_matrix(&self) -> &DensityMatrix {
        &self.rho
    }

    /// The final classical register, indexed by [`crate::ClassicalBit`].
    #[must_use]
    pub fn classical(&self) -> &[bool] {
        &self.classical
    }

    /// Consumes the execution, returning the owned density matrix.
    #[must_use]
    pub fn into_density_matrix(self) -> DensityMatrix {
        self.rho
    }
}

impl DensityMatrixBackend {
    /// Runs `circuit` ideally (no noise, default seed).
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit is malformed.
    pub fn run(circuit: &Circuit) -> crate::Result<DensityMatrixExecution> {
        Self::run_with_noise_seeded(circuit, &NoiseModel::ideal(), 0)
    }

    /// Runs `circuit` with the given noise model and a default seed.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit is malformed.
    pub fn run_with_noise(
        circuit: &Circuit,
        noise: &NoiseModel,
    ) -> crate::Result<DensityMatrixExecution> {
        Self::run_with_noise_seeded(circuit, noise, 0)
    }

    /// Runs `circuit` with the given noise model and explicit RNG seed.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit is malformed.
    pub fn run_with_noise_seeded(
        circuit: &Circuit,
        noise: &NoiseModel,
        seed: u64,
    ) -> crate::Result<DensityMatrixExecution> {
        circuit.validate()?;
        let mut backend = Self {
            rho: DensityMatrix::zero(circuit.num_qubits()),
            rng: Rng::seed_from_u64(seed),
            noise: noise.clone(),
        };
        let classical = drive(&mut backend, circuit.ops(), circuit.num_classical());
        Ok(DensityMatrixExecution {
            rho: backend.rho,
            classical,
        })
    }
}

impl Backend for DensityMatrixBackend {
    fn apply_1q(&mut self, gate: &Gate1, target: usize) {
        // apply the unitary U rho U†.
        let u: Vec<Complex64> = gate.m.to_vec();
        self.rho.apply_unitary(&u, &[target]);
        // apply gate-error channel if configured.
        if let Some(ch) = &self.noise.after_1q.clone() {
            self.rho.apply_channel(&ch.kraus_ops(), target);
        }
    }

    fn apply_2q(&mut self, gate: &Gate2, a: usize, b: usize) {
        let u: Vec<Complex64> = gate.m.to_vec();
        // gate.m uses row ordering |q_a q_b> with a=MSB (bit 1), b=LSB (bit 0).
        // apply_unitary expects targets[i] to correspond to bit i of the local
        // index, so targets[0]=b (bit 0) and targets[1]=a (bit 1).
        self.rho.apply_unitary(&u, &[b, a]);
        if let Some(ch) = &self.noise.after_2q.clone() {
            let ops = ch.kraus_ops();
            self.rho.apply_channel(&ops, a);
            self.rho.apply_channel(&ops, b);
        }
    }

    fn apply_controlled(&mut self, controls: &[usize], gate: &Gate1, target: usize) {
        // lift controlled gate into full-register unitary, then apply.
        let n = self.rho.num_qubits();
        let u = controlled_unitary(controls, gate, target, n);
        let all_qubits: Vec<usize> = (0..n).collect();
        self.rho.apply_unitary(&u, &all_qubits);
        if let Some(ch) = &self.noise.after_1q.clone() {
            let ops = ch.kraus_ops();
            self.rho.apply_channel(&ops, target);
            for &c in controls {
                self.rho.apply_channel(&ops, c);
            }
        }
    }

    fn measure(&mut self, qubit: usize) -> bool {
        let p1 = measure_probability(&self.rho, qubit);
        let raw_outcome = self.rng.next_f64() < p1;

        // collapse and renormalize.
        collapse(&mut self.rho, qubit, raw_outcome);
        self.rho.renormalize();

        // apply readout error: flip the recorded bit with probability p.
        if self.noise.readout_error > 0.0 && self.rng.next_f64() < self.noise.readout_error {
            !raw_outcome
        } else {
            raw_outcome
        }
    }
}

// helpers

/// Returns P(qubit = 1) from the diagonal of `rho`.
fn measure_probability(rho: &DensityMatrix, qubit: usize) -> f64 {
    debug_assert!(
        qubit < rho.num_qubits(),
        "measure qubit {qubit} out of range"
    );
    let bit = 1usize << qubit;
    let dim = rho.dim();
    (0..dim)
        .filter(|&j| j & bit != 0)
        .map(|j| rho.probability(j))
        .sum()
}

/// Projects the density matrix onto the `qubit = outcome` subspace (in-place).
fn collapse(rho: &mut DensityMatrix, qubit: usize, outcome: bool) {
    debug_assert!(
        qubit < rho.num_qubits(),
        "collapse qubit {qubit} out of range"
    );
    let bit = 1usize << qubit;
    let dim = rho.dim();
    // zero out all rows and columns where the qubit bit doesn't match the outcome.
    let data = rho.data_mut();
    for i in 0..dim {
        let i_matches = ((i & bit) != 0) == outcome;
        for j in 0..dim {
            let j_matches = ((j & bit) != 0) == outcome;
            if !i_matches || !j_matches {
                data[i * dim + j] = Complex64::ZERO;
            }
        }
    }
}

/// Builds the full `2^n x 2^n` unitary matrix for a controlled single-qubit
/// gate. Rows and columns are ordered by the full n-qubit computational basis
/// index (qubit 0 = LSB). Each column `j` of U gives the image of basis state `j`.
fn controlled_unitary(controls: &[usize], gate: &Gate1, target: usize, n: usize) -> Vec<Complex64> {
    debug_assert!(
        target < n,
        "target {target} out of range for {n}-qubit register"
    );
    debug_assert!(
        controls.iter().all(|&c| c < n),
        "control qubit out of range"
    );
    debug_assert!(!controls.contains(&target), "target appears in controls");
    let dim = 1usize << n;
    let mut u = vec![Complex64::ZERO; dim * dim];
    let control_mask: usize = controls.iter().fold(0, |acc, &c| acc | (1 << c));
    let m = &gate.m;

    for col in 0..dim {
        if col & control_mask == control_mask {
            // all controls set: the gate mixes col with its target-flipped partner.
            let col_t = (col >> target) & 1; // target-qubit value of this column
            let partner = col ^ (1 << target); // column with target bit flipped
            if col_t == 0 {
                // input |col> has target=0; output is m00*|col> + m10*|partner>
                u[col * dim + col] = m[0]; // row=col,    col=col
                u[partner * dim + col] = m[2]; // row=partner, col=col
            } else {
                // input |col> has target=1; output is m01*|partner> + m11*|col>
                u[partner * dim + col] = m[1]; // row=partner, col=col
                u[col * dim + col] = m[3]; // row=col,     col=col
            }
        } else {
            // controls not all set: identity on this column.
            u[col * dim + col] = Complex64::ONE;
        }
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{Gate1, Gate2};
    use crate::{Circuit, StateVectorBackend};

    fn bell_circuit() -> Circuit {
        let mut c = Circuit::new(2);
        c.h(0).cnot(0, 1);
        c
    }

    #[test]
    fn ideal_matches_statevector_probabilities() {
        // without noise the density-matrix probabilities must match statevector.
        let c = bell_circuit();
        let sv = StateVectorBackend::run(&c).unwrap();
        let dm = DensityMatrixBackend::run(&c).unwrap();
        let rho = dm.density_matrix();
        for j in 0..4 {
            let sv_prob = sv.state().probability(j);
            let dm_prob = rho.probability(j);
            assert!(
                (sv_prob - dm_prob).abs() < 1e-12,
                "basis {j}: sv={sv_prob} dm={dm_prob}"
            );
        }
    }

    #[test]
    fn ideal_state_is_pure() {
        let c = bell_circuit();
        let exec = DensityMatrixBackend::run(&c).unwrap();
        assert!((exec.density_matrix().purity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn depolarizing_reduces_purity() {
        let c = bell_circuit();
        let noise = NoiseModel::uniform_depolarizing(0.05);
        let exec = DensityMatrixBackend::run_with_noise(&c, &noise).unwrap();
        assert!(exec.density_matrix().purity() < 1.0);
    }

    #[test]
    fn trace_preserved_under_noise() {
        let c = bell_circuit();
        let noise = NoiseModel::uniform_depolarizing(0.1);
        let exec = DensityMatrixBackend::run_with_noise(&c, &noise).unwrap();
        assert!((exec.density_matrix().trace() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn amplitude_damping_decays_excited_state() {
        // start in |1> (X|0>), apply heavy damping; should be close to |0>.
        let mut c = Circuit::new(1);
        c.x(0);
        let noise = NoiseModel::amplitude_damping(0.99);
        let exec = DensityMatrixBackend::run_with_noise(&c, &noise).unwrap();
        let rho = exec.density_matrix();
        // after strong damping, P(|0>) >> P(|1>).
        assert!(rho.probability(0) > 0.9, "P(0)={}", rho.probability(0));
    }

    #[test]
    fn dephasing_kills_coherences() {
        // |+> has maximum coherence. the dephasing channel rho -> (1-p) rho + p Z rho Z
        // fully kills off-diagonals at p=0.5 (giving the 50/50 classical mixture).
        let mut c = Circuit::new(1);
        c.h(0);
        let noise = NoiseModel::dephasing(0.5);
        let exec = DensityMatrixBackend::run_with_noise(&c, &noise).unwrap();
        let rho = exec.density_matrix();
        assert!(rho.get(0, 1).norm() < 1e-12, "off-diag: {}", rho.get(0, 1));
        assert!(rho.get(1, 0).norm() < 1e-12, "off-diag: {}", rho.get(1, 0));
    }

    #[test]
    fn measurement_collapses_state() {
        // after measuring a qubit the density matrix must be a computational
        // basis state (purity = 1 for a 1-qubit circuit).
        let mut c = Circuit::with_classical(1, 1);
        c.h(0).measure(0, 0);
        let exec = DensityMatrixBackend::run(&c).unwrap();
        assert!((exec.density_matrix().purity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ideal_cnot_matches_statevector_all_n() {
        // spot check for n=2..4 that ideal density-matrix backend matches sv.
        for n in 2..=4 {
            let mut c = Circuit::new(n);
            c.h(0);
            for q in 0..n - 1 {
                c.cnot(q, q + 1);
            }
            let sv = StateVectorBackend::run(&c).unwrap();
            let dm = DensityMatrixBackend::run(&c).unwrap();
            let dim = 1 << n;
            for j in 0..dim {
                let a = sv.state().probability(j);
                let b = dm.density_matrix().probability(j);
                assert!((a - b).abs() < 1e-10, "n={n} basis {j}: sv={a} dm={b}");
            }
        }
    }

    #[test]
    fn expectation_z_of_zero_state_is_one() {
        let rho = DensityMatrix::zero(1);
        let z = rho.expectation_pauli(0, 'Z');
        assert!((z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn expectation_x_of_plus_state_is_one() {
        // H|0> = |+>; <X>_{|+>} = 1.
        let mut c = Circuit::new(1);
        c.h(0);
        let exec = DensityMatrixBackend::run(&c).unwrap();
        let x_exp = exec.density_matrix().expectation_pauli(0, 'X');
        assert!((x_exp - 1.0).abs() < 1e-10, "<X> = {x_exp}");
    }

    #[test]
    fn controlled_unitary_cnot_matches_gate2_cnot() {
        // the controlled_unitary helper for CNOT must match Gate2::cnot().
        let n = 2;
        let u = controlled_unitary(&[1], &Gate1::x(), 0, n);
        let cnot = Gate2::cnot();
        for (a, b) in u.iter().zip(cnot.m.iter()) {
            assert!((*a - *b).norm() < 1e-12, "a={a} b={b}");
        }
    }
}
