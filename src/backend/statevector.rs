//! The dense statevector backend.

use super::{Backend, drive};
use crate::circuit::Circuit;
use crate::gate::{Gate1, Gate2};
use crate::kernel;
use crate::measure::measure_qubit;
use crate::rng::Rng;
use crate::state::State;

/// Simulates a circuit by evolving a dense [`State`] of `2^n` amplitudes.
///
/// # Examples
///
/// ```
/// use everett::prelude::*;
///
/// let mut c = Circuit::new(1);
/// c.h(0);
/// let exec = StateVectorBackend::run(&c)?;
/// // |+> has equal probability on |0> and |1>.
/// assert!((exec.state().probability(0) - 0.5).abs() < 1e-12);
/// # Ok::<(), everett::Error>(())
/// ```
pub struct StateVectorBackend {
    state: State,
    rng: Rng,
}

/// The result of running a circuit: the final state and classical register.
#[derive(Clone, Debug)]
pub struct Execution {
    state: State,
    classical: Vec<bool>,
}

impl Execution {
    /// The final statevector.
    #[must_use]
    pub fn state(&self) -> &State {
        &self.state
    }

    /// The final classical register, indexed by [`crate::ClassicalBit`].
    #[must_use]
    pub fn classical(&self) -> &[bool] {
        &self.classical
    }

    /// Consumes the execution, returning the owned final state.
    #[must_use]
    pub fn into_state(self) -> State {
        self.state
    }
}

impl StateVectorBackend {
    /// Runs `circuit` from the all-zeros state with a default RNG seed.
    ///
    /// Because the seed is fixed, runs are reproducible. Use [`Self::run_seeded`]
    /// to vary the measurement stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit is malformed (e.g. an out-of-range qubit
    /// index). Such circuits are normally rejected at build time; this is the
    /// final guard.
    pub fn run(circuit: &Circuit) -> crate::Result<Execution> {
        Self::run_seeded(circuit, 0)
    }

    /// Runs `circuit` from the all-zeros state with the given RNG seed.
    ///
    /// # Errors
    ///
    /// See [`Self::run`].
    pub fn run_seeded(circuit: &Circuit, seed: u64) -> crate::Result<Execution> {
        circuit.validate()?;
        let mut backend = Self {
            state: State::zero(circuit.num_qubits()),
            rng: Rng::seed_from_u64(seed),
        };
        let classical = drive(&mut backend, circuit.ops(), circuit.num_classical());
        Ok(Execution {
            state: backend.state,
            classical,
        })
    }
}

impl Backend for StateVectorBackend {
    fn apply_1q(&mut self, gate: &Gate1, target: usize) {
        kernel::apply_1q(self.state.amplitudes_mut(), target, gate);
    }

    fn apply_2q(&mut self, gate: &Gate2, a: usize, b: usize) {
        kernel::apply_2q(self.state.amplitudes_mut(), a, b, gate);
    }

    fn apply_controlled(&mut self, controls: &[usize], gate: &Gate1, target: usize) {
        kernel::apply_controlled_1q(self.state.amplitudes_mut(), controls, target, gate);
    }

    fn measure(&mut self, qubit: usize) -> bool {
        measure_qubit(&mut self.state, qubit, &mut self.rng)
    }
}
