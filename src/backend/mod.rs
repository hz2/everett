//! Execution backends.
//!
//! A [`Backend`] knows how to apply gates and measure qubits on some state
//! representation. The default backend is [`StateVectorBackend`], which evolves
//! a dense [`crate::state::State`]. The trait exists so the circuit front end
//! stays independent of how a circuit is simulated, and so the shared op-walking
//! logic — including measurement and classical control — is written once.

mod stabilizer;
mod statevector;

pub use stabilizer::{PauliString, StabilizerBackend, StabilizerExecution};
pub use statevector::{Execution, StateVectorBackend};

use crate::gate::{Gate1, Gate2};
use crate::op::Op;

/// A target that gates can be applied to and qubits measured from.
///
/// Implementors handle only the primitive operations; an internal driver layers
/// circuit traversal, classical registers, and conditional execution on top, so
/// a new backend needs to define just these four methods.
pub trait Backend {
    /// Applies a single-qubit gate to qubit `target`.
    fn apply_1q(&mut self, gate: &Gate1, target: usize);

    /// Applies a two-qubit gate to operands `a` and `b`.
    fn apply_2q(&mut self, gate: &Gate2, a: usize, b: usize);

    /// Applies a single-qubit gate to `target` on the subspace where every
    /// qubit in `controls` is set.
    fn apply_controlled(&mut self, controls: &[usize], gate: &Gate1, target: usize);

    /// Measures `qubit` in the computational basis, collapsing the state, and
    /// returns the outcome.
    fn measure(&mut self, qubit: usize) -> bool;
}

/// Walks a circuit's ops against a backend, returning the final classical
/// register. Shared by every backend.
pub(crate) fn drive<B: Backend>(backend: &mut B, ops: &[Op], num_classical: usize) -> Vec<bool> {
    let mut classical = vec![false; num_classical];
    for op in ops {
        apply_op(backend, op, &mut classical);
    }
    classical
}

fn apply_op<B: Backend>(backend: &mut B, op: &Op, classical: &mut [bool]) {
    match op {
        Op::Apply1 { gate, target } => backend.apply_1q(gate, target.index()),
        Op::Apply2 { gate, a, b } => backend.apply_2q(gate, a.index(), b.index()),
        Op::Controlled {
            controls,
            gate,
            target,
        } => {
            let cs: Vec<usize> = controls.iter().map(|q| q.index()).collect();
            backend.apply_controlled(&cs, gate, target.index());
        }
        Op::Measure { qubit, into } => {
            classical[into.index()] = backend.measure(qubit.index());
        }
        Op::IfClassic { bit, then } => {
            if classical[bit.index()] {
                apply_op(backend, then, classical);
            }
        }
    }
}
