//! The circuit instruction set.
//!
//! A [`crate::circuit::Circuit`] is a sequence of [`Op`]s. Keeping operations as
//! data — rather than as direct calls into a backend — lets a circuit be
//! inspected, reused across backends, and (eventually) serialized. Measurement
//! and classical control are first-class so that algorithms like teleportation,
//! which branch on mid-circuit measurement outcomes, are expressible.

use crate::gate::{Gate1, Gate2};
use crate::qubit::{ClassicalBit, QubitId};

/// A single circuit operation.
#[derive(Clone, PartialEq, Debug)]
pub enum Op {
    /// Apply a single-qubit gate to `target`.
    Apply1 {
        /// The gate to apply.
        gate: Gate1,
        /// The qubit it acts on.
        target: QubitId,
    },
    /// Apply a two-qubit gate to operands `a` and `b` (in the gate's basis
    /// order, `a` more significant).
    Apply2 {
        /// The gate to apply.
        gate: Gate2,
        /// The first (more-significant) operand.
        a: QubitId,
        /// The second operand.
        b: QubitId,
    },
    /// Apply a single-qubit gate to `target`, conditioned on every qubit in
    /// `controls` being set.
    Controlled {
        /// The control qubits.
        controls: Vec<QubitId>,
        /// The gate applied on the all-controls-set subspace.
        gate: Gate1,
        /// The target qubit.
        target: QubitId,
    },
    /// Measure `qubit` in the computational basis and store the outcome in the
    /// classical bit `into`.
    Measure {
        /// The qubit to measure.
        qubit: QubitId,
        /// The classical bit receiving the outcome.
        into: ClassicalBit,
    },
    /// Run `then` only if classical bit `bit` is set. The body is boxed because
    /// `Op` is otherwise small and we do not want every variant to pay for it.
    IfClassic {
        /// The classical bit guarding the body.
        bit: ClassicalBit,
        /// The operation to run when the bit is set.
        then: Box<Op>,
    },
}
