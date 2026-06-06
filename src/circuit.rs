//! The circuit builder: the primary way to describe a computation.

use crate::error::{Error, Result};
use crate::gate::{Gate1, Gate2};
use crate::op::Op;
use crate::qubit::{ClassicalBit, QubitId};

/// A quantum circuit: an ordered list of [`Op`]s over `n` qubits and some
/// number of classical bits.
///
/// Gate methods take plain `usize` indices, return `&mut Self`, and so chain
/// fluently. They record operations without validating eagerly; call
/// [`Circuit::validate`] (which backends do automatically) to check all indices
/// at once.
///
/// # Examples
///
/// ```
/// use everett::prelude::*;
///
/// let mut c = Circuit::new(3);
/// c.h(0).cnot(0, 1).cnot(1, 2); // a 3-qubit GHZ circuit
/// assert_eq!(c.len(), 3);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct Circuit {
    ops: Vec<Op>,
    num_qubits: usize,
    num_classical: usize,
}

impl Circuit {
    /// Creates an empty circuit over `num_qubits` qubits and no classical bits.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            ops: Vec::new(),
            num_qubits,
            num_classical: 0,
        }
    }

    /// Creates an empty circuit with both a quantum and a classical register.
    #[must_use]
    pub fn with_classical(num_qubits: usize, num_classical: usize) -> Self {
        Self {
            ops: Vec::new(),
            num_qubits,
            num_classical,
        }
    }

    /// The number of qubits.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The number of classical bits.
    #[inline]
    #[must_use]
    pub fn num_classical(&self) -> usize {
        self.num_classical
    }

    /// The recorded operations.
    #[inline]
    #[must_use]
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// The number of operations.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the circuit has no operations.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    // --- the escape hatch and generic builders -------------------------------

    /// Appends an arbitrary single-qubit gate on `target`.
    pub fn gate1(&mut self, gate: Gate1, target: usize) -> &mut Self {
        self.ops.push(Op::Apply1 {
            gate,
            target: QubitId(target),
        });
        self
    }

    /// Appends an arbitrary two-qubit gate on operands `a` and `b`.
    pub fn gate2(&mut self, gate: Gate2, a: usize, b: usize) -> &mut Self {
        self.ops.push(Op::Apply2 {
            gate,
            a: QubitId(a),
            b: QubitId(b),
        });
        self
    }

    /// Appends a single-qubit gate on `target`, controlled on all of `controls`.
    pub fn controlled(&mut self, controls: &[usize], gate: Gate1, target: usize) -> &mut Self {
        self.ops.push(Op::Controlled {
            controls: controls.iter().map(|&c| QubitId(c)).collect(),
            gate,
            target: QubitId(target),
        });
        self
    }

    /// Appends another circuit's operations into this one, in place. Useful for
    /// inlining a subroutine such as a width-`k` QFT.
    ///
    /// The sub-circuit's qubit and classical indices are used as-is, so they
    /// must fit within this circuit's registers (checked by [`Self::validate`]).
    pub fn compose(&mut self, sub: &Circuit) -> &mut Self {
        self.ops.extend_from_slice(&sub.ops);
        self
    }

    // --- named single-qubit gates --------------------------------------------

    /// Appends a Hadamard on `q`.
    pub fn h(&mut self, q: usize) -> &mut Self {
        self.gate1(Gate1::h(), q)
    }
    /// Appends a Pauli-X on `q`.
    pub fn x(&mut self, q: usize) -> &mut Self {
        self.gate1(Gate1::x(), q)
    }
    /// Appends a Pauli-Y on `q`.
    pub fn y(&mut self, q: usize) -> &mut Self {
        self.gate1(Gate1::y(), q)
    }
    /// Appends a Pauli-Z on `q`.
    pub fn z(&mut self, q: usize) -> &mut Self {
        self.gate1(Gate1::z(), q)
    }
    /// Appends an S gate on `q`.
    pub fn s(&mut self, q: usize) -> &mut Self {
        self.gate1(Gate1::s(), q)
    }
    /// Appends a T gate on `q`.
    pub fn t(&mut self, q: usize) -> &mut Self {
        self.gate1(Gate1::t(), q)
    }
    /// Appends an X-rotation by `theta` on `q`.
    pub fn rx(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate1(Gate1::rx(theta), q)
    }
    /// Appends a Y-rotation by `theta` on `q`.
    pub fn ry(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate1(Gate1::ry(theta), q)
    }
    /// Appends a Z-rotation by `theta` on `q`.
    pub fn rz(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate1(Gate1::rz(theta), q)
    }
    /// Appends a relative phase `e^{i*lambda}` on `q`.
    pub fn phase(&mut self, q: usize, lambda: f64) -> &mut Self {
        self.gate1(Gate1::phase(lambda), q)
    }

    // --- named two-qubit gates -----------------------------------------------

    /// Appends a CNOT with `control` and `target`.
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.gate2(Gate2::cnot(), control, target)
    }
    /// Appends a controlled-Z on `a` and `b` (symmetric).
    pub fn cz(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate2(Gate2::cz(), a, b)
    }
    /// Appends a SWAP of `a` and `b`.
    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate2(Gate2::swap(), a, b)
    }

    // --- measurement and classical control -----------------------------------

    /// Measures `qubit`, storing the outcome in classical bit `into`.
    pub fn measure(&mut self, qubit: usize, into: usize) -> &mut Self {
        self.ops.push(Op::Measure {
            qubit: QubitId(qubit),
            into: ClassicalBit(into),
        });
        self
    }

    /// Applies a Pauli-X on `q` if classical bit `bit` is set. The basic
    /// feed-forward correction used by teleportation.
    pub fn x_if(&mut self, bit: usize, q: usize) -> &mut Self {
        self.if_classic(
            bit,
            Op::Apply1 {
                gate: Gate1::x(),
                target: QubitId(q),
            },
        )
    }

    /// Applies a Pauli-Z on `q` if classical bit `bit` is set.
    pub fn z_if(&mut self, bit: usize, q: usize) -> &mut Self {
        self.if_classic(
            bit,
            Op::Apply1 {
                gate: Gate1::z(),
                target: QubitId(q),
            },
        )
    }

    /// Runs `op` only if classical `bit` is set. General classical control.
    pub fn if_classic(&mut self, bit: usize, op: Op) -> &mut Self {
        self.ops.push(Op::IfClassic {
            bit: ClassicalBit(bit),
            then: Box::new(op),
        });
        self
    }

    // --- internal helpers ----------------------------------------------------

    /// Appends an `Op` directly. Used by the QASM parser to reconstruct circuits.
    pub(crate) fn push_op(&mut self, op: Op) {
        self.ops.push(op);
    }

    // --- validation ----------------------------------------------------------

    /// Checks that every qubit and classical-bit index is in range and that no
    /// multi-qubit gate names the same qubit twice.
    ///
    /// # Errors
    ///
    /// Returns the first [`Error`] found, or `Ok(())` if the circuit is valid.
    pub fn validate(&self) -> Result<()> {
        for op in &self.ops {
            self.validate_op(op)?;
        }
        Ok(())
    }

    fn validate_op(&self, op: &Op) -> Result<()> {
        match op {
            Op::Apply1 { target, .. } => self.check_qubit(*target),
            Op::Apply2 { a, b, .. } => {
                self.check_qubit(*a)?;
                self.check_qubit(*b)?;
                if a == b {
                    return Err(Error::DuplicateQubit { qubit: *a });
                }
                Ok(())
            }
            Op::Controlled {
                controls, target, ..
            } => {
                self.check_qubit(*target)?;
                for c in controls {
                    self.check_qubit(*c)?;
                    if c == target {
                        return Err(Error::DuplicateQubit { qubit: *c });
                    }
                }
                Ok(())
            }
            Op::Measure { qubit, into } => {
                self.check_qubit(*qubit)?;
                self.check_classical(*into)
            }
            Op::IfClassic { bit, then } => {
                self.check_classical(*bit)?;
                self.validate_op(then)
            }
        }
    }

    fn check_qubit(&self, q: QubitId) -> Result<()> {
        if q.index() < self.num_qubits {
            Ok(())
        } else {
            Err(Error::QubitOutOfRange {
                qubit: q,
                num_qubits: self.num_qubits,
            })
        }
    }

    fn check_classical(&self, b: ClassicalBit) -> Result<()> {
        if b.index() < self.num_classical {
            Ok(())
        } else {
            Err(Error::ClassicalBitOutOfRange {
                bit: b,
                num_classical: self.num_classical,
            })
        }
    }

    // --- OpenQASM 3 import/export --------------------------------------------

    /// Emits a valid `OpenQASM` 3.0 string for this circuit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Qasm`] if any gate has no `OpenQASM` 3 name.
    pub fn to_qasm(&self) -> Result<String> {
        crate::qasm::emit(self)
    }

    /// Parses an `OpenQASM` 3.0 string into a `Circuit`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Qasm`] on any parse error.
    pub fn from_qasm(src: &str) -> Result<Self> {
        crate::qasm::parse(src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fluent_chaining_records_ops() {
        let mut c = Circuit::new(2);
        c.h(0).cnot(0, 1);
        assert_eq!(c.len(), 2);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn out_of_range_qubit_is_rejected() {
        let mut c = Circuit::new(2);
        c.h(5);
        assert!(matches!(
            c.validate(),
            Err(Error::QubitOutOfRange { num_qubits: 2, .. })
        ));
    }

    #[test]
    fn duplicate_qubit_in_two_qubit_gate_is_rejected() {
        let mut c = Circuit::new(2);
        c.cnot(1, 1);
        assert!(matches!(c.validate(), Err(Error::DuplicateQubit { .. })));
    }

    #[test]
    fn measure_into_missing_classical_bit_is_rejected() {
        let mut c = Circuit::new(1); // no classical bits
        c.measure(0, 0);
        assert!(matches!(
            c.validate(),
            Err(Error::ClassicalBitOutOfRange { .. })
        ));
    }

    #[test]
    fn compose_inlines_ops() {
        let mut sub = Circuit::new(2);
        sub.h(0).cnot(0, 1);
        let mut c = Circuit::new(2);
        c.compose(&sub).x(0);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn classical_control_validates_recursively() {
        let mut c = Circuit::with_classical(1, 1);
        c.x_if(0, 3); // qubit 3 doesn't exist
        assert!(matches!(c.validate(), Err(Error::QubitOutOfRange { .. })));
    }
}
