//! The crate-wide error type.

use std::fmt;

use crate::qubit::{ClassicalBit, QubitId};

/// Errors returned when building or running a circuit.
#[derive(Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum Error {
    /// A qubit index referenced a wire outside the register.
    QubitOutOfRange {
        /// The offending qubit index.
        qubit: QubitId,
        /// The number of qubits in the register.
        num_qubits: usize,
    },
    /// A classical-bit index referenced a bit outside the classical register.
    ClassicalBitOutOfRange {
        /// The offending classical-bit index.
        bit: ClassicalBit,
        /// The number of classical bits in the register.
        num_classical: usize,
    },
    /// A gate was given the same qubit for two distinct operands.
    DuplicateQubit {
        /// The repeated qubit index.
        qubit: QubitId,
    },
    /// An amplitude buffer length was not a power of two, or did not match the
    /// declared qubit count.
    DimensionMismatch {
        /// The number of amplitudes provided.
        len: usize,
        /// The number of amplitudes expected (`2^n`).
        expected: usize,
    },
    /// A circuit run on the stabilizer backend used a gate outside the Clifford
    /// group, which that backend cannot simulate.
    NonClifford {
        /// A human-readable name for the offending gate.
        gate: &'static str,
    },
    /// A parse or emit error from the `OpenQASM` 3 interface.
    Qasm {
        /// Source line (1-indexed; 0 if not applicable).
        line: usize,
        /// Source column (1-indexed; 0 if not applicable).
        col: usize,
        /// Human-readable description.
        message: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QubitOutOfRange { qubit, num_qubits } => {
                write!(
                    f,
                    "qubit {qubit} out of range for {num_qubits}-qubit register"
                )
            }
            Self::ClassicalBitOutOfRange { bit, num_classical } => {
                write!(
                    f,
                    "classical bit {bit} out of range for {num_classical}-bit register"
                )
            }
            Self::DuplicateQubit { qubit } => {
                write!(f, "qubit {qubit} used more than once in a single gate")
            }
            Self::DimensionMismatch { len, expected } => {
                write!(f, "amplitude buffer has length {len}, expected {expected}")
            }
            Self::NonClifford { gate } => {
                write!(
                    f,
                    "{gate} is not in the Clifford group; the stabilizer backend cannot simulate it"
                )
            }
            Self::Qasm { line, col, message } => {
                write!(f, "OpenQASM 3 error at {line}:{col}: {message}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// A `Result` whose error is this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubit::{ClassicalBit, QubitId};

    #[test]
    fn qubit_out_of_range_message() {
        let e = Error::QubitOutOfRange {
            qubit: QubitId(3),
            num_qubits: 2,
        };
        assert_eq!(e.to_string(), "qubit q3 out of range for 2-qubit register");
    }

    #[test]
    fn classical_bit_out_of_range_message() {
        let e = Error::ClassicalBitOutOfRange {
            bit: ClassicalBit(5),
            num_classical: 3,
        };
        assert_eq!(
            e.to_string(),
            "classical bit c5 out of range for 3-bit register"
        );
    }

    #[test]
    fn duplicate_qubit_message() {
        let e = Error::DuplicateQubit { qubit: QubitId(1) };
        assert_eq!(
            e.to_string(),
            "qubit q1 used more than once in a single gate"
        );
    }

    #[test]
    fn dimension_mismatch_message() {
        let e = Error::DimensionMismatch {
            len: 3,
            expected: 4,
        };
        assert_eq!(e.to_string(), "amplitude buffer has length 3, expected 4");
    }

    #[test]
    fn non_clifford_message() {
        let e = Error::NonClifford { gate: "T" };
        assert!(e.to_string().contains("T"));
        assert!(e.to_string().contains("Clifford"));
    }

    #[test]
    fn qasm_error_message() {
        let e = Error::Qasm {
            line: 5,
            col: 10,
            message: "unexpected token".into(),
        };
        assert_eq!(e.to_string(), "OpenQASM 3 error at 5:10: unexpected token");
    }

    #[test]
    fn error_implements_std_error() {
        let e: &dyn std::error::Error = &Error::NonClifford { gate: "Toffoli" };
        assert!(e.source().is_none());
    }
}
