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
        }
    }
}

impl std::error::Error for Error {}

/// A `Result` whose error is this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
