//! Type-safe index newtypes for qubits and classical bits.
//!
//! Both are thin wrappers over `usize`. Keeping them distinct stops a classical
//! bit index from being passed where a qubit index is expected, which is an easy
//! mistake once mid-circuit measurement enters the picture.

use std::fmt;

/// The index of a qubit within a register, counting from 0.
///
/// Qubit `0` is the least-significant bit of a basis state: in a 3-qubit system
/// the basis state `|q2 q1 q0>` has integer value `4*q2 + 2*q1 + q0`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct QubitId(pub usize);

/// The index of a classical bit, used as the destination of a measurement.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ClassicalBit(pub usize);

impl QubitId {
    /// Returns the underlying index.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl ClassicalBit {
    /// Returns the underlying index.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for QubitId {
    #[inline]
    fn from(i: usize) -> Self {
        Self(i)
    }
}

impl From<usize> for ClassicalBit {
    #[inline]
    fn from(i: usize) -> Self {
        Self(i)
    }
}

impl fmt::Display for QubitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "q{}", self.0)
    }
}

impl fmt::Display for ClassicalBit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "c{}", self.0)
    }
}
