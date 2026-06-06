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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qubit_id_index_roundtrips() {
        assert_eq!(QubitId(3).index(), 3);
    }

    #[test]
    fn classical_bit_index_roundtrips() {
        assert_eq!(ClassicalBit(7).index(), 7);
    }

    #[test]
    fn qubit_id_from_usize() {
        assert_eq!(QubitId::from(2), QubitId(2));
    }

    #[test]
    fn classical_bit_from_usize() {
        assert_eq!(ClassicalBit::from(5), ClassicalBit(5));
    }

    #[test]
    fn qubit_id_display() {
        assert_eq!(format!("{}", QubitId(4)), "q4");
    }

    #[test]
    fn classical_bit_display() {
        assert_eq!(format!("{}", ClassicalBit(2)), "c2");
    }
}
