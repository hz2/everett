//! Worked, reusable circuits.
//!
//! Each submodule builds a [`crate::Circuit`] for a standard construction, so it
//! serves as both a library and as executable documentation. These are also the
//! integration-test oracles: their outputs have known closed forms (a Bell state
//! is maximally entangled, teleportation reproduces its input, the QFT matches
//! its analytic form, and so on).

pub mod hamiltonian;
pub mod phase_estimation;
pub mod prep;
pub mod qft;
pub mod superdense;
pub mod teleport;
