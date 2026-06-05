//! # everett
//!
//! A clean, zero-dependency statevector quantum simulator.
//!
//! `everett` represents the state of `n` qubits as a complex vector of `2^n`
//! amplitudes and applies gates in place over amplitude index-groups — never by
//! materializing a `2^n x 2^n` matrix. You describe a computation by building a
//! [`Circuit`], then run it on a backend such as [`StateVectorBackend`].
//!
//! ## Example: a Bell state
//!
//! ```
//! use everett::prelude::*;
//!
//! let mut circuit = Circuit::new(2);
//! circuit.h(0).cnot(0, 1);
//!
//! let exec = StateVectorBackend::run(&circuit)?;
//! let state = exec.state();
//! assert!((state.probability(0b00) - 0.5).abs() < 1e-12);
//! assert!((state.probability(0b11) - 0.5).abs() < 1e-12);
//! # Ok::<(), everett::Error>(())
//! ```
//!
//! ## Layout
//!
//! - [`Complex64`] — the amplitude scalar.
//! - [`State`] — the statevector and its observables.
//! - [`Gate1`], [`Gate2`] — gate matrices.
//! - [`Circuit`] — the fluent builder.
//! - [`StateVectorBackend`] — the dense execution backend.
//! - [`algorithms`] — worked, reusable circuits (Bell, GHZ, teleportation, …).

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod algorithms;
mod backend;
mod circuit;
mod complex;
mod error;
mod gate;
pub mod kernel;
mod measure;
mod op;
mod qubit;
mod rng;
mod state;

pub use backend::{
    Execution, PauliString, StabilizerBackend, StabilizerExecution, StateVectorBackend,
};
pub use circuit::Circuit;
pub use complex::Complex64;
pub use error::{Error, Result};
pub use gate::{Gate1, Gate2};
pub use op::Op;
pub use qubit::{ClassicalBit, QubitId};
pub use rng::Rng;
pub use state::State;

/// The backend trait that execution targets implement.
pub use backend::Backend;

/// Common imports for everyday use: `use everett::prelude::*;`.
pub mod prelude {
    pub use crate::circuit::Circuit;
    pub use crate::complex::Complex64;
    pub use crate::gate::{Gate1, Gate2};
    pub use crate::state::State;
    pub use crate::{Backend, Execution, StateVectorBackend};
}
