//! Quantum phase estimation (QPE).
// LaTeX math notation is intentional in this module's docs.
#![allow(clippy::doc_markdown)]
//!
//! # Overview
//!
//! Given a unitary $U$ and an eigenstate $|u\rangle$ satisfying
//! $U|u\rangle = e^{2\pi i\phi}|u\rangle$, phase estimation produces an
//! $m$-bit approximation of $\phi \in [0, 1)$ in the counting register.
//!
//! # Circuit layout
//!
//! This module uses $U = \text{phase}(2\pi\phi)$, whose sole eigenstate is
//! $|1\rangle$ with eigenvalue $e^{2\pi i\phi}$. The circuit layout is:
//!
//! - Qubits $0 \ldots m-1$: counting register (initialised in $|0\rangle$).
//! - Qubit $m$: eigenstate target (prepared as $|1\rangle$).
//!
//! Steps:
//! 1. Prepare target: $X$ on qubit $m$.
//! 2. Hadamard on each counting qubit $j$.
//! 3. For each counting qubit $j$: controlled-$U^{2^j}$ on the target,
//!    i.e. `controlled([j], phase(2π·φ·2^j), m)`.
//! 4. Apply the inverse QFT on qubits $0\ldots m-1$.
//!
//! After running the circuit and measuring the counting register, the observed
//! integer $s$ satisfies $s/2^m \approx \phi$.

use std::f64::consts::TAU;

use crate::algorithms::qft;
use crate::circuit::Circuit;
use crate::gate::Gate1;
use crate::state::State;

/// Returns an $m$-qubit phase estimation circuit for a phase gate with
/// eigenphase $\phi$.
///
/// Qubit layout: counting qubits $0\ldots m-1$, target qubit $m$.
/// After running the circuit, use [`most_probable_phase`] to read off $\phi$.
///
/// # Examples
///
/// ```
/// use everett::{StateVectorBackend, algorithms::phase_estimation};
///
/// // phi = 0.5 is exactly representable with 1 counting qubit.
/// let c = phase_estimation::phase_estimation(1, 0.5);
/// let exec = StateVectorBackend::run(&c).unwrap();
/// let phi_est = phase_estimation::most_probable_phase(exec.state(), 1);
/// assert!((phi_est - 0.5).abs() < 1e-12);
/// ```
#[must_use]
pub fn phase_estimation(counting_qubits: usize, phi: f64) -> Circuit {
    let m = counting_qubits;
    let total = m + 1; // counting register + 1 eigenstate qubit
    let target = m; // the eigenstate qubit index

    let mut c = Circuit::new(total);

    // step 1: prepare eigenstate |1⟩
    c.x(target);

    // step 2: uniform superposition over counting register
    for j in 0..m {
        c.h(j);
    }

    // step 3: controlled-U^{2^j} for each counting qubit j.
    // U = phase(2π·φ), so U^{2^j} = phase(2π·φ·2^j).
    for j in 0..m {
        let angle = TAU * phi * f64::from(1u32 << j);
        c.controlled(&[j], Gate1::phase(angle), target);
    }

    // step 4: inverse QFT on the counting register.
    // iqft(m) acts on qubits 0..m, which are exactly our counting register.
    let iqft_circ = qft::iqft(m);
    // compose only works if the sub-circuit size <= our circuit size, which
    // holds since iqft(m) has m qubits and we have m+1.
    c.compose(&iqft_circ);

    c
}

/// Reads the most-probable phase estimate from the counting register.
///
/// Marginalises over the target qubit (qubit index `counting_qubits`) and
/// returns the most probable counting-register value divided by $2^m$, which
/// approximates $\phi$.
///
/// # Panics
///
/// Panics if `counting_qubits == 0` or if the state dimension is smaller than
/// `2^counting_qubits`.
#[must_use]
pub fn most_probable_phase(state: &State, counting_qubits: usize) -> f64 {
    let m = counting_qubits;
    let counting_states = 1usize << m;

    // marginalise over the target qubit: sum probabilities over target=0 and target=1.
    let mut marginal = vec![0.0f64; counting_states];
    for (basis, &amp) in state.amplitudes().iter().enumerate() {
        let counting_index = basis & (counting_states - 1); // low m bits
        marginal[counting_index] += amp.norm_sqr();
    }

    // find the most probable counting-register value
    let best = marginal
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    best as f64 / counting_states as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateVectorBackend;

    fn estimate(m: usize, phi: f64) -> f64 {
        let c = phase_estimation(m, phi);
        let exec = StateVectorBackend::run(&c).unwrap();
        most_probable_phase(exec.state(), m)
    }

    #[test]
    fn dyadic_phases_recovered_exactly() {
        // phi = k/2^m is exactly representable; recovery must be exact.
        assert!((estimate(1, 0.5) - 0.5).abs() < 1e-10);
        assert!((estimate(2, 0.25) - 0.25).abs() < 1e-10);
        assert!((estimate(2, 0.75) - 0.75).abs() < 1e-10);
        assert!((estimate(3, 0.125) - 0.125).abs() < 1e-10);
        assert!((estimate(3, 0.375) - 0.375).abs() < 1e-10);
    }

    #[test]
    fn non_dyadic_phase_approximated() {
        // phi = 0.3 is not dyadic; with m=4 we get the nearest multiple of 1/16.
        // nearest: round(0.3 * 16) / 16 = 5/16 = 0.3125
        let est = estimate(4, 0.3);
        assert!((est - 0.3).abs() < 0.1, "expected ~0.3, got {est}");
    }
}
