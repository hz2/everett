//! The quantum Fourier transform (QFT) and its inverse.
// the module-level docs use LaTeX math notation ($F_n$, $R_m$, etc.) which
// clippy::doc_markdown would flag as missing backticks. the LaTeX is intentional.
#![allow(clippy::doc_markdown)]
//!
//! # Mathematical definition
//!
//! The $n$-qubit QFT is the unitary $F_n$ defined by
//!
//! $$F_n |j\rangle = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} e^{2\pi i jk/2^n} |k\rangle$$
//!
//! It is the discrete Fourier transform over $\mathbb{Z}_{2^n}$.
//!
//! # Circuit decomposition
//!
//! This implementation follows the standard textbook decomposition
//! (Nielsen & Chuang §5.1). Qubits are indexed $0$ (LSB) through $n-1$ (MSB).
//! The circuit processes qubits from MSB down to LSB:
//!
//! For qubit $j$ (descending from $n-1$ to $0$):
//! 1. Apply $H$ to qubit $j$.
//! 2. For each qubit $k < j$: apply $R_{j-k+1}$ controlled on qubit $k$,
//!    where $R_m = \text{phase}(2\pi / 2^m)$.
//! 3. After all qubits are processed, SWAP qubits to reverse the bit order
//!    (qubits $0 \leftrightarrow n-1$, $1 \leftrightarrow n-2$, …).
//!
//! The SWAP reversal is included so that the circuit's output matches
//! the standard $|k\rangle$ ordering with qubit 0 as the LSB of $k$.

use std::f64::consts::TAU; // 2π

use crate::circuit::Circuit;

/// Returns the $n$-qubit quantum Fourier transform circuit.
///
/// Qubits in the returned circuit are numbered $0$ (LSB) to $n-1$ (MSB). The
/// circuit includes the final bit-reversal SWAPs, so the output state has
/// the same qubit-index convention as the input.
///
/// # Examples
///
/// ```
/// use everett::{StateVectorBackend, algorithms::qft};
///
/// // QFT of |0⟩ is uniform superposition.
/// let circuit = qft::qft(2);
/// let exec = StateVectorBackend::run(&circuit).unwrap();
/// let state = exec.state();
/// for basis in 0..4 {
///     assert!((state.probability(basis) - 0.25).abs() < 1e-12);
/// }
/// ```
#[must_use]
pub fn qft(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    apply_qft_no_swap(&mut c, n);
    // reverse qubit order: swap qubit i with qubit n-1-i
    for i in 0..n / 2 {
        c.swap(i, n - 1 - i);
    }
    c
}

/// Returns the inverse QFT circuit ($F_n^\dagger$).
///
/// Undoes the QFT: applying `qft(n)` then `iqft(n)` restores the original state.
///
/// # Examples
///
/// ```
/// use everett::{StateVectorBackend, State, algorithms::qft};
///
/// let n = 3;
/// let mut c = qft::qft(n);
/// c.compose(&qft::iqft(n));
/// let exec = StateVectorBackend::run(&c).unwrap();
/// // |0⟩ is an eigenstate of QFT∘IQFT, so we recover it.
/// assert!((exec.state().probability(0) - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn iqft(n: usize) -> Circuit {
    // the inverse is the reverse circuit with negated rotation angles:
    // undo SWAPs first, then apply the adjoint of each layer in reverse order.
    let mut c = Circuit::new(n);
    // reverse the bit-reversal SWAPs
    for i in 0..n / 2 {
        c.swap(i, n - 1 - i);
    }
    apply_iqft_no_swap(&mut c, n);
    c
}

// appends the QFT body (without final SWAPs) to `c`.
fn apply_qft_no_swap(c: &mut Circuit, n: usize) {
    // process qubits MSB → LSB
    for j in (0..n).rev() {
        c.h(j);
        // controlled rotations: R_{j-k+1} controlled by qubit k < j
        for k in (0..j).rev() {
            let m = (j - k + 1) as u32;
            let angle = TAU / f64::from(1u32 << m);
            c.controlled(&[k], crate::gate::Gate1::phase(angle), j);
        }
    }
}

// appends the inverse QFT body (without the leading SWAP reversal) to `c`.
fn apply_iqft_no_swap(c: &mut Circuit, n: usize) {
    // mirror of apply_qft_no_swap: inner loops first, then H, angles negated.
    for j in 0..n {
        for k in 0..j {
            let m = (j - k + 1) as u32;
            let angle = -TAU / f64::from(1u32 << m);
            c.controlled(&[k], crate::gate::Gate1::phase(angle), j);
        }
        c.h(j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateVectorBackend;
    use crate::complex::Complex64;
    use crate::state::State;

    fn run(c: &Circuit) -> State {
        StateVectorBackend::run(c).unwrap().into_state()
    }

    #[test]
    fn qft_of_zero_is_uniform() {
        for n in 1..=5 {
            let state = run(&qft(n));
            let expected = 1.0 / f64::from(1u32 << n);
            for basis in 0..(1usize << n) {
                assert!(
                    (state.probability(basis) - expected).abs() < 1e-12,
                    "n={n} basis={basis}"
                );
            }
        }
    }

    #[test]
    fn qft_then_iqft_is_identity() {
        // check on a few basis states for n=3
        let n = 3;
        for input_basis in [0, 1, 3, 5, 7] {
            // prepare |input_basis> by building the amplitude vector directly
            let mut amps = vec![Complex64::ZERO; 1 << n];
            amps[input_basis] = Complex64::ONE;
            let input = State::from_amplitudes(amps.clone()).unwrap();

            // build a circuit that starts in |input_basis> using X gates
            let mut c = Circuit::new(n);
            for bit in 0..n {
                if input_basis & (1 << bit) != 0 {
                    c.x(bit);
                }
            }
            c.compose(&qft(n)).compose(&iqft(n));
            let output = run(&c);

            assert!(output.fidelity(&input) > 1.0 - 1e-12, "basis={input_basis}");
        }
    }
}
