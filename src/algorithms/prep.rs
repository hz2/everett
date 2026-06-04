//! Entangled-state preparation: Bell pairs and GHZ states.

use crate::Circuit;

/// Returns a 2-qubit circuit that prepares the Bell state
/// $(|00\rangle + |11\rangle)/\sqrt{2}$ from $|00\rangle$.
///
/// Qubit 0 is the least-significant bit, so the entangled pair spans qubits
/// 0 and 1.
///
/// # Examples
///
/// ```
/// use everett::prelude::*;
/// use everett::algorithms::prep;
///
/// let exec = StateVectorBackend::run(&prep::bell()).unwrap();
/// let s = exec.state();
/// assert!((s.probability(0b00) - 0.5).abs() < 1e-12);
/// assert!((s.probability(0b11) - 0.5).abs() < 1e-12);
/// ```
#[must_use]
pub fn bell() -> Circuit {
    let mut c = Circuit::new(2);
    c.h(0).cnot(0, 1);
    c
}

/// Returns an `n`-qubit circuit that prepares the GHZ state
/// $(|0\cdots 0\rangle + |1\cdots 1\rangle)/\sqrt{2}$ from $|0\cdots 0\rangle$.
///
/// The construction is $H$ on qubit 0 followed by a chain of CNOTs:
/// $\text{CNOT}(0,1), \text{CNOT}(1,2), \ldots, \text{CNOT}(n-2, n-1)$.
///
/// # Panics
///
/// Panics if `n < 2`.
///
/// # Examples
///
/// ```
/// use everett::prelude::*;
/// use everett::algorithms::prep;
///
/// let exec = StateVectorBackend::run(&prep::ghz(3)).unwrap();
/// let s = exec.state();
/// assert!((s.probability(0b000) - 0.5).abs() < 1e-12);
/// assert!((s.probability(0b111) - 0.5).abs() < 1e-12);
/// ```
#[must_use]
pub fn ghz(n: usize) -> Circuit {
    assert!(n >= 2, "GHZ state requires at least 2 qubits");
    let mut c = Circuit::new(n);
    c.h(0);
    for k in 0..n - 1 {
        c.cnot(k, k + 1);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateVectorBackend;

    #[test]
    fn bell_has_correct_probabilities() {
        let exec = StateVectorBackend::run(&bell()).unwrap();
        let s = exec.state();
        assert!((s.probability(0b00) - 0.5).abs() < 1e-12);
        assert!((s.probability(0b11) - 0.5).abs() < 1e-12);
        assert!(s.probability(0b01) < 1e-12);
        assert!(s.probability(0b10) < 1e-12);
    }

    #[test]
    fn ghz3_has_correct_probabilities() {
        let exec = StateVectorBackend::run(&ghz(3)).unwrap();
        let s = exec.state();
        assert!((s.probability(0b000) - 0.5).abs() < 1e-12);
        assert!((s.probability(0b111) - 0.5).abs() < 1e-12);
        // all other basis states should be zero
        for b in 1..=6 {
            assert!(s.probability(b) < 1e-12, "unexpected amplitude at {b}");
        }
    }
}
