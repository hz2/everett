//! Quantum teleportation.
//!
//! Teleportation moves an unknown qubit state from qubit 0 to qubit 2 using a
//! shared Bell pair (qubits 1 and 2) and two bits of classical communication.
//! The circuit contains mid-circuit measurements and classically-controlled
//! corrections ($X$ and $Z$), so it exercises the full instruction set.

use crate::circuit::Circuit;
use crate::complex::Complex64;

/// Returns the core teleportation circuit: 3 qubits, 2 classical bits.
///
/// - Qubit 0: the message qubit (caller prepares this before running).
/// - Qubits 1, 2: Alice and Bob's halves of the Bell pair.
/// - Classical bit 0: outcome of measuring qubit 0.
/// - Classical bit 1: outcome of measuring qubit 1.
///
/// After the circuit runs, qubit 2 holds whatever state qubit 0 was in at the
/// start, regardless of measurement outcomes. Use [`teleport_state`] for the
/// common case of preparing and teleporting in one step.
///
/// ## Protocol
///
/// 1. Prepare Bell pair on (1, 2): H on 1 then CNOT(1,2).
/// 2. Bell measurement on (0, 1): CNOT(0,1), H on 0, then measure both.
/// 3. Feed-forward: X on qubit 2 if classical bit 1 is set; Z on qubit 2 if
///    classical bit 0 is set.
///
/// # Examples
///
/// ```
/// use everett::prelude::*;
/// use everett::algorithms::teleport;
///
/// // teleport the |+> state
/// let circuit = teleport::teleport_state(
///     Complex64::new(1.0 / 2f64.sqrt(), 0.0),
///     Complex64::new(1.0 / 2f64.sqrt(), 0.0),
/// );
/// let exec = StateVectorBackend::run(&circuit).unwrap();
/// // qubit 2 should now be |+>; applying H and checking |0> verifies it.
/// // (see integration tests for the full correctness check)
/// ```
#[must_use]
pub fn teleport_circuit() -> Circuit {
    let mut c = Circuit::with_classical(3, 2);
    // prepare Bell pair on (1, 2)
    c.h(1).cnot(1, 2);
    // Bell basis measurement of (0, 1)
    c.cnot(0, 1).h(0);
    c.measure(0, 0).measure(1, 1);
    // classically-controlled corrections: X if c1=1, Z if c0=1
    c.x_if(1, 2).z_if(0, 2);
    c
}

/// Returns a 3-qubit circuit that first prepares qubit 0 in
/// $\alpha|0\rangle + \beta|1\rangle$ and then teleports it to qubit 2.
///
/// The message is prepared using a single custom gate built from the supplied
/// amplitudes. The caller is responsible for ensuring $|\alpha|^2 + |\beta|^2
/// \approx 1$.
///
/// # Examples
///
/// ```
/// use everett::prelude::*;
/// use everett::algorithms::teleport;
///
/// // teleport |1>
/// let circuit = teleport::teleport_state(Complex64::ZERO, Complex64::ONE);
/// let exec = StateVectorBackend::run(&circuit).unwrap();
/// // qubit 2 should be in state |1>, with Bloch vector (0, 0, -1).
/// let [x, y, z] = exec.state().bloch_vector(2);
/// assert!(x.abs() < 1e-10 && y.abs() < 1e-10);
/// assert!((z + 1.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn teleport_state(alpha: Complex64, beta: Complex64) -> Circuit {
    // build a single-qubit gate whose first column is (alpha, beta): applying
    // it to |0> yields alpha|0> + beta|1>.  the second column only needs to be
    // orthogonal so the gate is unitary; we use (-conj(beta), conj(alpha)).
    let prep = crate::Gate1::new(alpha, -beta.conj(), beta, alpha.conj());
    let mut c = Circuit::with_classical(3, 2);
    c.gate1(prep, 0);
    c.compose(&teleport_circuit());
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateVectorBackend;

    // check teleportation of |1>: qubit 2 Bloch vector should be (0, 0, -1).
    #[test]
    fn teleport_one_state() {
        let c = teleport_state(Complex64::ZERO, Complex64::ONE);
        let exec = StateVectorBackend::run(&c).unwrap();
        let [x, y, z] = exec.state().bloch_vector(2);
        assert!(x.abs() < 1e-10, "x={x}");
        assert!(y.abs() < 1e-10, "y={y}");
        assert!((z + 1.0).abs() < 1e-10, "z={z}");
    }

    #[test]
    fn teleport_zero_state() {
        let c = teleport_state(Complex64::ONE, Complex64::ZERO);
        let exec = StateVectorBackend::run(&c).unwrap();
        let [x, y, z] = exec.state().bloch_vector(2);
        assert!(x.abs() < 1e-10, "x={x}");
        assert!(y.abs() < 1e-10, "y={y}");
        assert!((z - 1.0).abs() < 1e-10, "z={z}");
    }
}
