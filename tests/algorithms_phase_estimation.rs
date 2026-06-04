//! Integration tests for quantum phase estimation.

use everett::StateVectorBackend;
use everett::algorithms::phase_estimation::{most_probable_phase, phase_estimation};

fn estimate(m: usize, phi: f64) -> f64 {
    let c = phase_estimation(m, phi);
    let exec = StateVectorBackend::run(&c).unwrap();
    most_probable_phase(exec.state(), m)
}

#[test]
fn dyadic_fractions_recovered_exactly() {
    // phi = k/2^m is exactly representable by m counting qubits
    assert!((estimate(1, 0.5) - 0.5).abs() < 1e-10);
    assert!((estimate(2, 0.25) - 0.25).abs() < 1e-10);
    assert!((estimate(2, 0.75) - 0.75).abs() < 1e-10);
    assert!((estimate(3, 0.125) - 0.125).abs() < 1e-10);
    assert!((estimate(3, 0.875) - 0.875).abs() < 1e-10);
}

#[test]
fn non_dyadic_phase_is_approximated() {
    // phi = 0.3; nearest 4-bit dyadic is round(0.3*16)/16 = 5/16 = 0.3125
    let est = estimate(4, 0.3);
    assert!(
        (est - 0.3).abs() <= 1.0 / 16.0 + 1e-10,
        "expected within 1/16 of 0.3, got {est}"
    );
}

#[test]
fn phi_zero_recovers_zero() {
    // e^{0} is the identity; the counting register stays |0...0⟩
    let est = estimate(4, 0.0);
    assert!((est - 0.0).abs() < 1e-10, "got {est}");
}
