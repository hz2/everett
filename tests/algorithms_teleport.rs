//! Integration tests for quantum teleportation.
//!
//! Correctness approach: the Bloch vector of qubit 2 after teleportation must
//! equal the Bloch vector of the intended message. This is seed-independent
//! because the feed-forward corrections ensure all four measurement branches
//! yield the same final state on qubit 2.

use everett::Complex64;
use everett::algorithms::teleport;
use everett::prelude::*;

/// Run teleportation of the given state and return qubit 2's Bloch vector.
fn teleport_bloch(alpha: Complex64, beta: Complex64) -> [f64; 3] {
    let c = teleport::teleport_state(alpha, beta);
    let exec = StateVectorBackend::run(&c).unwrap();
    exec.state().bloch_vector(2)
}

/// expected Bloch vector for alpha|0>+beta|1>: (2Re(<0|rho|1>), 2Im(<0|rho|1>), |alpha|^2-|beta|^2)
fn expected_bloch(alpha: Complex64, beta: Complex64) -> [f64; 3] {
    let r01 = alpha * beta.conj();
    [
        2.0 * r01.re,
        2.0 * r01.im,
        alpha.norm_sqr() - beta.norm_sqr(),
    ]
}

fn assert_bloch_near(actual: [f64; 3], expected: [f64; 3], tol: f64) {
    for i in 0..3 {
        assert!(
            (actual[i] - expected[i]).abs() < tol,
            "Bloch[{i}]: got {}, expected {} (tol {tol})",
            actual[i],
            expected[i]
        );
    }
}

#[test]
fn teleport_zero() {
    let [x, y, z] = teleport_bloch(Complex64::ONE, Complex64::ZERO);
    assert!(x.abs() < 1e-10);
    assert!(y.abs() < 1e-10);
    assert!((z - 1.0).abs() < 1e-10);
}

#[test]
fn teleport_one() {
    let [x, y, z] = teleport_bloch(Complex64::ZERO, Complex64::ONE);
    assert!(x.abs() < 1e-10);
    assert!(y.abs() < 1e-10);
    assert!((z + 1.0).abs() < 1e-10);
}

#[test]
fn teleport_plus() {
    let s = 1.0_f64 / 2.0_f64.sqrt();
    let alpha = Complex64::new(s, 0.0);
    let beta = Complex64::new(s, 0.0);
    let bloch = teleport_bloch(alpha, beta);
    let exp = expected_bloch(alpha, beta);
    assert_bloch_near(bloch, exp, 1e-10);
}

#[test]
fn teleport_generic_ry_rz_state() {
    // ry(0.7) rz(1.1) applied to |0>: alpha = cos(0.35) * e^{-i*0.55}, beta = sin(0.35) * e^{i*0.55}
    let (sin_half, cos_half) = (0.7_f64 / 2.0).sin_cos();
    let alpha = Complex64::new(cos_half * (-0.55_f64).cos(), cos_half * (-0.55_f64).sin());
    let beta = Complex64::new(sin_half * 0.55_f64.cos(), sin_half * 0.55_f64.sin());
    let bloch = teleport_bloch(alpha, beta);
    let exp = expected_bloch(alpha, beta);
    assert_bloch_near(bloch, exp, 1e-10);
}

#[test]
fn teleport_is_seed_independent() {
    // the corrections make the final state identical regardless of measurement outcomes.
    // verify by running with several seeds and confirming qubit 2's Bloch vector is the same.
    let s = 1.0_f64 / 2.0_f64.sqrt();
    let alpha = Complex64::new(s, 0.0);
    let beta = Complex64::new(0.0, s);
    let c = teleport::teleport_state(alpha, beta);
    let exp = expected_bloch(alpha, beta);
    for seed in 0..8 {
        let exec = StateVectorBackend::run_seeded(&c, seed).unwrap();
        let bloch = exec.state().bloch_vector(2);
        assert_bloch_near(bloch, exp, 1e-10);
    }
}
