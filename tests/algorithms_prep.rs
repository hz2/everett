#![allow(missing_docs)]
use everett::algorithms::prep;
use everett::prelude::*;

#[test]
fn bell_probabilities() {
    let exec = StateVectorBackend::run(&prep::bell()).unwrap();
    let s = exec.state();
    assert!((s.probability(0b00) - 0.5).abs() < 1e-12);
    assert!((s.probability(0b11) - 0.5).abs() < 1e-12);
    assert!(s.probability(0b01) < 1e-12);
    assert!(s.probability(0b10) < 1e-12);
}

#[test]
fn bell_is_normalized() {
    let exec = StateVectorBackend::run(&prep::bell()).unwrap();
    assert!((exec.state().norm_sqr() - 1.0).abs() < 1e-12);
}

#[test]
fn ghz3_probabilities() {
    let exec = StateVectorBackend::run(&prep::ghz(3)).unwrap();
    let s = exec.state();
    assert!((s.probability(0b000) - 0.5).abs() < 1e-12);
    assert!((s.probability(0b111) - 0.5).abs() < 1e-12);
    for b in 1..=6 {
        assert!(s.probability(b) < 1e-12);
    }
}

#[test]
fn ghz4_probabilities() {
    let exec = StateVectorBackend::run(&prep::ghz(4)).unwrap();
    let s = exec.state();
    assert!((s.probability(0b0000) - 0.5).abs() < 1e-12);
    assert!((s.probability(0b1111) - 0.5).abs() < 1e-12);
    // all other 14 basis states zero
    for b in 1..=14 {
        assert!(s.probability(b) < 1e-12);
    }
}

#[test]
fn ghz5_probabilities() {
    let exec = StateVectorBackend::run(&prep::ghz(5)).unwrap();
    let s = exec.state();
    assert!((s.probability(0b00000) - 0.5).abs() < 1e-12);
    assert!((s.probability(0b11111) - 0.5).abs() < 1e-12);
}

#[test]
fn ghz_qubits_are_maximally_mixed() {
    // each qubit's Bloch vector should lie near the origin (maximally entangled)
    let exec = StateVectorBackend::run(&prep::ghz(3)).unwrap();
    let state = exec.state();
    for qubit in 0..3 {
        let [bx, by, bz] = state.bloch_vector(qubit);
        let radius = (bx * bx + by * by + bz * bz).sqrt();
        assert!(
            radius < 0.01,
            "qubit {qubit} Bloch radius {radius} is not near zero"
        );
    }
}
