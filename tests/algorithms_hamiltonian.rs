//! Integration tests for Hamiltonian simulation.

use everett::algorithms::hamiltonian::{trotter_single_z, trotter_tfim};
use everett::{Circuit, StateVectorBackend};

fn norm_after(c: &Circuit) -> f64 {
    StateVectorBackend::run(c).unwrap().into_state().norm_sqr()
}

#[test]
fn single_z_norm_preserved_all_step_counts() {
    for steps in [1, 3, 10, 100] {
        assert!(
            (norm_after(&trotter_single_z(0.7, steps)) - 1.0).abs() < 1e-12,
            "steps={steps}"
        );
    }
}

#[test]
fn tfim_norm_preserved() {
    let c = trotter_tfim(4, 1.0, 0.5, 1.0, 100);
    assert!((norm_after(&c) - 1.0).abs() < 1e-12);
}

#[test]
fn pure_x_field_matches_exact_rx() {
    // H = -hx X, so e^{-iHt} = e^{i hx t X} = Rx(-2 hx t)
    let hx = 0.8;
    let t = 0.4;

    let trotter_state = StateVectorBackend::run(&trotter_tfim(1, 0.0, hx, t, 500))
        .unwrap()
        .into_state();

    let mut exact = Circuit::new(1);
    exact.rx(0, -2.0 * hx * t);
    let exact_state = StateVectorBackend::run(&exact).unwrap().into_state();

    assert!(
        trotter_state.fidelity(&exact_state) > 0.9999,
        "fidelity = {}",
        trotter_state.fidelity(&exact_state)
    );
}

#[test]
fn trotter_converges_with_more_steps() {
    let (jzz, hx, t) = (1.0, 0.5, 0.5);
    let coarse = StateVectorBackend::run(&trotter_tfim(2, jzz, hx, t, 10))
        .unwrap()
        .into_state();
    let fine = StateVectorBackend::run(&trotter_tfim(2, jzz, hx, t, 1000))
        .unwrap()
        .into_state();
    let f = coarse.fidelity(&fine);
    assert!(f > 0.99, "coarse vs fine fidelity = {f}");
}
