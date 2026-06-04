//! Demonstrates Hamiltonian simulation and Trotter convergence.
//!
//! Shows that increasing steps improves the fidelity of the transverse-field
//! Ising simulation against a high-step reference.

use everett::StateVectorBackend;
use everett::algorithms::hamiltonian::{trotter_single_z, trotter_tfim};

fn main() {
    println!("=== single-qubit Z rotation (trotter_single_z) ===");
    println!("t=1.0, steps=1..10: norm should always be 1.0");
    for steps in 1..=10 {
        let state = StateVectorBackend::run(&trotter_single_z(1.0, steps))
            .unwrap()
            .into_state();
        println!("  steps={steps}  norm_sqr={:.15}", state.norm_sqr());
    }

    println!();
    println!("=== TFIM convergence (n=2, Jzz=1.0, hx=0.5, t=0.5) ===");
    let (n, jzz, hx, t) = (2, 1.0, 0.5, 0.5);
    // high-fidelity reference
    let reference = StateVectorBackend::run(&trotter_tfim(n, jzz, hx, t, 2000))
        .unwrap()
        .into_state();

    println!("{:>6}  {:>12}  {:>12}", "steps", "fidelity", "norm_sqr");
    for steps in [1, 5, 10, 50, 100, 500] {
        let state = StateVectorBackend::run(&trotter_tfim(n, jzz, hx, t, steps))
            .unwrap()
            .into_state();
        println!(
            "{steps:>6}  {:.10}  {:.12}",
            state.fidelity(&reference),
            state.norm_sqr()
        );
    }
}
