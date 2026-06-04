//! Demonstrates quantum phase estimation.
//!
//! Estimates several eigenphases and prints the recovered values.

use everett::StateVectorBackend;
use everett::algorithms::phase_estimation::{most_probable_phase, phase_estimation};

fn main() {
    let cases: &[(usize, f64)] = &[
        (1, 0.5),
        (2, 0.25),
        (2, 0.75),
        (3, 0.125),
        (4, 0.3), // non-dyadic; approximate
        (4, 0.7),
    ];

    println!(
        "{:>3}  {:>8}  {:>12}  {:>10}",
        "m", "phi", "estimated", "error"
    );
    for &(m, phi) in cases {
        let c = phase_estimation(m, phi);
        let exec = StateVectorBackend::run(&c).unwrap();
        let est = most_probable_phase(exec.state(), m);
        println!(
            "{m:>3}  {phi:>8.4}  {est:>12.6}  {:>10.6}",
            (est - phi).abs()
        );
    }
}
