//! Demonstrates the quantum Fourier transform on a basis state.
//!
//! Runs QFT on |1> (3 qubits) and prints the output amplitudes, which should
//! match the closed form e^{2*pi*i*k/8}/sqrt(8) for k=0..7.

use everett::StateVectorBackend;
use everett::algorithms::qft;
use std::f64::consts::TAU;

fn main() {
    let n = 3;
    let input_basis = 1usize; // |001> = |1>

    let mut c = everett::Circuit::new(n);
    // prepare |input_basis>
    for bit in 0..n {
        if input_basis & (1 << bit) != 0 {
            c.x(bit);
        }
    }
    c.compose(&qft::qft(n));

    let state = StateVectorBackend::run(&c).unwrap().into_state();
    let nn = 1 << n;
    let scale = 1.0 / (nn as f64).sqrt();

    println!("QFT |{input_basis}> on {n} qubits (N={nn}):");
    println!(
        "{:>4}  {:>24}  {:>18}  {:>18}",
        "k", "amplitude", "expected_re", "expected_im"
    );
    for k in 0..nn {
        let amp = state.amplitudes()[k];
        let expected_re = scale * (TAU * k as f64 / nn as f64).cos();
        let expected_im = scale * (TAU * k as f64 / nn as f64).sin();
        println!(
            "{k:>4}  {:.6}+{:.6}i  {:.6}          {:.6}",
            amp.re, amp.im, expected_re, expected_im
        );
    }
}
