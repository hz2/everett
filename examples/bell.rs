//! Bell state example: runs the Bell circuit and prints measurement probabilities.

use everett::algorithms::prep;
use everett::prelude::*;

fn main() -> everett::Result<()> {
    let exec = StateVectorBackend::run(&prep::bell())?;
    let s = exec.state();
    println!("Bell state probabilities:");
    println!("  P(|00>) = {:.6}", s.probability(0b00));
    println!("  P(|01>) = {:.6}", s.probability(0b01));
    println!("  P(|10>) = {:.6}", s.probability(0b10));
    println!("  P(|11>) = {:.6}", s.probability(0b11));
    Ok(())
}
