//! Superdense coding example: encode and decode all four bit pairs.

use everett::algorithms::superdense;
use everett::prelude::*;

fn main() -> everett::Result<()> {
    println!("Superdense coding — all four bit pairs:");
    for bit0 in [false, true] {
        for bit1 in [false, true] {
            let exec = StateVectorBackend::run(&superdense::superdense_circuit(bit0, bit1))?;
            let c = exec.classical();
            let ok = c[0] == bit0 && c[1] == bit1;
            println!(
                "  sent ({}, {})  decoded ({}, {})  {}",
                u8::from(bit0),
                u8::from(bit1),
                u8::from(c[0]),
                u8::from(c[1]),
                if ok { "✓" } else { "✗" }
            );
        }
    }
    Ok(())
}
