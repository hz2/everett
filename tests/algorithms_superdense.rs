#![allow(missing_docs)]
use everett::algorithms::superdense;
use everett::prelude::*;

fn check_decode(bit0: bool, bit1: bool) {
    let exec = StateVectorBackend::run(&superdense::superdense_circuit(bit0, bit1)).unwrap();
    assert_eq!(
        exec.classical()[0],
        bit0,
        "bit0 mismatch for ({bit0},{bit1})"
    );
    assert_eq!(
        exec.classical()[1],
        bit1,
        "bit1 mismatch for ({bit0},{bit1})"
    );
}

#[test]
fn superdense_00() {
    check_decode(false, false);
}

#[test]
fn superdense_01() {
    check_decode(false, true);
}

#[test]
fn superdense_10() {
    check_decode(true, false);
}

#[test]
fn superdense_11() {
    check_decode(true, true);
}
