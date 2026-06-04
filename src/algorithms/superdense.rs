//! Superdense coding.
//!
//! Alice encodes two classical bits into one qubit of a shared Bell pair, then
//! sends it to Bob. Bob decodes both bits using a Bell measurement. The circuit
//! encodes and decodes, and includes measurement into 2 classical bits.

use crate::Circuit;

/// Returns a 2-qubit circuit that encodes `(bit0, bit1)` via superdense coding
/// and decodes them, producing the two classical bits in `c0` and `c1`.
///
/// The protocol:
/// 1. Prepare Bell pair: $H(0)$, $\text{CNOT}(0,1)$.
/// 2. Alice's encoding on qubit 0:
///    - $(0,0) \to I$ (do nothing)
///    - $(0,1) \to X$
///    - $(1,0) \to Z$
///    - $(1,1) \to iY = ZX$
/// 3. Bob's decoding: $\text{CNOT}(0,1)$, $H(0)$.
/// 4. Measure qubit 0 into classical bit 0, qubit 1 into classical bit 1.
///
/// After running, `execution.classical()[0] == bit0` and
/// `execution.classical()[1] == bit1`.
///
/// # Examples
///
/// ```
/// use everett::prelude::*;
/// use everett::algorithms::superdense;
///
/// let exec = StateVectorBackend::run(&superdense::superdense_circuit(true, false)).unwrap();
/// assert_eq!(exec.classical(), &[true, false]);
/// ```
#[must_use]
pub fn superdense_circuit(bit0: bool, bit1: bool) -> Circuit {
    let mut c = Circuit::with_classical(2, 2);
    // prepare Bell pair
    c.h(0).cnot(0, 1);
    // Alice encodes on qubit 0
    if bit1 {
        c.x(0);
    }
    if bit0 {
        c.z(0);
    }
    // Bob decodes
    c.cnot(0, 1).h(0);
    // measure both qubits
    c.measure(0, 0).measure(1, 1);
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateVectorBackend;

    fn check(bit0: bool, bit1: bool) {
        let exec = StateVectorBackend::run(&superdense_circuit(bit0, bit1)).unwrap();
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
    fn all_four_bit_pairs_decode_correctly() {
        check(false, false);
        check(false, true);
        check(true, false);
        check(true, true);
    }
}
