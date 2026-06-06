//! Gate name table: bidirectional mapping between Rust gate values and
//! `OpenQASM` 3 stdgates.inc names.

use crate::complex::Complex64;
use crate::gate::{Gate1, Gate2};
use crate::{Error, Result};

// matrices reconstructed from an extracted angle differ from the original by a
// rounding step (atan2 then sin_cos is not bit-exact), so verify with a
// tolerance rather than `==`.
fn gate1_close(a: &Gate1, b: &Gate1) -> bool {
    (0..4).all(|k| (a.m[k] - b.m[k]).norm() < 1e-12)
}

fn no_name() -> Error {
    Error::Qasm {
        line: 0,
        col: 0,
        message: "gate has no OpenQASM 3 name; use a named gate constructor".into(),
    }
}

/// Returns `(name, formatted_arg_string)` for a `Gate1`.
///
/// `arg_string` is empty for no-parameter gates; for parametric gates it is
/// `"(theta)"` with the angle formatted to 17 significant digits for exact f64
/// round-trip.
pub fn gate_info_1q(gate: &Gate1) -> Result<(&'static str, String)> {
    // no-parameter gates: exact matrix equality.
    if *gate == Gate1::x() {
        return Ok(("x", String::new()));
    }
    if *gate == Gate1::y() {
        return Ok(("y", String::new()));
    }
    if *gate == Gate1::z() {
        return Ok(("z", String::new()));
    }
    if *gate == Gate1::h() {
        return Ok(("h", String::new()));
    }
    if *gate == Gate1::s() {
        return Ok(("s", String::new()));
    }
    if *gate == Gate1::t() {
        return Ok(("t", String::new()));
    }
    if *gate == Gate1::id() {
        return Ok(("id", String::new()));
    }

    // parametric gates: check the structural form, extract the angle, verify.
    // rx(theta): [[cos t/2, -i sin t/2], [-i sin t/2, cos t/2]]
    if gate.m[0].im == 0.0
        && gate.m[1].re == 0.0
        && gate.m[2].re == 0.0
        && gate.m[3].im == 0.0
        && (gate.m[0].re - gate.m[3].re).abs() < 1e-15
    {
        // m[2].im = -sin(t/2); atan2(sin, cos) recovers the full range.
        let theta = 2.0 * (-gate.m[2].im).atan2(gate.m[0].re);
        if gate1_close(gate, &Gate1::rx(theta)) {
            return Ok(("rx", format!("({theta:.17})")));
        }
    }

    // ry(theta): [[cos t/2, -sin t/2], [sin t/2, cos t/2]]
    if gate.m[0].im == 0.0
        && gate.m[1].im == 0.0
        && gate.m[2].im == 0.0
        && gate.m[3].im == 0.0
        && (gate.m[0].re - gate.m[3].re).abs() < 1e-15
        && (gate.m[1].re + gate.m[2].re).abs() < 1e-15
    {
        // m[2].re = sin(t/2), m[0].re = cos(t/2).
        let theta = 2.0 * gate.m[2].re.atan2(gate.m[0].re);
        if gate1_close(gate, &Gate1::ry(theta)) {
            return Ok(("ry", format!("({theta:.17})")));
        }
    }

    // diagonal gates: rz(theta) = diag(e^{-i t/2}, e^{i t/2}), phase = diag(1, e^{i l}).
    if gate.m[1] == Complex64::ZERO && gate.m[2] == Complex64::ZERO {
        // theta = -2 * arg(m[0]).
        let theta = -2.0 * gate.m[0].im.atan2(gate.m[0].re);
        if gate1_close(gate, &Gate1::rz(theta)) {
            return Ok(("rz", format!("({theta:.17})")));
        }
        // lambda = arg(m[3]).
        let lambda = gate.m[3].im.atan2(gate.m[3].re);
        if gate1_close(gate, &Gate1::phase(lambda)) {
            return Ok(("p", format!("({lambda:.17})")));
        }
    }

    Err(no_name())
}

/// Returns the stdgates.inc name for a `Gate2`.
pub fn gate_info_2q(gate: &Gate2) -> Result<&'static str> {
    if *gate == Gate2::cnot() {
        return Ok("cx");
    }
    if *gate == Gate2::cz() {
        return Ok("cz");
    }
    if *gate == Gate2::swap() {
        return Ok("swap");
    }
    Err(no_name())
}

/// Maps an `OpenQASM` gate name and argument list to a `Gate1`.
pub fn lookup_gate_1q(name: &str, args: &[f64]) -> Result<Gate1> {
    let err = |msg: String| Error::Qasm {
        line: 0,
        col: 0,
        message: msg,
    };
    match (name, args.len()) {
        ("h", 0) => Ok(Gate1::h()),
        ("x", 0) => Ok(Gate1::x()),
        ("y", 0) => Ok(Gate1::y()),
        ("z", 0) => Ok(Gate1::z()),
        ("s", 0) => Ok(Gate1::s()),
        ("t", 0) => Ok(Gate1::t()),
        ("id", 0) => Ok(Gate1::id()),
        ("rx", 1) => Ok(Gate1::rx(args[0])),
        ("ry", 1) => Ok(Gate1::ry(args[0])),
        ("rz", 1) => Ok(Gate1::rz(args[0])),
        ("p" | "phase", 1) => Ok(Gate1::phase(args[0])),
        (n, _) => Err(err(format!("unknown single-qubit gate '{n}'"))),
    }
}

/// Maps an `OpenQASM` gate name to a `Gate2`.
pub fn lookup_gate_2q(name: &str) -> Result<Gate2> {
    match name {
        "cx" | "cnot" | "CX" => Ok(Gate2::cnot()),
        "cz" => Ok(Gate2::cz()),
        "swap" => Ok(Gate2::swap()),
        n => Err(Error::Qasm {
            line: 0,
            col: 0,
            message: format!("unknown two-qubit gate '{n}'"),
        }),
    }
}
