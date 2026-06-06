//! `OpenQASM` 3.0 import/export for [`crate::Circuit`].
//!
//! Two entry points:
//! - [`emit`] converts a `Circuit` to a valid `OpenQASM` 3.0 string.
//! - [`parse`] converts an `OpenQASM` 3.0 string back to a `Circuit`.
//!
//! Circuits must use named gate constructors (`Gate1::h()`, `Gate2::cnot()`, …).
//! Arbitrary matrix gates return [`crate::Error::Qasm`] at emit time.
//!
//! # Example
//!
//! ```
//! use everett::Circuit;
//!
//! let mut c = Circuit::new(2);
//! c.h(0).cnot(0, 1);
//!
//! let src = c.to_qasm().unwrap();
//! let c2 = Circuit::from_qasm(&src).unwrap();
//! assert_eq!(c.len(), c2.len());
//! ```

mod emit;
mod gates;
mod parse;

pub use emit::emit;
pub use parse::parse;

#[cfg(test)]
mod tests {
    use crate::algorithms::prep::{bell, ghz};
    use crate::algorithms::qft::qft;
    use crate::backend::StateVectorBackend;
    use crate::circuit::Circuit;
    use crate::gate::{Gate1, Gate2};
    use crate::op::Op;
    use crate::qubit::{ClassicalBit, QubitId};

    use super::{emit, parse};

    fn roundtrip(c: &Circuit) -> Circuit {
        let src = emit(c).expect("emit failed");
        parse(&src).expect("parse failed")
    }

    fn fidelity_after_roundtrip(c: &Circuit) -> f64 {
        let c2 = roundtrip(c);
        let s1 = StateVectorBackend::run(c).unwrap().state().clone();
        let s2 = StateVectorBackend::run(&c2).unwrap().state().clone();
        s1.fidelity(&s2)
    }

    // ── per-op variant tests ──────────────────────────────────────────────

    #[test]
    fn roundtrip_apply1_named_gates() {
        let gates: &[(&str, Gate1)] = &[
            ("h", Gate1::h()),
            ("x", Gate1::x()),
            ("y", Gate1::y()),
            ("z", Gate1::z()),
            ("s", Gate1::s()),
            ("t", Gate1::t()),
            ("id", Gate1::id()),
            ("rx(0.7)", Gate1::rx(0.7)),
            ("ry(-1.3)", Gate1::ry(-1.3)),
            ("rz(2.1)", Gate1::rz(2.1)),
            ("phase(0.9)", Gate1::phase(0.9)),
        ];
        for (label, g) in gates {
            let mut c = Circuit::new(1);
            c.gate1(*g, 0);
            assert!(
                fidelity_after_roundtrip(&c) > 1.0 - 1e-10,
                "roundtrip failed for {label}"
            );
        }
    }

    #[test]
    fn roundtrip_apply2() {
        for g in [Gate2::cnot(), Gate2::cz(), Gate2::swap()] {
            let mut c = Circuit::new(2);
            c.gate2(g, 0, 1);
            assert!(fidelity_after_roundtrip(&c) > 1.0 - 1e-10);
        }
    }

    #[test]
    fn roundtrip_controlled_single() {
        let mut c = Circuit::new(2);
        c.controlled(&[0], Gate1::x(), 1);
        assert!(fidelity_after_roundtrip(&c) > 1.0 - 1e-10);
    }

    #[test]
    fn roundtrip_controlled_two_controls() {
        let mut c = Circuit::new(3);
        c.controlled(&[0, 1], Gate1::x(), 2);
        assert!(fidelity_after_roundtrip(&c) > 1.0 - 1e-10);
    }

    #[test]
    fn roundtrip_measure() {
        let mut c = Circuit::with_classical(1, 1);
        c.measure(0, 0);
        let src = emit(&c).unwrap();
        let c2 = parse(&src).unwrap();
        assert_eq!(c2.num_classical(), 1);
        assert_eq!(c2.len(), 1);
    }

    #[test]
    fn roundtrip_if_classic_apply1() {
        let mut c = Circuit::with_classical(1, 1);
        c.push_op(Op::IfClassic {
            bit: ClassicalBit(0),
            then: Box::new(Op::Apply1 {
                gate: Gate1::x(),
                target: QubitId(0),
            }),
        });
        let src = emit(&c).unwrap();
        let c2 = parse(&src).unwrap();
        assert_eq!(c2.len(), 1);
    }

    // ── algorithm round-trip tests ────────────────────────────────────────

    #[test]
    fn roundtrip_bell() {
        let c = bell();
        assert!(fidelity_after_roundtrip(&c) > 1.0 - 1e-10);
    }

    #[test]
    fn roundtrip_ghz() {
        let c = ghz(4);
        assert!(fidelity_after_roundtrip(&c) > 1.0 - 1e-10);
    }

    #[test]
    fn roundtrip_qft() {
        let c = qft(4);
        assert!(fidelity_after_roundtrip(&c) > 1.0 - 1e-10);
    }

    // ── emitter output tests ──────────────────────────────────────────────

    #[test]
    fn emit_header() {
        let c = Circuit::new(3);
        let src = emit(&c).unwrap();
        assert!(src.starts_with("OPENQASM 3.0;\n"));
        assert!(src.contains("include \"stdgates.inc\";"));
        assert!(src.contains("qubit[3] q;"));
    }

    #[test]
    fn emit_no_classical_omits_bit_line() {
        let c = Circuit::new(2);
        let src = emit(&c).unwrap();
        // "qubit[..." contains "bit[" as a substring, so check line-by-line for
        // a standalone classical-register declaration.
        assert!(!src.lines().any(|l| l.starts_with("bit[")));
    }

    #[test]
    fn emit_controlled_single_uses_ctrl_at() {
        let mut c = Circuit::new(2);
        c.controlled(&[0], Gate1::x(), 1);
        let src = emit(&c).unwrap();
        assert!(src.contains("ctrl @ x q[0], q[1];"));
    }

    #[test]
    fn emit_cnot_uses_cx() {
        let mut c = Circuit::new(2);
        c.cnot(0, 1);
        let src = emit(&c).unwrap();
        assert!(src.contains("cx q[0], q[1];"));
    }

    // ── parser edge cases ─────────────────────────────────────────────────

    #[test]
    fn parse_both_measure_forms() {
        // arrow form
        let src = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nmeasure q[0] -> c[0];\n";
        let c = parse(src).unwrap();
        assert_eq!(c.len(), 1);

        // assignment form
        let src2 = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nc[0] = measure q[0];\n";
        let c2 = parse(src2).unwrap();
        assert_eq!(c2.len(), 1);
    }

    #[test]
    fn parse_pi_expressions() {
        let src = "OPENQASM 3.0;\nqubit[1] q;\nrx(pi/2) q[0];\n";
        let c = parse(src).unwrap();
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn parse_gate_definition_skipped() {
        let src = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\ngate mygate q { h q; }\nh q[0];\n";
        let c = parse(src).unwrap();
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn parse_ctrl_multi_control() {
        let src = "OPENQASM 3.0;\nqubit[3] q;\nctrl(2) @ x q[0], q[1], q[2];\n";
        let c = parse(src).unwrap();
        assert_eq!(c.len(), 1);
        match &c.ops()[0] {
            Op::Controlled { controls, .. } => assert_eq!(controls.len(), 2),
            _ => panic!("expected Controlled op"),
        }
    }

    #[test]
    fn parse_version_optional() {
        let src = "qubit[1] q;\nh q[0];\n";
        let c = parse(src).unwrap();
        assert_eq!(c.len(), 1);
    }
}
