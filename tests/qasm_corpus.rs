//! Parse-corpus tests for the `OpenQASM` 3 reader.
//
// the qe-compiler test files (bell-v0, teleport, measure, ...) all fall outside
// the supported subset: they use physical-qubit syntax (`qubit $0;`), scalar bit
// declarations (`bit c0;`), and the builtin `U(θ,φ,λ)` gate rather than the
// stdgates.inc named-gate subset everett reads. so the corpus here is a set of
// hand-written snippets that exercise the supported grammar directly.

use everett::Circuit;

fn parse_ok(src: &str) -> Circuit {
    Circuit::from_qasm(src).unwrap_or_else(|e| panic!("expected parse to succeed: {e}\n{src}"))
}

#[test]
fn bell() {
    let c = parse_ok(
        r#"OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"#,
    );
    assert_eq!(c.num_qubits(), 2);
    assert_eq!(c.num_classical(), 2);
    assert_eq!(c.len(), 4);
}

#[test]
fn rotations_with_pi_expressions() {
    let c = parse_ok(
        r#"OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
rx(pi/2) q[0];
ry(-pi/4) q[0];
rz(2*pi) q[0];
p(pi) q[0];
"#,
    );
    assert_eq!(c.len(), 4);
}

#[test]
fn arrow_and_assignment_measure_forms() {
    let arrow = parse_ok("OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nmeasure q[0] -> c[0];\n");
    let assign = parse_ok("OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nc[0] = measure q[0];\n");
    assert_eq!(arrow.len(), 1);
    assert_eq!(assign.len(), 1);
}

#[test]
fn ctrl_modifier_and_multi_control() {
    let c = parse_ok(
        r#"OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ x q[0], q[1];
ctrl(2) @ x q[0], q[1], q[2];
"#,
    );
    assert_eq!(c.len(), 2);
}

#[test]
fn gate_definitions_are_skipped() {
    // a circuit that defines a custom gate before using stdgates primitives.
    let c = parse_ok(
        r#"OPENQASM 3.0;
include "stdgates.inc";
gate mygate q { h q; }
qubit[1] q;
h q[0];
"#,
    );
    assert_eq!(c.len(), 1);
}

#[test]
fn classical_control_if() {
    let c = parse_ok(
        r#"OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
if (c[0] == 1) x q[0];
"#,
    );
    assert_eq!(c.len(), 2);
}

#[test]
fn line_and_block_comments() {
    let c = parse_ok(
        r#"OPENQASM 3.0;
// a line comment
include "stdgates.inc";
qubit[1] q; /* trailing block comment */
/* multi
   line
   comment */
h q[0];
"#,
    );
    assert_eq!(c.len(), 1);
}

#[test]
fn version_line_is_optional() {
    let c = parse_ok("qubit[1] q;\nh q[0];\n");
    assert_eq!(c.len(), 1);
}

#[test]
fn alternate_register_names_accepted() {
    // the emitter always uses q/c, but the parser must accept any identifier.
    let c = parse_ok(
        r"OPENQASM 3.0;
qubit[2] qreg;
bit[2] creg;
h qreg[0];
cx qreg[0], qreg[1];
creg[0] = measure qreg[0];
",
    );
    assert_eq!(c.num_qubits(), 2);
    assert_eq!(c.len(), 3);
}
