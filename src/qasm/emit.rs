//! `OpenQASM` 3.0 emitter: `Circuit` to `String`.

use std::fmt::Write as _;

use crate::circuit::Circuit;
use crate::op::Op;
use crate::qasm::gates::{gate_info_1q, gate_info_2q};
use crate::{Error, Result};

/// Emits a valid `OpenQASM` 3.0 string for `circuit`.
///
/// # Errors
///
/// Returns [`Error::Qasm`] if any gate in the circuit has no `OpenQASM` 3 name
/// (i.e. was built from an arbitrary matrix rather than a named constructor),
/// or if an `IfClassic` op wraps another `IfClassic`.
pub fn emit(circuit: &Circuit) -> Result<String> {
    let mut buf = String::with_capacity(64 + circuit.ops().len() * 16);
    buf.push_str("OPENQASM 3.0;\n");
    buf.push_str("include \"stdgates.inc\";\n");
    if circuit.num_qubits() > 0 {
        let _ = writeln!(buf, "qubit[{}] q;", circuit.num_qubits());
    }
    if circuit.num_classical() > 0 {
        let _ = writeln!(buf, "bit[{}] c;", circuit.num_classical());
    }
    buf.push('\n');

    for op in circuit.ops() {
        emit_op(&mut buf, op, 0)?;
        buf.push('\n');
    }

    Ok(buf)
}

// writes one statement into `buf`. `depth` guards against nested `if`.
fn emit_op(buf: &mut String, op: &Op, depth: usize) -> Result<()> {
    match op {
        Op::Apply1 { gate, target } => {
            let (name, args) = gate_info_1q(gate)?;
            let _ = write!(buf, "{name}{args} q[{}];", target.index());
        }
        Op::Apply2 { gate, a, b } => {
            let name = gate_info_2q(gate)?;
            let _ = write!(buf, "{name} q[{}], q[{}];", a.index(), b.index());
        }
        Op::Controlled {
            controls,
            gate,
            target,
        } => {
            let (name, args) = gate_info_1q(gate)?;
            if controls.len() == 1 {
                buf.push_str("ctrl @ ");
            } else {
                let _ = write!(buf, "ctrl({}) @ ", controls.len());
            }
            let _ = write!(buf, "{name}{args} ");
            for c in controls {
                let _ = write!(buf, "q[{}], ", c.index());
            }
            let _ = write!(buf, "q[{}];", target.index());
        }
        Op::Measure { qubit, into } => {
            let _ = write!(buf, "c[{}] = measure q[{}];", into.index(), qubit.index());
        }
        Op::IfClassic { bit, then } => {
            if depth > 0 {
                return Err(Error::Qasm {
                    line: 0,
                    col: 0,
                    message: "nested IfClassic is not supported in OpenQASM 3 emit".into(),
                });
            }
            let _ = write!(buf, "if (c[{}] == 1) ", bit.index());
            emit_op(buf, then, depth + 1)?;
        }
    }
    Ok(())
}
