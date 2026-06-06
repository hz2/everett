//! Hand-rolled recursive-descent `OpenQASM` 3.0 parser (minimal subset).

use crate::circuit::Circuit;
use crate::op::Op;
use crate::qasm::gates::{lookup_gate_1q, lookup_gate_2q};
use crate::qubit::{ClassicalBit, QubitId};
use crate::{Error, Result};

#[derive(Clone, PartialEq, Debug)]
enum Tok<'s> {
    Ident(&'s str),
    Int(usize),
    Float(f64),
    Str(&'s str),
    // keywords
    KwOpenQasm,
    KwInclude,
    KwQubit,
    KwBit,
    KwMeasure,
    KwIf,
    KwGate,
    KwCtrl,
    KwPi,
    KwTau,
    // symbols
    Semi,
    LParen,
    RParen,
    LBrack,
    RBrack,
    LBrace,
    RBrace,
    Comma,
    Eq,
    EqEq,
    Arrow,
    Slash,
    Star,
    Plus,
    Minus,
    At,
    Eof,
}

struct Lexer<'s> {
    src: &'s str,
    pos: usize,
    line: usize,
    col: usize,
}

impl<'s> Lexer<'s> {
    fn new(src: &'s str) -> Self {
        Self {
            src,
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn remaining(&self) -> &'s str {
        &self.src[self.pos..]
    }

    fn advance(&mut self, n: usize) {
        let chunk = &self.src[self.pos..self.pos + n];
        for ch in chunk.chars() {
            if ch == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
        }
        self.pos += n;
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            let rem = self.remaining();
            // whitespace
            let ws = rem.len() - rem.trim_start().len();
            if ws > 0 {
                self.advance(ws);
                continue;
            }
            let rem = self.remaining();
            // line comment
            if rem.starts_with("//") {
                let end = rem.find('\n').map_or(rem.len(), |i| i + 1);
                self.advance(end);
                continue;
            }
            // block comment
            if rem.starts_with("/*") {
                let end = rem.find("*/").map_or(rem.len(), |i| i + 2);
                self.advance(end);
                continue;
            }
            break;
        }
    }

    fn next_token(&mut self) -> (Tok<'s>, usize, usize) {
        self.skip_ws_and_comments();
        let line = self.line;
        let col = self.col;
        let rem = self.remaining();

        if rem.is_empty() {
            return (Tok::Eof, line, col);
        }

        let ch = rem.chars().next().unwrap();

        // string literal
        if ch == '"' {
            let inner_start = self.pos + 1;
            let inner = &rem[1..];
            let end = inner.find('"').unwrap_or(inner.len());
            let s = &self.src[inner_start..inner_start + end];
            self.advance(end + 2);
            return (Tok::Str(s), line, col);
        }

        // two-character symbols first
        if rem.starts_with("==") {
            self.advance(2);
            return (Tok::EqEq, line, col);
        }
        if rem.starts_with("->") {
            self.advance(2);
            return (Tok::Arrow, line, col);
        }

        // single-character symbols
        let single = match ch {
            ';' => Some(Tok::Semi),
            '(' => Some(Tok::LParen),
            ')' => Some(Tok::RParen),
            '[' => Some(Tok::LBrack),
            ']' => Some(Tok::RBrack),
            '{' => Some(Tok::LBrace),
            '}' => Some(Tok::RBrace),
            ',' => Some(Tok::Comma),
            '=' => Some(Tok::Eq),
            '/' => Some(Tok::Slash),
            '*' => Some(Tok::Star),
            '+' => Some(Tok::Plus),
            '-' => Some(Tok::Minus),
            '@' => Some(Tok::At),
            _ => None,
        };
        if let Some(tok) = single {
            self.advance(1);
            return (tok, line, col);
        }

        // number (float or int)
        if ch.is_ascii_digit() {
            let s = Self::scan_number(rem);
            let tok = if s.contains('.') || s.contains('e') || s.contains('E') {
                Tok::Float(s.parse().unwrap_or(0.0))
            } else {
                s.parse::<usize>()
                    .map_or(Tok::Float(s.parse().unwrap_or(0.0)), Tok::Int)
            };
            self.advance(s.len());
            return (tok, line, col);
        }

        // identifier or keyword
        if ch.is_ascii_alphabetic() || ch == '_' {
            let end = rem
                .char_indices()
                .find(|(_, c)| !c.is_ascii_alphanumeric() && *c != '_')
                .map_or(rem.len(), |(i, _)| i);
            let word = &rem[..end];
            self.advance(end);
            let tok = match word {
                "OPENQASM" => Tok::KwOpenQasm,
                "include" => Tok::KwInclude,
                "qubit" => Tok::KwQubit,
                "bit" => Tok::KwBit,
                "measure" => Tok::KwMeasure,
                "if" => Tok::KwIf,
                "gate" => Tok::KwGate,
                "ctrl" => Tok::KwCtrl,
                "pi" => Tok::KwPi,
                "tau" => Tok::KwTau,
                _ => Tok::Ident(word),
            };
            return (tok, line, col);
        }

        // unrecognised: skip one byte
        self.advance(1);
        (Tok::Ident("?"), line, col)
    }

    fn scan_number(s: &str) -> &str {
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
            i += 1;
            if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                i += 1;
            }
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        &s[..i]
    }
}

// parser over the lexer token stream. operands accept any register identifier,
// so the declared register names need not be tracked.
struct Parser<'s> {
    lex: Lexer<'s>,
    peek: Option<(Tok<'s>, usize, usize)>,
}

impl<'s> Parser<'s> {
    fn new(src: &'s str) -> Self {
        Self {
            lex: Lexer::new(src),
            peek: None,
        }
    }

    fn peek_tok(&mut self) -> &Tok<'s> {
        if self.peek.is_none() {
            self.peek = Some(self.lex.next_token());
        }
        &self.peek.as_ref().unwrap().0
    }

    fn peek_line_col(&mut self) -> (usize, usize) {
        if self.peek.is_none() {
            self.peek = Some(self.lex.next_token());
        }
        let (_, l, c) = self.peek.as_ref().unwrap();
        (*l, *c)
    }

    fn consume(&mut self) -> (Tok<'s>, usize, usize) {
        self.peek.take().unwrap_or_else(|| self.lex.next_token())
    }

    fn err(&mut self, msg: &str) -> Error {
        let (line, col) = self.peek_line_col();
        Error::Qasm {
            line,
            col,
            message: msg.to_string(),
        }
    }

    fn expect_semi(&mut self) -> Result<()> {
        if *self.peek_tok() == Tok::Semi {
            self.consume();
            Ok(())
        } else {
            Err(self.err("expected ';'"))
        }
    }

    // skip a gate body { ... } by counting braces.
    fn skip_brace_block(&mut self) -> Result<()> {
        // consume '{'
        match self.consume().0 {
            Tok::LBrace => {}
            _ => return Err(self.err("expected '{' for gate body")),
        }
        let mut depth = 1usize;
        loop {
            match self.consume().0 {
                Tok::LBrace => depth += 1,
                Tok::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                Tok::Eof => return Err(self.err("unexpected EOF in gate body")),
                _ => {}
            }
        }
        Ok(())
    }

    pub fn parse_program(&mut self) -> Result<Circuit> {
        let mut num_qubits = 0usize;
        let mut num_classical = 0usize;
        let mut ops: Vec<Op> = Vec::new();

        // optional version line
        if *self.peek_tok() == Tok::KwOpenQasm {
            self.consume();
            // consume version spec (e.g. "3.0" or "3")
            match self.peek_tok() {
                Tok::Int(_) | Tok::Float(_) => {
                    self.consume();
                }
                _ => {}
            }
            self.expect_semi()?;
        }

        // optional include line
        if *self.peek_tok() == Tok::KwInclude {
            self.consume();
            self.consume(); // string
            self.expect_semi()?;
        }

        // declarations, gate definitions, and statements may interleave in any
        // order after the header, so handle them all in one loop.
        loop {
            match self.peek_tok() {
                Tok::Eof => break,
                Tok::KwQubit => num_qubits = self.parse_qubit_decl()?,
                Tok::KwBit => num_classical = self.parse_bit_decl()?,
                Tok::KwGate => {
                    // skip gate definition: consume up to '{', then the body.
                    self.consume(); // 'gate'
                    while !matches!(self.peek_tok(), Tok::LBrace | Tok::Eof) {
                        self.consume();
                    }
                    self.skip_brace_block()?;
                }
                _ => {
                    if let Some(op) = self.parse_statement(num_qubits, num_classical)? {
                        ops.push(op);
                    }
                }
            }
        }

        let mut c = Circuit::with_classical(num_qubits, num_classical);
        for op in ops {
            c.push_op(op);
        }
        Ok(c)
    }

    fn parse_qubit_decl(&mut self) -> Result<usize> {
        self.consume(); // 'qubit'
        self.expect_lbrack()?;
        let n = self.parse_int()?;
        self.expect_rbrack()?;
        // register name, accepted and discarded
        if !matches!(self.consume().0, Tok::Ident(_)) {
            return Err(self.err("expected qubit register name"));
        }
        self.expect_semi()?;
        Ok(n)
    }

    fn parse_bit_decl(&mut self) -> Result<usize> {
        self.consume(); // 'bit'
        self.expect_lbrack()?;
        let n = self.parse_int()?;
        self.expect_rbrack()?;
        if !matches!(self.consume().0, Tok::Ident(_)) {
            return Err(self.err("expected classical register name"));
        }
        self.expect_semi()?;
        Ok(n)
    }

    fn expect_lbrack(&mut self) -> Result<()> {
        match self.peek_tok() {
            Tok::LBrack => {
                self.consume();
                Ok(())
            }
            _ => Err(self.err("expected '['")),
        }
    }

    fn expect_rbrack(&mut self) -> Result<()> {
        match self.peek_tok() {
            Tok::RBrack => {
                self.consume();
                Ok(())
            }
            _ => Err(self.err("expected ']'")),
        }
    }

    fn parse_int(&mut self) -> Result<usize> {
        match self.consume().0 {
            Tok::Int(n) => Ok(n),
            _ => Err(self.err("expected integer")),
        }
    }

    /// Parse one statement; returns `None` for skippable constructs (reset,
    /// barrier, empty).
    fn parse_statement(&mut self, _num_qubits: usize, _num_classical: usize) -> Result<Option<Op>> {
        match self.peek_tok() {
            Tok::Semi => {
                self.consume();
                Ok(None)
            }
            Tok::KwMeasure => {
                let op = self.parse_measure_stmt()?;
                Ok(Some(op))
            }
            Tok::KwIf => {
                let op = self.parse_if()?;
                Ok(Some(op))
            }
            Tok::KwCtrl => {
                let op = self.parse_gate_call()?;
                Ok(Some(op))
            }
            Tok::Ident(name) => {
                // check for: classical_bit[i] = measure q[j];
                // or gate call
                // or known-skippable gates like 'reset', 'barrier'
                let name = *name;
                if name == "reset" || name == "barrier" {
                    // skip until ';'
                    loop {
                        match self.consume().0 {
                            Tok::Semi | Tok::Eof => break,
                            _ => {}
                        }
                    }
                    return Ok(None);
                }
                // is this a classical assignment: <reg>[i] = measure ...
                // we peek to see if it's <ident>[...] =
                // we handle this by trying to detect the pattern
                // look ahead: consume ident, then peek
                self.consume(); // consume the ident
                match self.peek_tok() {
                    Tok::LBrack => {
                        // could be c[i] = measure q[j];
                        self.consume(); // '['
                        let idx = self.parse_int()?;
                        self.expect_rbrack()?;
                        match self.peek_tok() {
                            Tok::Eq => {
                                self.consume(); // '='
                                match self.peek_tok() {
                                    Tok::KwMeasure => {
                                        let op = self.parse_measure_assign(idx)?;
                                        Ok(Some(op))
                                    }
                                    _ => Err(self.err("expected 'measure' after '='")),
                                }
                            }
                            _ => Err(self.err("unexpected token after register index")),
                        }
                    }
                    Tok::LParen | Tok::Ident(_) | Tok::Semi | Tok::Comma => {
                        // gate call: name already consumed, reconstruct
                        let op = self.parse_gate_call_after_name(name)?;
                        Ok(Some(op))
                    }
                    _ => Err(self.err("unexpected token in statement")),
                }
            }
            _ => {
                // skip unknown token
                self.consume();
                Ok(None)
            }
        }
    }

    // parse: measure q[j]; (standalone measure — arrow form)
    fn parse_measure_stmt(&mut self) -> Result<Op> {
        self.consume(); // 'measure'
        let qubit = self.parse_qubit_index()?;
        // optional -> c[i]
        match self.peek_tok() {
            Tok::Arrow => {
                self.consume();
                let into = self.parse_cbit_index()?;
                self.expect_semi()?;
                Ok(Op::Measure {
                    qubit: QubitId(qubit),
                    into: ClassicalBit(into),
                })
            }
            Tok::Semi => {
                self.consume();
                // measure without storing — not representable, skip
                Err(Error::Qasm {
                    line: 0,
                    col: 0,
                    message: "bare 'measure' without destination is not supported".into(),
                })
            }
            _ => Err(self.err("expected '->' or ';' after measure operand")),
        }
    }

    // parse the RHS of c[i] = measure q[j];  (ident and '[i]' already consumed)
    fn parse_measure_assign(&mut self, into: usize) -> Result<Op> {
        self.consume(); // 'measure'
        let qubit = self.parse_qubit_index()?;
        self.expect_semi()?;
        Ok(Op::Measure {
            qubit: QubitId(qubit),
            into: ClassicalBit(into),
        })
    }

    fn parse_if(&mut self) -> Result<Op> {
        self.consume(); // 'if'
        match self.consume().0 {
            Tok::LParen => {}
            _ => return Err(self.err("expected '(' after 'if'")),
        }
        // consume 'c' or bit_reg name
        match self.consume().0 {
            Tok::Ident(_) => {}
            _ => return Err(self.err("expected classical register in if condition")),
        }
        self.expect_lbrack()?;
        let bit = self.parse_int()?;
        self.expect_rbrack()?;
        // == 1
        match self.consume().0 {
            Tok::EqEq => {}
            _ => return Err(self.err("expected '==' in if condition")),
        }
        match self.consume().0 {
            Tok::Int(1) => {}
            _ => {
                return Err(Error::Qasm {
                    line: 0,
                    col: 0,
                    message: "only 'if (c[i] == 1)' is supported".into(),
                });
            }
        }
        match self.consume().0 {
            Tok::RParen => {}
            _ => return Err(self.err("expected ')' to close if condition")),
        }
        // inner statement
        let inner = self
            .parse_statement(0, 0)?
            .ok_or_else(|| self.err("expected statement in if body"))?;
        Ok(Op::IfClassic {
            bit: ClassicalBit(bit),
            then: Box::new(inner),
        })
    }

    fn parse_gate_call(&mut self) -> Result<Op> {
        // caller left 'ctrl' in the stream
        self.consume(); // 'ctrl'
        let n_controls = if *self.peek_tok() == Tok::LParen {
            self.consume(); // '('
            let n = self.parse_int()?;
            match self.consume().0 {
                Tok::RParen => {}
                _ => return Err(self.err("expected ')'")),
            }
            n
        } else {
            1
        };
        match self.consume().0 {
            Tok::At => {}
            _ => return Err(self.err("expected '@' after ctrl")),
        }
        // gate name
        let Tok::Ident(name) = self.consume().0 else {
            return Err(self.err("expected gate name after 'ctrl @'"));
        };
        let args = self.parse_arg_list()?;
        let operands = self.parse_operand_list()?;
        self.expect_semi()?;

        if operands.len() < n_controls + 1 {
            return Err(self.err("not enough operands for ctrl gate"));
        }
        let controls: Vec<QubitId> = operands[..n_controls].iter().map(|&i| QubitId(i)).collect();
        let target = operands[n_controls];
        let gate = lookup_gate_1q(name, &args).map_err(|e| {
            if let Error::Qasm { message, .. } = e {
                let (l, c) = self.peek_line_col();
                Error::Qasm {
                    line: l,
                    col: c,
                    message,
                }
            } else {
                e
            }
        })?;
        Ok(Op::Controlled {
            controls,
            gate,
            target: QubitId(target),
        })
    }

    // gate call when the gate name has already been consumed from the stream.
    fn parse_gate_call_after_name(&mut self, name: &str) -> Result<Op> {
        let args = self.parse_arg_list()?;
        let operands = self.parse_operand_list()?;
        self.expect_semi()?;

        match operands.len() {
            1 => {
                let gate = lookup_gate_1q(name, &args).map_err(|e| {
                    if let Error::Qasm { message, .. } = e {
                        let (l, c) = self.peek_line_col();
                        Error::Qasm {
                            line: l,
                            col: c,
                            message,
                        }
                    } else {
                        e
                    }
                })?;
                Ok(Op::Apply1 {
                    gate,
                    target: QubitId(operands[0]),
                })
            }
            2 if args.is_empty() => {
                let gate = lookup_gate_2q(name).map_err(|e| {
                    if let Error::Qasm { message, .. } = e {
                        let (l, c) = self.peek_line_col();
                        Error::Qasm {
                            line: l,
                            col: c,
                            message,
                        }
                    } else {
                        e
                    }
                })?;
                Ok(Op::Apply2 {
                    gate,
                    a: QubitId(operands[0]),
                    b: QubitId(operands[1]),
                })
            }
            _ => Err(self.err("unexpected gate arity")),
        }
    }

    fn parse_arg_list(&mut self) -> Result<Vec<f64>> {
        if *self.peek_tok() != Tok::LParen {
            return Ok(vec![]);
        }
        self.consume(); // '('
        let mut args = vec![];
        loop {
            match self.peek_tok() {
                Tok::RParen => {
                    self.consume();
                    break;
                }
                Tok::Comma => {
                    self.consume();
                }
                _ => {
                    args.push(self.parse_expr()?);
                }
            }
        }
        Ok(args)
    }

    fn parse_operand_list(&mut self) -> Result<Vec<usize>> {
        let mut out = vec![];
        loop {
            match self.peek_tok() {
                Tok::Ident(_) => {
                    self.consume(); // register name
                    let idx = self.parse_index()?;
                    out.push(idx);
                }
                Tok::Comma => {
                    self.consume();
                }
                _ => break,
            }
        }
        Ok(out)
    }

    fn parse_index(&mut self) -> Result<usize> {
        self.expect_lbrack()?;
        let i = self.parse_int()?;
        self.expect_rbrack()?;
        Ok(i)
    }

    fn parse_qubit_index(&mut self) -> Result<usize> {
        // consume register name then [i]
        match self.peek_tok() {
            Tok::Ident(_) => {
                self.consume();
            }
            _ => return Err(self.err("expected qubit register name")),
        }
        self.parse_index()
    }

    fn parse_cbit_index(&mut self) -> Result<usize> {
        match self.peek_tok() {
            Tok::Ident(_) => {
                self.consume();
            }
            _ => return Err(self.err("expected classical register name")),
        }
        self.parse_index()
    }

    /// Minimal expression parser for angle arguments.
    /// Handles: float/int literals, pi, tau, unary -, binary * / + -.
    fn parse_expr(&mut self) -> Result<f64> {
        self.parse_additive()
    }

    fn parse_additive(&mut self) -> Result<f64> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            match self.peek_tok() {
                Tok::Plus => {
                    self.consume();
                    lhs += self.parse_multiplicative()?;
                }
                Tok::Minus => {
                    self.consume();
                    lhs -= self.parse_multiplicative()?;
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<f64> {
        let mut lhs = self.parse_unary()?;
        loop {
            match self.peek_tok() {
                Tok::Star => {
                    self.consume();
                    lhs *= self.parse_unary()?;
                }
                Tok::Slash => {
                    self.consume();
                    let rhs = self.parse_unary()?;
                    if rhs == 0.0 {
                        return Err(self.err("division by zero in angle expression"));
                    }
                    lhs /= rhs;
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<f64> {
        if *self.peek_tok() == Tok::Minus {
            self.consume();
            return Ok(-self.parse_primary()?);
        }
        if *self.peek_tok() == Tok::Plus {
            self.consume();
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<f64> {
        match self.consume().0 {
            Tok::Float(f) => Ok(f),
            Tok::Int(n) => Ok(n as f64),
            Tok::KwPi => Ok(std::f64::consts::PI),
            Tok::KwTau => Ok(std::f64::consts::TAU),
            Tok::LParen => {
                let v = self.parse_expr()?;
                match self.consume().0 {
                    Tok::RParen => Ok(v),
                    _ => Err(self.err("expected ')'")),
                }
            }
            _ => Err(self.err("expected numeric expression")),
        }
    }
}

/// Parses an `OpenQASM` 3.0 source string into a [`Circuit`].
///
/// # Errors
///
/// Returns [`Error::Qasm`] on any parse error.
pub fn parse(src: &str) -> Result<Circuit> {
    let mut p = Parser::new(src);
    p.parse_program()
}
