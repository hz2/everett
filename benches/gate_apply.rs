//! Criterion benchmarks for gate application across qubit counts.
//!
//! Fleshed out alongside the performance work; for now this establishes the
//! harness so `cargo bench` and the manifest's `[[bench]]` entry resolve.

// the crate-wide missing_docs lint targets the public library API; criterion's
// macros generate undocumented items in this bench binary, so opt out here.
#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};

fn placeholder(_c: &mut Criterion) {
    // TODO: bench apply_1q / apply_2q across n once the backend lands.
}

criterion_group!(benches, placeholder);
criterion_main!(benches);
