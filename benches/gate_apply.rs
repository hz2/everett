//! Criterion benchmarks for the gate-application kernel.
//!
//! Measures the per-gate cost of the bit-insertion kernel across qubit counts,
//! plus a full QFT circuit as a representative workload. Run with `cargo bench`.

#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use everett::algorithms::qft;
use everett::kernel::{apply_1q, apply_2q, apply_controlled_1q};
use everett::{Complex64, Gate1, Gate2, StateVectorBackend};

// a normalized-ish statevector of `n` qubits to apply gates to. the exact
// amplitudes do not matter for timing, only that the buffer is the right size.
fn state(n: usize) -> Vec<Complex64> {
    let dim = 1usize << n;
    let scale = 1.0 / (dim as f64).sqrt();
    (0..dim)
        .map(|j| Complex64::new(scale, scale * 0.5) * (1.0 + (j % 7) as f64 * 0.01))
        .collect()
}

// qubit counts spanning in-cache (small) to memory-bound (large) regimes.
const QUBITS: [usize; 5] = [8, 12, 16, 20, 22];

fn bench_apply_1q(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_1q");
    let h = Gate1::h();
    for &n in &QUBITS {
        let mut amps = state(n);
        group.throughput(Throughput::Elements(1u64 << n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            // apply to the middle qubit: stride neither degenerate nor maximal.
            let k = n / 2;
            b.iter(|| apply_1q(black_box(&mut amps), black_box(k), black_box(&h)));
        });
    }
    group.finish();
}

fn bench_apply_1q_qubit0(c: &mut Criterion) {
    // qubit 0 is the worst case for cache: adjacent-pair stride of 1.
    let mut group = c.benchmark_group("apply_1q_q0");
    let h = Gate1::h();
    for &n in &QUBITS {
        let mut amps = state(n);
        group.throughput(Throughput::Elements(1u64 << n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| apply_1q(black_box(&mut amps), black_box(0), black_box(&h)));
        });
    }
    group.finish();
}

fn bench_apply_2q(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_2q_cnot");
    let cnot = Gate2::cnot();
    for &n in &QUBITS {
        let mut amps = state(n);
        group.throughput(Throughput::Elements(1u64 << n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                apply_2q(
                    black_box(&mut amps),
                    black_box(0),
                    black_box(n / 2),
                    black_box(&cnot),
                );
            });
        });
    }
    group.finish();
}

fn bench_apply_controlled(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_controlled_1q");
    let x = Gate1::x();
    for &n in &QUBITS {
        let mut amps = state(n);
        let controls = [1usize, 2];
        group.throughput(Throughput::Elements(1u64 << n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                apply_controlled_1q(
                    black_box(&mut amps),
                    black_box(&controls),
                    black_box(0),
                    black_box(&x),
                );
            });
        });
    }
    group.finish();
}

fn bench_qft(c: &mut Criterion) {
    // a full QFT: O(n^2) gates, a realistic end-to-end workload.
    let mut group = c.benchmark_group("qft_circuit");
    for &n in &[8usize, 12, 16] {
        let circuit = qft::qft(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| StateVectorBackend::run(black_box(&circuit)).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_apply_1q,
    bench_apply_1q_qubit0,
    bench_apply_2q,
    bench_apply_controlled,
    bench_qft,
);
criterion_main!(benches);
