//! Trampoline benchmarks: the fixed per-call overhead of Python -> Rust calls.
//!
//! Every `#[pyfunction]` / `#[pymethods]` entry point goes through the
//! trampoline in `src/impl_/trampoline.rs` (panic trap, attach guard,
//! `catch_unwind`) before any user code runs. These points call zero-work
//! functions so that the measured cost is the call machinery plus that
//! bookkeeping, with no body or conversion cost to drown it out. Compare
//! against the Rust -> Python direction measured by `bench_call`.

use std::hint::black_box;

use codspeed_criterion_compat::{criterion_group, criterion_main, Bencher, Criterion};

use pyo3::prelude::*;
use pyo3::types::PyInt;

/// Zero-work function to measure the fixed per-call overhead of the
/// Python -> Rust call path (vectorcall machinery, trampoline bookkeeping,
/// and the `()` -> `None` return conversion).
#[pyfunction]
fn noop() {}

/// Same call path with one argument, so that argument extraction is included.
#[pyfunction]
fn noop_arg(_x: u64) {}

fn trampoline_noargs_call(b: &mut Bencher<'_>) {
    Python::attach(|py| {
        let noop = wrap_pyfunction!(noop)(py).unwrap();
        b.iter(|| {
            for _ in 0..1000 {
                black_box(&noop).call0().unwrap();
            }
        });
    })
}

fn trampoline_one_arg_call(b: &mut Bencher<'_>) {
    Python::attach(|py| {
        let noop_arg = wrap_pyfunction!(noop_arg)(py).unwrap();
        let arg = PyInt::new(py, 42);
        b.iter(|| {
            for _ in 0..1000 {
                black_box(&noop_arg).call1((arg.clone(),)).unwrap();
            }
        });
    })
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("trampoline_noargs_call", trampoline_noargs_call);
    c.bench_function("trampoline_one_arg_call", trampoline_one_arg_call);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
