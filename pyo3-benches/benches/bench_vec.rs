//! `Vec<T>` extraction benchmarks.
//!
//! `Vec<T>` extraction from exact `list` / `tuple` inputs takes a fast path
//! (direct indexed access); all other sequences go through the generic
//! iterator protocol (`extract_sequence` in `src/conversions/std/vec.rs`:
//! `PyObject_GetIter` + `PyIter_Next` per element). These points time the
//! full path including per-element `T::extract`, so that both the fast path
//! and the fallback can be measured and compared.
//! `extract_vec_u64_from_range_large` covers the iterator fallback (a
//! non-list sequence), and `extract_vec_pyany_from_list_large` isolates the
//! element-access machinery from element conversion cost.

use std::hint::black_box;

use codspeed_criterion_compat::{criterion_group, criterion_main, Bencher, Criterion};

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

fn extract_vec_u64_from_list_small(b: &mut Bencher<'_>) {
    Python::attach(|py| {
        const LEN: usize = 10;
        let list = PyList::new(py, 0..LEN).unwrap().into_any();
        b.iter_with_large_drop(|| black_box(&list).extract::<Vec<u64>>());
    })
}

fn extract_vec_u64_from_list_large(b: &mut Bencher<'_>) {
    Python::attach(|py| {
        const LEN: usize = 100_000;
        let list = PyList::new(py, 0..LEN).unwrap().into_any();
        b.iter_with_large_drop(|| black_box(&list).extract::<Vec<u64>>());
    })
}

fn extract_vec_string_from_list_small(b: &mut Bencher<'_>) {
    Python::attach(|py| {
        const LEN: usize = 10;
        let strings: Vec<String> = (0..LEN).map(|i| format!("string_{i}")).collect();
        let list = PyList::new(py, strings).unwrap().into_any();
        b.iter_with_large_drop(|| black_box(&list).extract::<Vec<String>>());
    })
}

fn extract_vec_string_from_list_large(b: &mut Bencher<'_>) {
    Python::attach(|py| {
        const LEN: usize = 100_000;
        let strings: Vec<String> = (0..LEN).map(|i| format!("string_{i}")).collect();
        let list = PyList::new(py, strings).unwrap().into_any();
        b.iter_with_large_drop(|| black_box(&list).extract::<Vec<String>>());
    })
}

fn extract_vec_pyany_from_list_large(b: &mut Bencher<'_>) {
    Python::attach(|py| {
        const LEN: usize = 100_000;
        let list = PyList::new(py, 0..LEN).unwrap().into_any();
        b.iter_with_large_drop(|| black_box(&list).extract::<Vec<Py<PyAny>>>());
    })
}

fn extract_vec_u64_from_tuple_large(b: &mut Bencher<'_>) {
    Python::attach(|py| {
        const LEN: usize = 100_000;
        let tuple = PyTuple::new(py, 0..LEN).unwrap().into_any();
        b.iter_with_large_drop(|| black_box(&tuple).extract::<Vec<u64>>());
    })
}

fn extract_vec_u64_from_range_large(b: &mut Bencher<'_>) {
    Python::attach(|py| {
        let range = py.eval(c"range(100_000)", None, None).unwrap();
        b.iter_with_large_drop(|| black_box(&range).extract::<Vec<u64>>());
    })
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("extract_vec_u64_from_list_small", extract_vec_u64_from_list_small);
    c.bench_function("extract_vec_u64_from_list_large", extract_vec_u64_from_list_large);
    c.bench_function("extract_vec_string_from_list_small", extract_vec_string_from_list_small);
    c.bench_function("extract_vec_string_from_list_large", extract_vec_string_from_list_large);
    c.bench_function("extract_vec_pyany_from_list_large", extract_vec_pyany_from_list_large);
    c.bench_function("extract_vec_u64_from_tuple_large", extract_vec_u64_from_tuple_large);
    c.bench_function("extract_vec_u64_from_range_large", extract_vec_u64_from_range_large);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
