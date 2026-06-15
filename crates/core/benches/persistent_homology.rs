use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

mod common;

use common::{dataset_files, load_dataset, max_dim_from_env, run_once, should_bench, Mode};

fn bench_dense(c: &mut Criterion) {
    if !should_bench(Mode::Dense) {
        return;
    }

    let max_dim = max_dim_from_env();
    let mut group = c.benchmark_group(format!("dense_h{max_dim}"));
    for file in dataset_files(max_dim) {
        let dataset = load_dataset(&file);
        group.bench_function(dataset.name.clone(), |b| {
            b.iter(|| {
                black_box(run_once(
                    black_box(Mode::Dense),
                    black_box(&dataset),
                    black_box(max_dim),
                ))
            });
        });
    }
    group.finish();
}

fn bench_sparse(c: &mut Criterion) {
    if !should_bench(Mode::Sparse) {
        return;
    }

    let max_dim = max_dim_from_env();
    let mut group = c.benchmark_group(format!("sparse_h{max_dim}"));
    for file in dataset_files(max_dim) {
        let dataset = load_dataset(&file);
        group.bench_function(dataset.name.clone(), |b| {
            b.iter(|| {
                black_box(run_once(
                    black_box(Mode::Sparse),
                    black_box(&dataset),
                    black_box(max_dim),
                ))
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_dense, bench_sparse
}
criterion_main!(benches);
