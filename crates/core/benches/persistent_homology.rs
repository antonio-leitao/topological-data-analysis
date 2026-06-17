use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

mod common;

use common::{dataset_files, load_dataset, max_dim_from_env, run_once};

fn bench_persistent_homology(c: &mut Criterion) {
    let max_dim = max_dim_from_env();
    let mut group = c.benchmark_group(format!("bitcsr_h{max_dim}"));
    for file in dataset_files(max_dim) {
        let dataset = load_dataset(&file);
        group.bench_function(dataset.name.clone(), |b| {
            b.iter(|| black_box(run_once(black_box(&dataset), black_box(max_dim))));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_persistent_homology
}
criterion_main!(benches);
