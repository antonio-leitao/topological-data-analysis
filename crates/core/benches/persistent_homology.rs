use criterion::{criterion_group, criterion_main, Criterion};
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use tda_core::{persistent_homology, persistent_homology_sparse};

const H2_LAST_DATASET: &str = "dragon_2000.txt";

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

fn max_dim() -> usize {
    match env::var("TDA_MAX_DIM").ok().as_deref() {
        None | Some("") | Some("1") => 1,
        Some("2") => 2,
        Some(other) => panic!("TDA_MAX_DIM must be 1 or 2, got {other}"),
    }
}

fn dataset_files(max_dim: usize) -> Vec<String> {
    let mut files: Vec<String> = fs::read_to_string(data_dir().join("datasets.txt"))
        .expect("failed to read data/datasets.txt")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect();
    if max_dim == 2 {
        let end = files
            .iter()
            .position(|file| file == H2_LAST_DATASET)
            .expect("H2 cutoff dataset is missing from data/datasets.txt");
        files.truncate(end + 1);
    }
    files
}

fn dataset_name(file: &str) -> String {
    file.strip_suffix(".txt").unwrap_or(file).to_owned()
}

fn load_points(file: &str) -> (Vec<f32>, usize, usize) {
    let text = fs::read_to_string(data_dir().join(file)).expect("failed to read file");
    let mut n = 0usize;
    let mut d = 0usize;
    let flat: Vec<f32> = text
        .lines()
        .filter(|l| !l.is_empty())
        .flat_map(|l| {
            let row: Vec<f32> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
            if n == 0 {
                d = row.len();
            } else {
                assert_eq!(row.len(), d, "jagged input at row {n}");
            }
            n += 1;
            row
        })
        .collect();
    (flat, n, d)
}

fn bench_dense(c: &mut Criterion) {
    let max_dim = max_dim();
    let mut group = c.benchmark_group(format!("dense_h{max_dim}"));
    for file in dataset_files(max_dim) {
        let name = dataset_name(&file);
        let (points, n, d) = load_points(&file);
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(persistent_homology(
                    black_box(&points),
                    black_box(n),
                    black_box(d),
                    black_box(max_dim),
                    None,
                    false,
                    false,
                ))
            });
        });
    }
    group.finish();
}

fn bench_sparse(c: &mut Criterion) {
    let max_dim = max_dim();
    let mut group = c.benchmark_group(format!("sparse_h{max_dim}"));
    for file in dataset_files(max_dim) {
        let name = dataset_name(&file);
        let (points, n, d) = load_points(&file);
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(persistent_homology_sparse(
                    black_box(&points),
                    black_box(n),
                    black_box(d),
                    black_box(max_dim),
                    None,
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
