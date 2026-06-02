use criterion::{criterion_group, criterion_main, Criterion};
use std::fs;
use tda_core::persistent_homology;

fn load_points(path: &str) -> (Vec<f32>, usize, usize) {
    let text = fs::read_to_string(path).expect("failed to read file");
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

fn bench_persistent_homology(c: &mut Criterion) {
    let datasets = [
        ("celegans", "benches/data/celegans.txt"),
        ("vicsek", "benches/data/vicsek_300_of_300.txt"),
        ("klein_400", "benches/data/klein_400.txt"),
        ("klein_900", "benches/data/klein_900.txt"),
        ("dragon_1k", "benches/data/dragon_1000.txt"),
        ("dragon_2k", "benches/data/dragon_2000.txt"),
        ("hiv1", "benches/data/hiv1.txt"),
        ("o3_1k", "benches/data/o3_1024.txt"),
        ("o3_2k", "benches/data/o3_2048.txt"),
        ("pbmc_3k", "benches/data/pbmc3k_pca50.txt"),
    ];

    for (name, path) in datasets {
        let (points, n, d) = load_points(path);
        c.bench_function(&format!("{name}"), |b| {
            b.iter(|| persistent_homology(&points, n, d, 1, None, false, false));
        });
    }
}

criterion_group!(benches, bench_persistent_homology);
criterion_main!(benches);
