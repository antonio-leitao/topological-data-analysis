use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tda_core::{filtration_size_with, persistent_homology_with, Config};

const VARIANTS: [(&str, bool, bool, bool); 7] = [
    ("Vanilla", false, false, false),
    ("Peel", false, true, false),
    ("Clps", false, false, true),
    ("Peel+Clps", false, true, true),
    ("Quot", true, false, false),
    ("Quot+Peel", true, true, false),
    ("Q+P+C", true, true, true),
];

const N: usize = VARIANTS.len();
const W_NAME: usize = 12;
const W_COL: usize = 9;

fn cfg(quotient: bool, peel: bool, collapse: bool, parallel: bool) -> Config {
    Config {
        quotient,
        peel,
        collapse,
        parallel,
        ..Config::default()
    }
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

fn dataset_files() -> Vec<String> {
    fs::read_to_string(data_dir().join("datasets.txt"))
        .expect("failed to read data/datasets.txt")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect()
}

fn load_points(path: &str) -> (Vec<f32>, usize, usize) {
    let text = fs::read_to_string(data_dir().join(path))
        .unwrap_or_else(|_| panic!("Failed to read file: {}", path));
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
                assert_eq!(row.len(), d, "Jagged input at row {n}");
            }
            n += 1;
            row
        })
        .collect();
    (flat, n, d)
}

fn print_header() {
    print!("{:<w$}", "Dataset", w = W_NAME);
    for (label, ..) in VARIANTS {
        print!(" | {:>w$}", label, w = W_COL);
    }
    println!();

    print!("{:-<w$}", "", w = W_NAME);
    for _ in VARIANTS {
        print!("-+-{:-<w$}", "", w = W_COL);
    }
    println!();
}

fn print_row(name: &str, cells: &[String; N]) {
    print!("{:<w$}", name, w = W_NAME);
    for cell in cells {
        print!(" | {:>w$}", cell, w = W_COL);
    }
    println!();
}

fn main() {
    let is_parallel = env::args().any(|arg| arg == "--parallel" || arg == "parallel");
    let max_dim = 2;

    println!(
        "=== RUNTIME (max_dim = {}, parallel = {}) ===",
        max_dim, is_parallel
    );
    print_header();

    for path in dataset_files() {
        let name = path.strip_suffix(".txt").unwrap_or(&path);
        let (points, n, _d) = load_points(&path);

        let mut cells: [String; N] = std::array::from_fn(|_| String::new());
        for (i, (_, q, p, c)) in VARIANTS.iter().copied().enumerate() {
            let start = Instant::now();
            let _ =
                persistent_homology_with(&points, n, max_dim, &cfg(q, p, c, is_parallel)).unwrap();
            cells[i] = format!("{:.2?}", start.elapsed());
        }

        print_row(name, &cells);
    }

    println!();

    struct Row {
        name: String,
        sizes: [usize; N],
        times: [Duration; N],
    }

    let mut rows: Vec<Row> = Vec::new();

    for path in dataset_files() {
        let name = path.strip_suffix(".txt").unwrap_or(&path).to_owned();
        let (points, n, _d) = load_points(&path);

        let mut sizes = [0usize; N];
        let mut times = [Duration::ZERO; N];

        for (i, (_, q, p, c)) in VARIANTS.iter().copied().enumerate() {
            let start = Instant::now();
            sizes[i] =
                filtration_size_with(&points, n, max_dim, &cfg(q, p, c, is_parallel)).unwrap();
            times[i] = start.elapsed();
        }

        rows.push(Row { name, sizes, times });
    }

    println!("=== FILTRATION SIZE ===");
    print_header();
    for r in &rows {
        let cells: [String; N] = std::array::from_fn(|i| r.sizes[i].to_string());
        print_row(&r.name, &cells);
    }

    println!();

    println!("=== FILTRATION SIZE RUNTIME ===");
    print_header();
    for r in &rows {
        let cells: [String; N] = std::array::from_fn(|i| format!("{:.2?}", r.times[i]));
        print_row(&r.name, &cells);
    }
}
