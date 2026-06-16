#![allow(dead_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use tda_core::{persistent_homology, persistent_homology_sparse, BarcodeResult};

const H2_LAST_DATASET: &str = "hiv1.txt";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Dense,
    Sparse,
}

impl Mode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "dense" => Ok(Self::Dense),
            "sparse" => Ok(Self::Sparse),
            other => Err(format!("mode must be dense or sparse, got {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Sparse => "sparse",
        }
    }
}

pub struct Dataset {
    pub name: String,
    pub points: Vec<f32>,
    pub n: usize,
    pub d: usize,
}

pub fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

pub fn max_dim_from_env() -> usize {
    parse_max_dim(env::var("TDA_MAX_DIM").ok().as_deref().unwrap_or("1"))
        .expect("TDA_MAX_DIM must be 1 or 2")
}

pub fn parse_max_dim(value: &str) -> Result<usize, String> {
    match value {
        "" | "1" => Ok(1),
        "2" => Ok(2),
        other => Err(format!("max_dim must be 1 or 2, got {other}")),
    }
}

pub fn bench_mode_from_env() -> String {
    match env::var("TDA_BENCH_MODE").ok().as_deref() {
        None | Some("") | Some("both") => "both".to_owned(),
        Some("dense") => "dense".to_owned(),
        Some("sparse") => "sparse".to_owned(),
        Some(other) => panic!("TDA_BENCH_MODE must be dense, sparse, or both, got {other}"),
    }
}

pub fn should_bench(mode: Mode) -> bool {
    let bench_mode = bench_mode_from_env();
    bench_mode == "both" || bench_mode == mode.as_str()
}

pub fn dataset_files(max_dim: usize) -> Vec<String> {
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

pub fn dataset_name(file: &Path) -> String {
    match file.file_stem().and_then(|stem| stem.to_str()) {
        Some(name) => name.to_owned(),
        None => file.to_string_lossy().into_owned(),
    }
}

pub fn resolve_dataset_path(file: &str) -> PathBuf {
    let path = PathBuf::from(file);
    if path.exists() {
        path
    } else {
        data_dir().join(file)
    }
}

pub fn load_dataset(file: &str) -> Dataset {
    let path = resolve_dataset_path(file);
    let text = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!("failed to read {}: {err}", path.display());
    });
    let mut n = 0usize;
    let mut d = 0usize;
    let points: Vec<f32> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .flat_map(|line| {
            let row: Vec<f32> = line
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            if n == 0 {
                d = row.len();
            } else {
                assert_eq!(row.len(), d, "jagged input at row {n}");
            }
            n += 1;
            row
        })
        .collect();

    Dataset {
        name: dataset_name(&path),
        points,
        n,
        d,
    }
}

pub fn run_once(
    mode: Mode,
    dataset: &Dataset,
    max_dim: usize,
    parallel: bool,
) -> tda_core::Result<BarcodeResult> {
    match mode {
        Mode::Dense => persistent_homology(
            &dataset.points,
            dataset.n,
            dataset.d,
            max_dim,
            None,
            false,
            false,
        ),
        Mode::Sparse => persistent_homology_sparse(
            &dataset.points,
            dataset.n,
            dataset.d,
            max_dim,
            None,
            parallel,
        ),
    }
}

/// Whether the sparse backend should parallelise. Defaults to `true`; set
/// `TDA_PARALLEL=0`/`false` to measure or profile the single-thread path.
pub fn parallel_from_env() -> bool {
    !matches!(
        std::env::var("TDA_PARALLEL").ok().as_deref(),
        Some("0") | Some("false") | Some("no")
    )
}
