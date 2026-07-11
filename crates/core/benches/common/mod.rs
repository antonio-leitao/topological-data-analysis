#![allow(dead_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use tda_core::{persistent_homology, BarcodeResult};

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

/// Which optional preprocessing to exercise on top of the vanilla pipeline.
#[derive(Clone, Copy)]
pub struct Variant {
    pub name: &'static str,
    pub peel: bool,
    pub quotient: bool,
    pub collapse: bool,
}

pub const VANILLA: Variant = Variant {
    name: "vanilla",
    peel: false,
    quotient: false,
    collapse: false,
};

pub fn parse_variant(value: &str) -> Result<Variant, String> {
    match value {
        "" | "vanilla" => Ok(VANILLA),
        "peel" => Ok(Variant {
            name: "peel",
            peel: true,
            quotient: false,
            collapse: false,
        }),
        "quot" => Ok(Variant {
            name: "quot",
            peel: false,
            quotient: true,
            collapse: false,
        }),
        "clps" => Ok(Variant {
            name: "clps",
            peel: false,
            quotient: false,
            collapse: true,
        }),
        other => Err(format!("variant must be vanilla|peel|quot|clps, got {other}")),
    }
}

pub fn variant_from_env() -> Variant {
    parse_variant(env::var("TDA_VARIANT").ok().as_deref().unwrap_or("vanilla"))
        .expect("TDA_VARIANT must be vanilla, peel, quot, or clps")
}

/// Group / directory label: the historical `bitcsr_h{d}` for vanilla (keeps
/// existing criterion baselines comparable), `bitcsr_h{d}_{variant}` otherwise.
pub fn group_name(max_dim: usize, variant: Variant) -> String {
    if variant.peel || variant.quotient {
        format!("bitcsr_h{max_dim}_{}", variant.name)
    } else {
        format!("bitcsr_h{max_dim}")
    }
}

pub fn parse_max_dim(value: &str) -> Result<usize, String> {
    match value {
        "" | "1" => Ok(1),
        "2" => Ok(2),
        other => Err(format!("max_dim must be 1 or 2, got {other}")),
    }
}

/// The datasets to run are controlled by data/datasets.txt (order included).
pub fn dataset_files() -> Vec<String> {
    fs::read_to_string(data_dir().join("datasets.txt"))
        .expect("failed to read data/datasets.txt")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect()
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

/// One full end-to-end run (preprocessing included, so a variant's own cost is
/// charged against its engine savings).
pub fn run_once(
    dataset: &Dataset,
    max_dim: usize,
    variant: Variant,
) -> tda_core::Result<BarcodeResult> {
    persistent_homology(
        &dataset.points,
        dataset.n,
        max_dim,
        None,
        false,
        variant.quotient,
        variant.peel,
        variant.collapse,
        false, //benches should not be parallel
    )
}
