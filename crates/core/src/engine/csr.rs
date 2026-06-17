// Minimal CSR row storage used to build the BitCSR filtration.
//
// CSR is no longer an execution backend. It stays here only as a compact,
// sorted staging representation: point-cloud preprocessing emits CSR parts, and
// `BitCsrDistanceMatrix::from_csr` consumes row slices to build its bitsets.

use crate::engine::distance::DistanceMatrix;

/// Entries `>= NO_EDGE` are "no edge" and are excluded from the enclosing
/// radius reduction, mirroring `distance.rs`.
const NO_EDGE: f32 = f32::MAX;

/// Sparse distance rows in compressed-sparse-row form.
///
/// Stores only within-threshold neighbours. Symmetric: each kept edge appears
/// in both endpoint rows. Rows are descending by neighbour id, which is the
/// ordering BitCSR needs for deterministic filtration enumeration.
pub(crate) struct CsrDistanceMatrix {
    n: usize,
    row_ptr: Vec<usize>,
    col: Vec<u16>,
    val: Vec<f32>,
}

impl CsrDistanceMatrix {
    /// Assemble from raw CSR parts produced by `pdist_csr`.
    pub(crate) fn from_csr_parts(
        n: usize,
        row_ptr: Vec<usize>,
        col: Vec<u16>,
        val: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(row_ptr.len(), n + 1);
        debug_assert_eq!(col.len(), val.len());
        debug_assert_eq!(*row_ptr.last().unwrap_or(&0), col.len());
        debug_assert!((0..n).all(|v| col[row_ptr[v]..row_ptr[v + 1]]
            .windows(2)
            .all(|w| w[0] > w[1])));

        Self {
            n,
            row_ptr,
            col,
            val,
        }
    }

    #[inline(always)]
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    #[inline(always)]
    pub(crate) fn nnz(&self) -> usize {
        self.col.len()
    }

    #[inline(always)]
    pub(crate) fn neighbors(&self, v: usize) -> (&[u16], &[f32]) {
        let s = self.row_ptr[v];
        let e = self.row_ptr[v + 1];
        (&self.col[s..e], &self.val[s..e])
    }
}

/// Build CSR rows from a dense lower-triangular distance matrix.
///
/// This supports distance-matrix inputs and tests. Point-cloud inputs normally
/// use the fused `pdist_csr` path instead.
pub(crate) fn csr_from_distance_matrix(
    dist: &DistanceMatrix,
    threshold: f32,
) -> (CsrDistanceMatrix, f32) {
    let n = dist.n();
    if n < 2 {
        return (
            CsrDistanceMatrix {
                n,
                row_ptr: vec![0; n + 1],
                col: Vec::new(),
                val: Vec::new(),
            },
            f32::INFINITY,
        );
    }

    let raw = dist.raw();

    let mut ecc = vec![f32::NEG_INFINITY; n];
    {
        let mut p = 0;
        for i in 1..n {
            let mut row_max = ecc[i];
            for j in 0..i {
                let d = raw[p];
                p += 1;
                if d < NO_EDGE {
                    if d > row_max {
                        row_max = d;
                    }
                    if d > ecc[j] {
                        ecc[j] = d;
                    }
                }
            }
            ecc[i] = row_max;
        }
    }
    let r_cheb = ecc
        .iter()
        .copied()
        .filter(|&m| m > f32::NEG_INFINITY)
        .fold(f32::INFINITY, f32::min);
    let eff = threshold.min(r_cheb);

    let mut row_ptr = vec![0usize; n + 1];
    {
        let mut p = 0;
        for i in 1..n {
            for j in 0..i {
                let d = raw[p];
                p += 1;
                if d <= eff && d < NO_EDGE {
                    row_ptr[i + 1] += 1;
                    row_ptr[j + 1] += 1;
                }
            }
        }
        for v in 0..n {
            row_ptr[v + 1] += row_ptr[v];
        }
    }

    let nnz = row_ptr[n];
    let mut col = vec![0u16; nnz];
    let mut val = vec![0.0f32; nnz];

    {
        let mut cur = row_ptr[..n].to_vec();
        let mut p = 0;
        for i in 1..n {
            for j in 0..i {
                let d = raw[p];
                p += 1;
                if d <= eff && d < NO_EDGE {
                    let ci = cur[i];
                    col[ci] = j as u16;
                    val[ci] = d;
                    cur[i] = ci + 1;

                    let cj = cur[j];
                    col[cj] = i as u16;
                    val[cj] = d;
                    cur[j] = cj + 1;
                }
            }
        }
    }

    for v in 0..n {
        let s = row_ptr[v];
        let e = row_ptr[v + 1];
        col[s..e].reverse();
        val[s..e].reverse();
    }

    (
        CsrDistanceMatrix::from_csr_parts(n, row_ptr, col, val),
        r_cheb,
    )
}
