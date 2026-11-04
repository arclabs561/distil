//! The sentence-similarity graph and the random walks over it, shared by
//! [`TextRank`](super::TextRank), [`LexRank`](super::LexRank) and
//! [`BiasedLexRank`](super::BiasedLexRank).
//!
//! Build a weight matrix with [`similarity_matrix`] (optionally binarized
//! with [`threshold`]), then score its nodes with [`stationary`]. The
//! functions take any matrix, so callers can rank sentences that are not
//! from one document, or use the scores as a centrality signal without
//! selecting anything. The document-level methods expose the same scores
//! in document order through their `centrality` methods.

/// Symmetric matrix of `sim(i, j)` over `n` sentences (`sim` is called for
/// `i <= j` only); the diagonal is `sim(i, i)` with `self_loops` and 0
/// otherwise.
pub fn similarity_matrix(
    n: usize,
    self_loops: bool,
    sim: impl Fn(usize, usize) -> f64,
) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    let pairs = (0..n).flat_map(|i| (i..n).map(move |j| (i, j)));
    for (i, j) in pairs.filter(|&(i, j)| i != j || self_loops) {
        let s = sim(i, j);
        m[i][j] = s;
        m[j][i] = s;
    }
    m
}

/// Replace each weight by 1 when it exceeds `t` and 0 otherwise.
pub fn threshold(m: &mut [Vec<f64>], t: f64) {
    for w in m.iter_mut().flatten() {
        *w = if *w > t { 1.0 } else { 0.0 };
    }
}

/// Stationary distribution of a random walk that, at each step, jumps with
/// probability `jump` to a node drawn from `teleport` (uniform when `None`)
/// and otherwise follows an edge with probability proportional to its
/// weight in `weights` (row = from). A node with no out-edges jumps.
///
/// Power iteration from the uniform vector until the L1 change is below
/// 1e-12 (at most 10,000 steps). With `jump = 0` and a self-loop on every
/// node (as in LexRank's graphs) the walk is aperiodic, and on a graph of
/// several components the iteration converges to each component's
/// stationary distribution scaled by the component's share of nodes, which
/// is what sumy and the `lexrank` package compute.
pub fn stationary(weights: &[Vec<f64>], jump: f64, teleport: Option<&[f64]>) -> Vec<f64> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    let t: Vec<f64> = match teleport {
        Some(t) if t.iter().sum::<f64>() > 0.0 => {
            let s: f64 = t.iter().sum();
            t.iter().map(|x| x / s).collect()
        }
        _ => vec![1.0 / n as f64; n],
    };
    let out: Vec<f64> = weights.iter().map(|r| r.iter().sum()).collect();
    let mut p = vec![1.0 / n as f64; n];
    for _ in 0..10_000 {
        let dangling: f64 = (0..n).filter(|&i| out[i] <= 0.0).map(|i| p[i]).sum();
        let mut next: Vec<f64> = t
            .iter()
            .map(|ti| ti * (jump + (1.0 - jump) * dangling))
            .collect();
        for (u, row) in weights.iter().enumerate() {
            if out[u] > 0.0 {
                let share = (1.0 - jump) * p[u] / out[u];
                for (v, &w) in row.iter().enumerate() {
                    next[v] += share * w;
                }
            }
        }
        let diff: f64 = p.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
        p = next;
        if diff < 1e-12 {
            break;
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path a - b - c with self-loops and no jumps: a random walk on an
    /// undirected graph has stationary distribution proportional to degree,
    /// here (2, 3, 2) / 7. A separate node d with a self-loop keeps its
    /// starting share 1/4, and the path shares 3/4.
    #[test]
    fn stationary_is_proportional_to_degree_without_jumps() {
        let w = vec![
            vec![1.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let p = stationary(&w, 0.0, None);
        let want = [0.75 * 2.0 / 7.0, 0.75 * 3.0 / 7.0, 0.75 * 2.0 / 7.0, 0.25];
        for (a, b) in p.iter().zip(want) {
            assert!((a - b).abs() < 1e-9, "{p:?}");
        }
    }
}
