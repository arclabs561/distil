//! Latent semantic analysis: rank sentences by their weight in the leading
//! singular vectors of the term-sentence matrix.
//!
//! The matrix `A` has a row per content-word stem and a column per
//! sentence; a stem occurring `c` times in a sentence whose most frequent
//! stem occurs `m` times gets `0.4 + 0.6 c / m` (augmented term frequency,
//! as in sumy), and 0 when absent. With `A = U S V^T` and `r` dimensions:
//!
//! - [`LsaVariant::SteinbergerJezek`] (default) scores sentence `j` by its
//!   length in the reduced space, `sqrt(sum over k < r of s_k^2 v_jk^2)`.
//! - [`LsaVariant::GongLiu`] takes, for `k = 0, 1, ...`, the not yet chosen
//!   sentence with the largest `|v_jk|` and scores it `s_k`; sentences left
//!   after `r` dimensions follow in document order with score 0.
//!
//! `V` and `S^2` come from the eigendecomposition of `A^T A` (one row and
//! column per sentence) by the cyclic Jacobi method, which needs no
//! linear-algebra dependency and is exact to rounding for the few hundred
//! sentences of a document. `r` defaults to every dimension, sumy's
//! default (sumy uses `max(3, ratio * rank)`); Steinberger and Jezek keep
//! the dimensions whose singular value is at least half the largest.
//!
//! Y. Gong and X. Liu, "Generic text summarization using relevance measure
//! and latent semantic analysis", SIGIR 2001. J. Steinberger and K. Jezek,
//! "Using latent semantic analysis in text summarization and summary
//! evaluation", ISIM 2004. Parity: sumy `LsaSummarizer`, which also adds
//! 0.4 to the zero cells of a sentence's column (the fixture generator
//! patches this to the augmented frequency above).

use super::{by_score, content_stems, SentenceExtractor};
use crate::text::{Document, Roles};
use std::collections::HashSet;

/// Which LSA sentence score to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LsaVariant {
    /// One sentence per singular vector (Gong and Liu 2001).
    GongLiu,
    /// Length in the reduced space (Steinberger and Jezek 2004).
    #[default]
    SteinbergerJezek,
}

/// LSA sentence ranking.
#[derive(Debug, Clone)]
pub struct Lsa {
    stopwords: Roles,
    variant: LsaVariant,
    dimensions: Option<usize>,
}

impl Default for Lsa {
    fn default() -> Self {
        Self::new()
    }
}

impl Lsa {
    /// Steinberger and Jezek's score over all dimensions, English stopwords.
    pub fn new() -> Self {
        Lsa {
            stopwords: Roles::default(),
            variant: LsaVariant::default(),
            dimensions: None,
        }
    }

    /// Replace the stopword list.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords.set_stopwords(stopwords);
        self
    }

    /// Stopword language: its list replaces the current one. Sentences of a
    /// document in another language use that language's stopwords
    /// ([`Stoplist`](crate::text::Stoplist)).
    pub fn with_language(mut self, language: crate::Language) -> Self {
        self.stopwords.set_language(language);
        self
    }

    /// Choose the sentence score.
    pub fn with_variant(mut self, variant: LsaVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Keep only the `r` largest singular values.
    pub fn with_dimensions(mut self, r: usize) -> Self {
        self.dimensions = Some(r);
        self
    }
}

impl SentenceExtractor for Lsa {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        let terms = content_stems(doc, self.stopwords.language());
        let n = terms.len();
        let columns: Vec<super::Vector> = terms
            .iter()
            .map(|s| {
                let mut c = super::Vector::new();
                for &w in s {
                    *c.entry(w).or_insert(0.0) += 1.0;
                }
                let m = c.values().copied().fold(0.0, f64::max);
                c.values_mut().for_each(|x| *x = 0.4 + 0.6 * *x / m);
                c
            })
            .collect();
        let gram =
            super::graph::similarity_matrix(n, true, |i, j| super::dot(&columns[i], &columns[j]));
        let (values, vectors) = symmetric_eigen(gram);
        let r = self.dimensions.unwrap_or(n).min(n);
        match self.variant {
            LsaVariant::SteinbergerJezek => by_score(
                (0..n)
                    .map(|j| {
                        (0..r)
                            .map(|k| values[k].max(0.0) * vectors[k][j].powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect(),
            ),
            LsaVariant::GongLiu => {
                let mut taken = vec![false; n];
                let mut out = Vec::with_capacity(n);
                for k in 0..r {
                    let best = (0..n).filter(|&j| !taken[j]).max_by(|&a, &b| {
                        vectors[k][a]
                            .abs()
                            .total_cmp(&vectors[k][b].abs())
                            .then(b.cmp(&a))
                    });
                    if let Some(j) = best {
                        taken[j] = true;
                        out.push((j, values[k].max(0.0).sqrt()));
                    }
                }
                out.extend((0..n).filter(|&j| !taken[j]).map(|j| (j, 0.0)));
                out
            }
        }
    }
}

/// Eigenvalues (descending) and unit eigenvectors (`vectors[k]` belongs to
/// `values[k]`) of a symmetric matrix, by cyclic Jacobi rotations.
pub(crate) fn symmetric_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let scale: f64 = a.iter().flatten().map(|x| x * x).sum();
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| a[p][q] * a[p][q])
            .sum();
        if off <= 1e-30 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
                let (head, tail) = a.split_at_mut(q);
                for (x, y) in head[p].iter_mut().zip(tail[0].iter_mut()) {
                    let (u, w) = (*x, *y);
                    *x = c * u - s * w;
                    *y = s * u + c * w;
                }
                for row in v.iter_mut() {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]).then(i.cmp(&j)));
    let values = order.iter().map(|&k| a[k][k]).collect();
    let vectors = order
        .iter()
        .map(|&k| v.iter().map(|row| row[k]).collect())
        .collect();
    (values, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_decomposition_reconstructs_the_matrix() {
        let a = vec![
            vec![4.0, 1.0, 2.0, 0.5],
            vec![1.0, 3.0, 0.0, 1.0],
            vec![2.0, 0.0, 5.0, 1.5],
            vec![0.5, 1.0, 1.5, 2.0],
        ];
        let (values, vectors) = symmetric_eigen(a.clone());
        assert!(values.windows(2).all(|w| w[0] >= w[1]));
        for i in 0..4 {
            for j in 0..4 {
                let r: f64 = (0..4)
                    .map(|k| values[k] * vectors[k][i] * vectors[k][j])
                    .sum();
                assert!((r - a[i][j]).abs() < 1e-10, "{i} {j}: {r}");
            }
        }
        // Trace is the sum of eigenvalues.
        assert!((values.iter().sum::<f64>() - 14.0).abs() < 1e-10);
    }

    /// Sentences "a", "a", "b": columns e_a, e_a, e_b (each weight 1).
    /// `A^T A` has eigenvalues 2 (vector (1, 1, 0)/sqrt 2) and 1 (e_3), so
    /// all-dimension lengths are 1, 1, 1 and the first dimension alone gives
    /// 1, 1, 0. Gong and Liu pick sentence 0 for dimension 0 (tie with 1
    /// goes to the earlier) and sentence 2 for dimension 1.
    #[test]
    fn variants_follow_the_definitions() {
        let doc = Document::from_text("Alpha. Alpha. Beta.");
        let full = Lsa::new().rank(&doc);
        assert!(full.iter().all(|x| (x.1 - 1.0).abs() < 1e-9), "{full:?}");
        let one = Lsa::new().with_dimensions(1).rank(&doc);
        assert_eq!(one[2].0, 2);
        assert!(one[2].1.abs() < 1e-9 && (one[0].1 - 1.0).abs() < 1e-9);
        let gl = Lsa::new().with_variant(LsaVariant::GongLiu).rank(&doc);
        let order: Vec<usize> = gl.iter().map(|x| x.0).collect();
        assert_eq!(order, vec![0, 2, 1]);
        assert!((gl[0].1 - 2f64.sqrt()).abs() < 1e-9 && (gl[1].1 - 1.0).abs() < 1e-9);
    }
}
