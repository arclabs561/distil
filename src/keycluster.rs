//! KeyCluster (Liu, Li, Zheng and Sun, 2009).
//!
//! Liu, Z., Li, P., Zheng, Y. and Sun, M. (2009). Clustering to Find
//! Exemplar Terms for Keyphrase Extraction. Proceedings of EMNLP 2009, pages
//! 257-266. ACL Anthology D09-1027.
//!
//! # Method
//!
//! 1. Candidate terms are the distinct words (stems) that are neither
//!    stopwords nor punctuation.
//! 2. Relatedness of two terms is their co-occurrence count within a window
//!    of `w = 10` tokens over the unfiltered token sequence (the paper's
//!    co-occurrence measure, `w` in 2..10), or a caller-supplied function
//!    (the paper's Wikipedia-based measures).
//! 3. Terms are clustered into `m = ceil(2n / 3)` clusters (the paper's best
//!    setting) by agglomerative clustering, and each cluster's exemplar is
//!    its medoid: the member with the largest summed relatedness to the
//!    others.
//! 4. Keyphrases are the noun phrases `(ADJ)*(NOUN|PROPN)+` (untagged text:
//!    stopword- and punctuation-delimited phrases of up to five words) that
//!    contain at least one exemplar; single-word phrases in a caller-supplied
//!    frequent-word list are dropped.
//!
//! # Deviations
//!
//! - The paper compares hierarchical, spectral and affinity propagation
//!   clustering; this uses average-linkage agglomerative clustering on the
//!   relatedness itself.
//! - The paper returns a set. Phrases are ranked by the number of exemplar
//!   words they contain, then by first occurrence.

use crate::default_stopwords;
use crate::graph::rank;
use crate::text::{delimited_phrases, is_content, noun_phrases, Candidate, Document};
use std::collections::{HashMap, HashSet};

/// KeyCluster keyphrase extractor.
#[derive(Debug, Clone)]
pub struct KeyCluster {
    stopwords: HashSet<String>,
    window: usize,
    cluster_fraction: f64,
    frequent: HashSet<String>,
}

impl Default for KeyCluster {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyCluster {
    /// Window 10, `m = 2n / 3`, English stopwords, no frequent-word list.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            window: 10,
            cluster_fraction: 2.0 / 3.0,
            frequent: HashSet::new(),
        }
    }

    /// Stopwords excluded from terms (and delimiting untagged phrases).
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Co-occurrence window in tokens (at least 2).
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(2);
        self
    }

    /// Number of clusters as a fraction of the number of terms, in `(0, 1]`.
    pub fn with_cluster_fraction(mut self, fraction: f64) -> Self {
        self.cluster_fraction = fraction.clamp(f64::MIN_POSITIVE, 1.0);
        self
    }

    /// Lowercased words too common to be single-word keyphrases.
    pub fn with_frequent_words(mut self, words: HashSet<String>) -> Self {
        self.frequent = words;
        self
    }

    /// Up to `max` keyphrases with co-occurrence relatedness, best first.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        let counts = cooccurrence(doc, &self.stopwords, self.window);
        self.rank_with(
            doc,
            &|a, b| counts.get(&key(a, b)).copied().unwrap_or(0.0),
            max,
        )
    }

    /// Up to `max` keyphrases with a caller-supplied symmetric relatedness
    /// between stems, best first.
    pub fn rank_with(
        &self,
        doc: &Document,
        relatedness: &dyn Fn(&str, &str) -> f64,
        max: usize,
    ) -> Vec<(String, f64)> {
        let exemplars = self.exemplars(doc, relatedness);
        let phrases: Vec<Candidate> = if doc.is_tagged() {
            noun_phrases(doc, usize::MAX)
        } else {
            delimited_phrases(doc, &self.stopwords, 5)
        };
        let scored = phrases.iter().filter_map(|c| {
            let hits = c
                .lexical_form
                .iter()
                .filter(|w| exemplars.contains(*w))
                .count();
            let frequent = c.len() == 1 && self.frequent.contains(&c.surface);
            (hits > 0 && !frequent).then_some((c, hits as f64))
        });
        rank(scored, max)
    }

    /// The exemplar stems.
    pub fn exemplars(
        &self,
        doc: &Document,
        relatedness: &dyn Fn(&str, &str) -> f64,
    ) -> HashSet<String> {
        let mut terms: Vec<&str> = Vec::new();
        for t in doc.tokens().filter(|t| is_content(t, &self.stopwords)) {
            if !terms.contains(&t.stem.as_str()) {
                terms.push(&t.stem);
            }
        }
        let n = terms.len();
        let sim: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            0.0
                        } else {
                            relatedness(terms[i], terms[j])
                        }
                    })
                    .collect()
            })
            .collect();
        let m = ((n as f64 * self.cluster_fraction).ceil() as usize).max(1);
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        while clusters.len() > m {
            let mut best = (0, 1, f64::NEG_INFINITY);
            for a in 0..clusters.len() {
                for b in a + 1..clusters.len() {
                    let total: f64 = clusters[a]
                        .iter()
                        .flat_map(|&i| clusters[b].iter().map(move |&j| (i, j)))
                        .map(|(i, j)| sim[i][j])
                        .sum();
                    let avg = total / (clusters[a].len() * clusters[b].len()) as f64;
                    if avg > best.2 {
                        best = (a, b, avg);
                    }
                }
            }
            let merged = clusters.remove(best.1);
            clusters[best.0].extend(merged);
        }
        clusters
            .iter()
            .map(|c| {
                let score = |i: usize| c.iter().map(|&j| sim[i][j]).sum::<f64>();
                let mut best = c[0];
                for &i in c {
                    if score(i) > score(best) || (score(i) == score(best) && i < best) {
                        best = i;
                    }
                }
                terms[best].to_string()
            })
            .collect()
    }
}

fn key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Co-occurrence counts of content stems fewer than `window` tokens apart,
/// counting every token.
fn cooccurrence(
    doc: &Document,
    stopwords: &HashSet<String>,
    window: usize,
) -> HashMap<(String, String), f64> {
    let tokens: Vec<_> = doc.tokens().collect();
    let mut counts = HashMap::new();
    for (i, a) in tokens.iter().enumerate() {
        if !is_content(a, stopwords) {
            continue;
        }
        for b in tokens.iter().skip(i + 1).take(window - 1) {
            if is_content(b, stopwords) && a.stem != b.stem {
                *counts.entry(key(&a.stem, &b.stem)).or_insert(0.0) += 1.0;
            }
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two groups related only inside: {graph, node, edge} and {cell, gene}.
    /// With m = ceil(5 * 2/5) = 2 the clusters are the groups; the medoids
    /// are "node" (related 3 + 3) and "cell" or "gene" (tie, earlier wins).
    #[test]
    fn exemplars_are_cluster_medoids() {
        let doc = Document::from_text("graph node edge cell gene");
        let rel = |a: &str, b: &str| -> f64 {
            let pair = |x: &str, y: &str| (a == x && b == y) || (a == y && b == x);
            if pair("graph", "node") || pair("node", "edg") {
                3.0
            } else if pair("graph", "edg") {
                1.0
            } else if pair("cell", "gene") {
                2.0
            } else {
                0.0
            }
        };
        let k = KeyCluster::new().with_cluster_fraction(0.4);
        let ex = k.exemplars(&doc, &rel);
        assert_eq!(ex, HashSet::from(["node".to_string(), "cell".to_string()]));
    }

    #[test]
    fn phrases_need_an_exemplar_and_skip_frequent_words() {
        let doc = Document::from_text("graph node, cell gene, node");
        let rel = |a: &str, b: &str| f64::from((a == "node") != (b == "node"));
        let k = KeyCluster::new().with_cluster_fraction(0.5);
        let got: Vec<String> = k
            .rank_with(&doc, &rel, 10)
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        assert!(got.contains(&"node".to_string()), "{got:?}");
        let k = k.with_frequent_words(HashSet::from(["node".to_string()]));
        let got: Vec<String> = k
            .rank_with(&doc, &rel, 10)
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        assert!(!got.contains(&"node".to_string()), "{got:?}");
    }
}
