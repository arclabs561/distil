//! MultipartiteRank (Boudin, 2018).
//!
//! Boudin, F. (2018). Unsupervised Keyphrase Extraction with Multipartite
//! Graphs. Proceedings of NAACL-HLT 2018 (Volume 2: Short Papers), pages
//! 667-672. <https://aclanthology.org/N18-2105/>
//!
//! # Method
//!
//! 1. Candidates and topics are TopicRank's ([`crate::topicrank`]).
//! 2. Candidates are nodes of a directed complete k-partite graph: every
//!    pair of candidates from different topics is linked both ways with
//!    weight (Eq. 1)
//!
//!    $$w_{ij} = \sum_{p_i \in P(c_i)} \sum_{p_j \in P(c_j)} \frac{1}{|p_i - p_j|}$$
//!
//! 3. The first-occurring candidate `c_j` of each topic is promoted by
//!    raising its incoming weights (Eq. 2):
//!
//!    $$w_{ij} = w_{ij} + \alpha \cdot e^{1/p_j} \cdot \sum_{c_k \in T(c_j) \setminus \{c_j\}} w_{ki}$$
//!
//!    with `p_j` the 1-based position of `c_j`'s first occurrence and
//!    `alpha = 1.1`.
//! 4. Candidates are ranked by weighted PageRank, `lambda = 0.85`.
//!
//! # Deviations
//!
//! - The same choices as [`crate::topicrank`] for clustering and gaps.
//! - Eq. 2 is written with pke's reading: the boost lands on edges into the
//!   promoted candidate and uses that candidate's position.
//! - Ties are broken by first occurrence.

use crate::graph::{pagerank, rank};
use crate::text::Document;
use crate::topicrank::{cluster, proximity, topic_candidates};
use crate::{default_stopwords, KeywordExtractor};
use std::collections::HashSet;

/// MultipartiteRank keyphrase extractor.
#[derive(Debug, Clone)]
pub struct MultipartiteRank {
    stopwords: HashSet<String>,
    threshold: f64,
    alpha: f64,
}

impl Default for MultipartiteRank {
    fn default() -> Self {
        Self::new()
    }
}

impl MultipartiteRank {
    /// Clustering threshold 0.74, `alpha = 1.1`, English stopwords.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            threshold: 0.74,
            alpha: 1.1,
        }
    }

    /// Stopwords for candidate filtering (and selection in untagged text).
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Largest average-linkage distance at which topic clusters merge.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        if threshold.is_finite() {
            self.threshold = threshold;
        }
        self
    }

    /// Strength of the first-occurrence promotion; 0 disables it.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        if alpha.is_finite() && alpha >= 0.0 {
            self.alpha = alpha;
        }
        self
    }

    /// Up to `max` keyphrases, best first.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        let candidates = topic_candidates(doc, &self.stopwords);
        let topics = cluster(&candidates, self.threshold);
        let n = candidates.len();
        let mut topic_of = vec![0; n];
        for (t, members) in topics.iter().enumerate() {
            for &c in members {
                topic_of[c] = t;
            }
        }
        let mut adj = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i + 1..n {
                if topic_of[i] != topic_of[j] {
                    let w = proximity(&candidates[i], &candidates[j]);
                    adj[i][j] = w;
                    adj[j][i] = w;
                }
            }
        }
        if self.alpha > 0.0 {
            // Compute every boost from the unadjusted weights, then apply.
            let mut boosts = Vec::new();
            for members in topics.iter().filter(|m| m.len() > 1) {
                let first = *members
                    .iter()
                    .min_by_key(|&&c| candidates[c].first_offset())
                    .expect("topics are non-empty");
                let position = 1.0 / (1 + candidates[first].first_offset()) as f64;
                for end in (0..n).filter(|&e| adj[first][e] > 0.0) {
                    let sum: f64 = members
                        .iter()
                        .filter(|&&v| v != first)
                        .map(|&v| adj[v][end])
                        .sum();
                    boosts.push((end, first, sum * self.alpha * position.exp()));
                }
            }
            for (from, to, b) in boosts {
                adj[from][to] += b;
            }
        }
        let scores = pagerank(&adj, None);
        rank(candidates.iter().zip(scores), max)
    }
}

impl KeywordExtractor for MultipartiteRank {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.rank(&Document::from_text(text), max_keywords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "Neural networks model language. A neural network needs data. \
                        Language models predict words from data.";

    #[test]
    fn promotion_favours_the_first_variant() {
        let doc = Document::from_text(TEXT);
        let plain = MultipartiteRank::new().with_alpha(0.0).rank(&doc, 20);
        let boosted = MultipartiteRank::new().rank(&doc, 20);
        let score = |r: &[(String, f64)], p: &str| r.iter().find(|(q, _)| q == p).unwrap().1;
        let first = "neural networks model language";
        let second = "neural network needs data";
        assert!(
            score(&boosted, first) / score(&boosted, second)
                > score(&plain, first) / score(&plain, second),
            "{plain:?} {boosted:?}"
        );
    }

    #[test]
    fn single_candidate() {
        let got = MultipartiteRank::new().extract("graphs", 5);
        assert_eq!(got.len(), 1);
        assert!((got[0].1 - 1.0).abs() < 1e-12);
    }
}
