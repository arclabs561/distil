//! TextRank for sentences (Mihalcea and Tarau 2004): PageRank over a graph
//! whose edges weigh shared words.
//!
//! The similarity of sentences `i` and `j` with content-word lists `S_i`,
//! `S_j` (stems, repeats kept for the lengths) is the number of distinct
//! stems they share over `ln |S_i| + ln |S_j|`; when both have one word the
//! denominator is 0 and the similarity is the shared count (sumy's rule),
//! and an empty sentence has no edges. There are no self-loops. Scores are
//! the stationary distribution of the walk with damping 0.85 (jump 0.15),
//! from the shared graph core in [`graph`](super::graph); they sum to 1.
//!
//! R. Mihalcea and P. Tarau, "TextRank: Bringing order into texts", EMNLP
//! 2004. Parity: sumy `TextRankSummarizer`, which counts every repeat of a
//! shared word, adds self-loops and stops its power iteration at 1e-4; the
//! fixture generator patches all three.

use super::graph::{similarity_matrix, stationary};
use super::{by_score, content_stems, SentenceExtractor};
use crate::text::Document;
use std::collections::HashSet;

/// TextRank over sentences.
#[derive(Debug, Clone)]
pub struct TextRank {
    stopwords: HashSet<String>,
    damping: f64,
}

impl Default for TextRank {
    fn default() -> Self {
        Self::new()
    }
}

impl TextRank {
    /// English stopwords, damping 0.85.
    pub fn new() -> Self {
        TextRank {
            stopwords: crate::default_stopwords(),
            damping: 0.85,
        }
    }

    /// Replace the stopword list.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Probability of following an edge rather than jumping (default 0.85).
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }

    /// The sentence graph's edge weights.
    pub fn similarity(&self, doc: &Document) -> Vec<Vec<f64>> {
        let terms = content_stems(doc, &self.stopwords);
        let sets: Vec<HashSet<&str>> = terms.iter().map(|s| s.iter().copied().collect()).collect();
        similarity_matrix(terms.len(), false, |i, j| {
            let shared = sets[i].intersection(&sets[j]).count() as f64;
            if shared == 0.0 {
                return 0.0;
            }
            let norm = (terms[i].len() as f64).ln() + (terms[j].len() as f64).ln();
            if norm.abs() < 1e-12 {
                shared
            } else {
                shared / norm
            }
        })
    }

    /// Each sentence's TextRank score, in document order.
    pub fn centrality(&self, doc: &Document) -> Vec<f64> {
        stationary(&self.similarity(doc), 1.0 - self.damping, None)
    }
}

impl SentenceExtractor for TextRank {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        by_score(self.centrality(doc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sentences "a b", "a b", "c d": the first two share 2 words over
    /// ln 2 + ln 2, the third is isolated. With jump 0.15 the isolated node
    /// gets only teleport mass: s_3 = 0.15/3 + 0.85 s_3 / 3 (it is dangling,
    /// so its own mass is spread uniformly), s_3 = 0.05 / (1 - 0.85/3).
    #[test]
    fn isolated_sentence_gets_teleport_mass() {
        let doc = Document::from_text("Alpha beta. Alpha beta. Gamma delta.");
        let tr = TextRank::new();
        let sim = tr.similarity(&doc);
        assert!((sim[0][1] - 2.0 / (2.0 * 2f64.ln())).abs() < 1e-12);
        assert_eq!(sim[0][0], 0.0);
        let c = tr.centrality(&doc);
        let s3 = 0.05 / (1.0 - 0.85 / 3.0);
        assert!((c[2] - s3).abs() < 1e-9, "{c:?}");
        assert!((c[0] - c[1]).abs() < 1e-12 && (c.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
}
