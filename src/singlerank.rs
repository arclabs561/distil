//! SingleRank (Wan and Xiao, 2008).
//!
//! Wan, X. and Xiao, J. (2008). Single Document Keyphrase Extraction Using
//! Neighborhood Knowledge. Proceedings of AAAI 2008, pages 855-860.
//! SingleRank is that paper's baseline: ExpandRank with no neighbours.
//!
//! # Method
//!
//! 1. Nodes are stems of nouns and adjectives (untagged text: words that are
//!    neither stopwords nor punctuation). The edge between two words is
//!    weighted by how often they co-occur within a window of `w = 10`
//!    tokens.
//! 2. Word scores are weighted PageRank with `mu = 0.85`:
//!
//!    $$WS(v_i) = \mu \sum_{j} WS(v_j) \frac{M_{j,i}}{\sum_k M_{j,k}} + \frac{1 - \mu}{|V|}$$
//!
//! 3. Candidates are the longest sequences of nouns and adjectives
//!    (untagged text: stopword- and punctuation-delimited phrases of up to
//!    five words); a phrase scores the sum of its word scores.
//!
//! # Deviations
//!
//! - Candidates follow pke: the paper also drops phrases that end in an
//!   adjective (see [`crate::expandrank`], which keeps that rule).
//! - The window counts all tokens and ignores sentence boundaries, as in
//!   pke. Iteration runs to an L1 change of 1e-10 (the paper: 1e-4).
//! - Ties are broken by first occurrence.

use crate::graph::{is_graph_word, rank_by_word_sum, WordGraph};
use crate::text::{delimited_phrases, longest_sequences, Candidate, Document, Pos};
use crate::{default_stopwords, KeywordExtractor};
use std::collections::{HashMap, HashSet};

/// SingleRank keyphrase extractor.
#[derive(Debug, Clone)]
pub struct SingleRank {
    stopwords: HashSet<String>,
    window: usize,
}

impl Default for SingleRank {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleRank {
    /// Window 10, English stopwords for untagged text.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            window: 10,
        }
    }

    /// Stopwords used for untagged text.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Co-occurrence window in tokens (at least 2).
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(2);
        self
    }

    /// Score of each graph word (stem).
    pub fn word_scores(&self, doc: &Document) -> HashMap<String, f64> {
        let mut g = WordGraph::default();
        g.add_document(
            doc,
            &|t| is_graph_word(t, &self.stopwords),
            self.window,
            1.0,
        );
        g.scores(None)
    }

    /// Up to `max` keyphrases, best first.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        let candidates = noun_adjective_runs(doc, &self.stopwords);
        rank_by_word_sum(&candidates, &self.word_scores(doc), max)
    }
}

/// Longest noun/adjective sequences when tagged, otherwise delimited phrases
/// of at most five words.
pub(crate) fn noun_adjective_runs(doc: &Document, stopwords: &HashSet<String>) -> Vec<Candidate> {
    if doc.is_tagged() {
        longest_sequences(doc, |t| t.pos.is_some_and(Pos::is_noun_or_adjective))
    } else {
        delimited_phrases(doc, stopwords, 5)
    }
}

impl KeywordExtractor for SingleRank {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.rank(&Document::from_text(text), max_keywords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequent_connected_phrase_ranks_first() {
        let text = "Graph ranking scores words. Graph ranking needs a word graph. \
                    Graph ranking is simple.";
        let got = SingleRank::new().extract(text, 3);
        assert!(got[0].0.contains("graph ranking"), "{got:?}");
    }
}
