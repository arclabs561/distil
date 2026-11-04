//! PositionRank (Florescu and Caragea, 2017).
//!
//! Florescu, C. and Caragea, C. (2017). PositionRank: An Unsupervised
//! Approach to Keyphrase Extraction from Scholarly Documents. Proceedings of
//! ACL 2017 (Volume 1: Long Papers), pages 1105-1115.
//! <https://aclanthology.org/P17-1102/>
//!
//! # Method
//!
//! 1. Nodes are stems of nouns and adjectives; edges are weighted by
//!    co-occurrence counts within a window of `w` tokens (default 10).
//! 2. Each word's bias is the sum of its inverse positions, counted over all
//!    tokens and starting at 1: a word at positions 2, 5 and 10 gets
//!    `1/2 + 1/5 + 1/10`. The biases are normalized into `p~`.
//! 3. Scores are position-biased PageRank with `alpha = 0.85` (Eq. 2):
//!
//!    $$S(v_i) = (1 - \alpha) \tilde{p}_i + \alpha \sum_{v_j \in Adj(v_i)} \frac{w_{ji}}{O(v_j)} S(v_j)$$
//!
//! 4. Candidates are noun phrases `(adjective)*(noun)+` of up to three
//!    words, scored by the sum of their words' scores.
//!
//! Untagged text uses non-stopword, non-punctuation graph words and
//! stopword-delimited candidates of up to three words.
//!
//! # Deviations
//!
//! - The paper tunes `w` from 2 to 10; the default 10 is pke's.
//! - Iteration runs to an L1 change of 1e-10 (the paper: 0.001 or 100
//!   iterations). Dangling nodes teleport by the bias vector, as networkx
//!   does.

use crate::graph::{graph_words, rank_by_word_sum, WordGraph};
use crate::text::{delimited_phrases, noun_phrases, Document, Roles};
use crate::KeywordExtractor;
use std::collections::{HashMap, HashSet};

/// PositionRank keyphrase extractor.
#[derive(Debug, Clone)]
pub struct PositionRank {
    stopwords: Roles,
    window: usize,
    max_words: usize,
}

impl Default for PositionRank {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionRank {
    /// Window 10, phrases of up to three words, English stopwords for
    /// untagged text.
    pub fn new() -> Self {
        Self {
            stopwords: Roles::default(),
            window: 10,
            max_words: 3,
        }
    }

    /// Stopwords used for untagged text.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords.set_stopwords(stopwords);
        self
    }

    /// Stopword language: its list replaces the current one, and text given
    /// to `extract` is stemmed with its stemmer. Sentences of a document in
    /// another language use that language's stopwords
    /// ([`Stoplist`](crate::text::Stoplist)).
    pub fn with_language(mut self, language: crate::Language) -> Self {
        self.stopwords.set_language(language);
        self
    }

    /// Replace the extra words that end candidate phrases in untagged text
    /// but still count as content words for scoring. The default is the
    /// SMART list (570 words) in English sentences and nothing in other
    /// languages. A list given here applies to sentences in the method's
    /// language; include SMART yourself to extend it with a domain list
    /// ("et", "al", "fig"), or pass an empty list for NLTK's list alone.
    pub fn with_delimiters<I, S>(mut self, words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.stopwords.set_delimiters(words);
        self
    }

    /// Co-occurrence window in tokens (at least 2).
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(2);
        self
    }

    /// Longest candidate phrase in words (at least 1).
    pub fn with_max_words(mut self, max_words: usize) -> Self {
        self.max_words = max_words.max(1);
        self
    }

    /// Position-biased score of each graph word (stem).
    pub fn word_scores(&self, doc: &Document) -> HashMap<String, f64> {
        let accept = graph_words(self.stopwords.language());
        let mut g = WordGraph::default();
        g.add_document(doc, &accept, self.window, 1.0);
        let mut bias: HashMap<&str, f64> = HashMap::new();
        for (i, (t, l)) in doc.tokens_in().enumerate() {
            if accept(t, l) {
                *bias.entry(t.stem.as_str()).or_insert(0.0) += 1.0 / (i + 1) as f64;
            }
        }
        let p: Vec<f64> = g.nodes.iter().map(|w| bias[w.as_str()]).collect();
        g.scores(Some(&p))
    }

    /// Up to `max` keyphrases, best first.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        let candidates = if doc.is_tagged() {
            noun_phrases(doc, self.max_words)
        } else {
            delimited_phrases(doc, &self.stopwords.delimiting(doc), self.max_words)
        };
        rank_by_word_sum(&candidates, &self.word_scores(doc), max)
    }
}

impl KeywordExtractor for PositionRank {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.rank(&self.stopwords.document(text), max_keywords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two disconnected words, each alone: their scores are their biases.
    /// "alpha" at position 1 has bias 1, "beta" at position 3 has 1/3.
    #[test]
    fn isolated_words_score_their_normalized_bias() {
        let doc = Document::from_text("alpha. beta");
        let s = PositionRank::new().with_window(2).word_scores(&doc);
        assert!((s["alpha"] - 0.75).abs() < 1e-9, "{s:?}");
        assert!((s["beta"] - 0.25).abs() < 1e-9, "{s:?}");
    }

    #[test]
    fn earlier_words_win_ties_in_frequency() {
        let got = PositionRank::new().extract("solar power and wind power", 5);
        let solar = got.iter().position(|(p, _)| p == "solar power").unwrap();
        let wind = got.iter().position(|(p, _)| p == "wind power").unwrap();
        assert!(solar < wind, "{got:?}");
    }
}
