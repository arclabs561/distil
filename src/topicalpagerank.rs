//! Topical PageRank, single-PageRank variant (Sterckx et al., 2015).
//!
//! Liu, Z., Huang, W., Zheng, Y. and Sun, M. (2010). Automatic Keyphrase
//! Extraction via Topic Decomposition. Proceedings of EMNLP 2010, pages
//! 366-376. <https://aclanthology.org/D10-1036/>
//!
//! Sterckx, L., Demeester, T., Deleu, J. and Develder, C. (2015). Topical
//! Word Importance for Fast Keyphrase Extraction. Proceedings of the 24th
//! International Conference on World Wide Web (WWW '15 Companion), pages
//! 121-122. <https://doi.org/10.1145/2740908.2742730>
//!
//! Liu et al. run one biased PageRank per topic and combine them. This
//! module implements the single-PageRank variant of Sterckx et al., which
//! is also what pke's `TopicalPageRank` implements.
//!
//! # Method
//!
//! The caller supplies a topic model: for each word (stem) its vector
//! `P(w|Z) = (P(w|z_1), ..., P(w|z_K))`, and for the document
//! `P(Z|d) = (P(z_1|d), ..., P(z_K|d))`.
//!
//! 1. The word graph is SingleRank's: noun and adjective stems, edges
//!    weighted by co-occurrence counts within a window of 10 tokens.
//! 2. Each word's topical importance is the cosine similarity (Eq. 3)
//!
//!    $$W(w_i) = \frac{P(w_i|Z) \cdot P(Z|d)}{\lVert P(w_i|Z) \rVert \, \lVert P(Z|d) \rVert}$$
//!
//!    Graph words missing from the model get the smallest `W` among those
//!    present (pke's rule).
//! 3. Scores are one PageRank biased by the normalized `W`, `lambda = 0.85`
//!    (Eq. 4).
//! 4. Candidates are noun phrases `(ADJ)*(NOUN|PROPN)+`, scored by the sum of
//!    their words' scores. Untagged text uses stopword-delimited phrases of up
//!    to five words.
//!
//! # Deviations
//!
//! - Sterckx et al. use a directed graph (edges from a word to the words in
//!   the window after it); this uses pke's undirected SingleRank graph.
//! - Iteration runs to an L1 change of 1e-10. Ties are broken by first
//!   occurrence.

use crate::graph::{graph_words, rank_by_word_sum, WordGraph};
use crate::text::{delimited_phrases, noun_phrases, Document, Roles};
use std::collections::{HashMap, HashSet};

/// Single Topical PageRank keyphrase extractor.
#[derive(Debug, Clone)]
pub struct TopicalPageRank {
    stopwords: Roles,
    window: usize,
}

impl Default for TopicalPageRank {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicalPageRank {
    /// Window 10, English stopwords for untagged text.
    pub fn new() -> Self {
        Self {
            stopwords: Roles::default(),
            window: 10,
        }
    }

    /// Stopwords used for untagged text.
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

    /// Topic-biased score of each graph word. `word_topics` maps a stem to
    /// `P(w|Z)`; `doc_topics` is `P(Z|d)`. If no graph word is in the model
    /// the bias is uniform (plain SingleRank).
    pub fn word_scores(
        &self,
        doc: &Document,
        word_topics: &HashMap<String, Vec<f64>>,
        doc_topics: &[f64],
    ) -> HashMap<String, f64> {
        let mut g = WordGraph::default();
        g.add_document(
            doc,
            &graph_words(self.stopwords.language()),
            self.window,
            1.0,
        );
        let known: Vec<Option<f64>> = g
            .nodes
            .iter()
            .map(|w| word_topics.get(w).map(|v| cosine(v, doc_topics)))
            .collect();
        let floor = known.iter().flatten().copied().reduce(f64::min);
        let bias: Option<Vec<f64>> = floor.map(|f| known.iter().map(|k| k.unwrap_or(f)).collect());
        g.scores(bias.as_deref())
    }

    /// Up to `max` keyphrases, best first.
    pub fn rank(
        &self,
        doc: &Document,
        word_topics: &HashMap<String, Vec<f64>>,
        doc_topics: &[f64],
        max: usize,
    ) -> Vec<(String, f64)> {
        let candidates = if doc.is_tagged() {
            noun_phrases(doc, usize::MAX)
        } else {
            delimited_phrases(doc, &self.stopwords.delimiting(doc), 5)
        };
        let scores = self.word_scores(doc, word_topics, doc_topics);
        rank_by_word_sum(&candidates, &scores, max)
    }
}

/// Cosine similarity; 0 when either vector is zero or lengths differ.
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "alpha" and "beta" are isolated (a period between them and window 2),
    /// so each word's score is its normalized bias. alpha's vector matches
    /// the document's topics (cosine 1); beta's is orthogonal (cosine 0) and
    /// "gamma" is missing from the model, so it gets the minimum, 0.
    #[test]
    fn bias_is_cosine_to_document_topics() {
        let doc = Document::from_text("alpha. beta. gamma");
        let model: HashMap<String, Vec<f64>> = [
            ("alpha".to_string(), vec![0.2, 0.0]),
            ("beta".to_string(), vec![0.0, 0.3]),
        ]
        .into();
        let s = TopicalPageRank::new()
            .with_window(2)
            .word_scores(&doc, &model, &[0.9, 0.0]);
        assert!((s["alpha"] - 1.0).abs() < 1e-9, "{s:?}");
        assert!(s["beta"].abs() < 1e-9 && s["gamma"].abs() < 1e-9, "{s:?}");
    }

    #[test]
    fn empty_model_falls_back_to_uniform_bias() {
        let doc = Document::from_text("solar panel output and solar panel cost");
        let a = TopicalPageRank::new().rank(&doc, &HashMap::new(), &[], 3);
        let b = crate::singlerank::SingleRank::new().rank(&doc, 3);
        assert_eq!(
            a.iter().map(|x| &x.0).collect::<Vec<_>>(),
            b.iter().map(|x| &x.0).collect::<Vec<_>>()
        );
    }
}
