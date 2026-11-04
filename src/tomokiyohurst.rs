//! Phraseness and informativeness (Tomokiyo and Hurst, 2003).
//!
//! Tomokiyo, T. and Hurst, M. (2003). A Language Model Approach to
//! Keyphrase Extraction. ACL 2003 workshop paper, ACL Anthology W03-1805.
//!
//! # Method
//!
//! A phrase `w` is scored by pointwise KL divergence between language
//! models, `delta_w(p || q) = p(w) ln(p(w) / q(w))`:
//!
//! - phraseness, the loss from assuming its words independent:
//!   `delta_w(LM_fg^N || LM_fg^1)`;
//! - informativeness, the loss from assuming it comes from the background
//!   corpus: `delta_w(LM_fg^1 || LM_bg^1)` (the paper's equation 10, the
//!   variant used in its experiments);
//! - score = phraseness + informativeness (equation 12).
//!
//! `LM^N` is the unsmoothed n-gram model of the phrase's length (count over
//! the number of candidate n-grams of that length), `LM^1` the unigram
//! model (product of word probabilities). The foreground is the document;
//! the background is caller-supplied word counts. Candidates are n-grams of
//! up to three words inside stopword- and punctuation-delimited runs.
//!
//! # Deviations
//!
//! - The paper smooths the background unigram model with Katz smoothing;
//!   this uses add-one smoothing over the union of both vocabularies.
//! - Words are compared by stem (callers wanting surface forms can build a
//!   [`Document`] whose stems are the lowercased words); the paper does not
//!   stem.
//! - Ties are broken by first occurrence.

use crate::default_stopwords;
use crate::graph::rank;
use crate::text::{ngrams, Document};
use std::collections::{HashMap, HashSet};

/// Background word counts, keyed by stem.
#[derive(Debug, Clone, Default)]
pub struct Background {
    counts: HashMap<String, f64>,
    total: f64,
}

impl Background {
    /// From counts computed elsewhere (keys are stems).
    pub fn new(counts: HashMap<String, usize>) -> Self {
        let total = counts.values().sum::<usize>() as f64;
        let counts = counts.into_iter().map(|(k, v)| (k, v as f64)).collect();
        Self { counts, total }
    }

    /// Count the non-punctuation tokens of a corpus.
    pub fn from_documents<'a>(docs: impl IntoIterator<Item = &'a Document>) -> Self {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for t in docs.into_iter().flat_map(Document::tokens) {
            if !t.is_punct() {
                *counts.entry(t.stem.clone()).or_insert(0) += 1;
            }
        }
        Self::new(counts)
    }
}

/// Tomokiyo-Hurst keyphrase scorer.
#[derive(Debug, Clone)]
pub struct TomokiyoHurst {
    stopwords: HashSet<String>,
    max_n: usize,
}

impl Default for TomokiyoHurst {
    fn default() -> Self {
        Self::new()
    }
}

impl TomokiyoHurst {
    /// Up to trigrams, English stopwords.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            max_n: 3,
        }
    }

    /// Stopwords delimiting candidates.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Longest candidate in words (at least 1).
    pub fn with_max_words(mut self, n: usize) -> Self {
        self.max_n = n.max(1);
        self
    }

    /// Up to `max` keyphrases with phraseness + informativeness, best first.
    pub fn rank(&self, doc: &Document, background: &Background, max: usize) -> Vec<(String, f64)> {
        let mut fg: HashMap<&str, f64> = HashMap::new();
        for t in doc.tokens().filter(|t| !t.is_punct()) {
            *fg.entry(&t.stem).or_insert(0.0) += 1.0;
        }
        let fg_total: f64 = fg.values().sum();
        let vocab = background.counts.len() as f64
            + fg.keys()
                .filter(|w| !background.counts.contains_key(**w))
                .count() as f64;
        let p_fg = |w: &str| fg.get(w).copied().unwrap_or(0.0) / fg_total;
        let p_bg = |w: &str| {
            (background.counts.get(w).copied().unwrap_or(0.0) + 1.0) / (background.total + vocab)
        };
        let candidates = ngrams(doc, &self.stopwords, self.max_n);
        let mut per_length: HashMap<usize, f64> = HashMap::new();
        for c in &candidates {
            *per_length.entry(c.len()).or_insert(0.0) += c.offsets.len() as f64;
        }
        let scored = candidates.iter().map(|c| {
            let p_n = c.offsets.len() as f64 / per_length[&c.len()];
            let p_1: f64 = c.lexical_form.iter().map(|w| p_fg(w)).product();
            let q_bg: f64 = c.lexical_form.iter().map(|w| p_bg(w)).product();
            let phraseness = p_n * (p_n / p_1).ln();
            let informativeness = p_1 * (p_1 / q_bg).ln();
            (c, phraseness + informativeness)
        });
        rank(scored, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "graph model graph model tree": foreground unigrams graph 2/5,
    /// model 2/5, tree 1/5. Background {graph: 1, other: 1}, vocabulary
    /// {graph, other, model, tree} = 4, so p_bg(graph) = 2/6 and
    /// p_bg(model) = 1/6. Bigrams: "graph model" x2, "model graph" x1,
    /// "model tree" x1, so p_2(graph model) = 2/4.
    #[test]
    fn scores_follow_pointwise_kl() {
        let doc = Document::from_text("graph model graph model tree");
        let bg = Background::new(HashMap::from([
            ("graph".to_string(), 1),
            ("other".to_string(), 1),
        ]));
        let got: HashMap<String, f64> = TomokiyoHurst::new()
            .rank(&doc, &bg, 100)
            .into_iter()
            .collect();
        let (pg, pm) = (0.4f64, 0.4f64);
        let graph = pg * (pg / (2.0 / 6.0)).ln();
        assert!((got["graph"] - graph).abs() < 1e-12, "{got:?}");
        let p1 = pg * pm;
        let phr = 0.5 * (0.5 / p1).ln();
        let inf = p1 * (p1 / ((2.0 / 6.0) * (1.0 / 6.0))).ln();
        assert!((got["graph model"] - (phr + inf)).abs() < 1e-12, "{got:?}");
    }

    #[test]
    fn background_frequency_lowers_informativeness() {
        let doc = Document::from_text("protein folding uses protein energy");
        let common = Background::new(HashMap::from([("protein".to_string(), 1000)]));
        let rare = Background::new(HashMap::from([("energi".to_string(), 1000)]));
        let s = |bg: &Background| -> f64 {
            TomokiyoHurst::new()
                .rank(&doc, bg, 100)
                .into_iter()
                .find(|(p, _)| p == "protein")
                .unwrap()
                .1
        };
        assert!(s(&common) < s(&rare));
    }
}
