//! TF-IDF over n-gram candidates with caller-supplied document frequencies.
//!
//! Salton, G. and Buckley, C. (1988). Term-weighting approaches in automatic
//! text retrieval. Information Processing and Management 24(5), 513-523.
//! The candidate selection and constants follow pke's `TfIdf`.
//!
//! # Method
//!
//! 1. Candidates are every n-gram of up to three tokens
//!    ([`text::all_ngrams`]) that passes pke's filter
//!    ([`text::filter_candidates`]): no stopword, no punctuation-only word,
//!    at least three characters, every word at least two characters.
//! 2. With `D` documents in the corpus and `df(c)` of them containing the
//!    candidate's lexical form (its stems joined by spaces),
//!
//!    $$w(c) = tf(c) \cdot \log_2 \frac{1 + D}{1 + df(c)}$$
//!
//!    where `tf(c)` counts the candidate's occurrences in the document.
//!
//! # Deviations
//!
//! - The `+ 1`s are pke's: they keep the weight finite for an unseen
//!   candidate and count the document itself as part of the corpus.
//! - Ties are broken by first occurrence.
//!

use crate::default_stopwords;
use crate::graph::rank;
use crate::text::{self, Candidate, Document};
use std::collections::{HashMap, HashSet};

/// Document frequencies of lexical forms (stems joined by single spaces)
/// over a corpus of `documents` documents.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DocumentFrequency {
    documents: usize,
    counts: HashMap<String, usize>,
}

impl DocumentFrequency {
    /// From counts computed elsewhere, for example a file pke wrote with
    /// `compute_document_frequency`.
    pub fn new(documents: usize, counts: HashMap<String, usize>) -> Self {
        Self { documents, counts }
    }

    /// Count, for each lexical form, the documents with an n-gram candidate
    /// of up to `max_n` tokens of that form (selected as in [`TfIdf`]).
    pub fn from_documents<'a>(
        docs: impl IntoIterator<Item = &'a Document>,
        stopwords: &HashSet<String>,
        max_n: usize,
    ) -> Self {
        let mut df = Self::default();
        for doc in docs {
            df.documents += 1;
            for c in ngram_candidates(doc, stopwords, max_n) {
                *df.counts.entry(c.lexical_form.join(" ")).or_insert(0) += 1;
            }
        }
        df
    }

    /// Number of documents in the corpus.
    pub fn documents(&self) -> usize {
        self.documents
    }

    /// Number of documents containing `lexical_form`.
    pub fn get(&self, lexical_form: &str) -> usize {
        self.counts.get(lexical_form).copied().unwrap_or(0)
    }

    /// `log2((1 + D) / (1 + df))` for a candidate.
    pub(crate) fn idf(&self, candidate: &Candidate) -> f64 {
        let n = 1.0 + self.documents as f64;
        (n / (1.0 + self.get(&candidate.lexical_form.join(" ")) as f64)).log2()
    }
}

/// pke's n-gram candidates: all n-grams, then its default filter (which
/// also drops candidates of more than five words).
pub(crate) fn ngram_candidates(
    doc: &Document,
    stopwords: &HashSet<String>,
    max_n: usize,
) -> Vec<Candidate> {
    text::filter_candidates(text::all_ngrams(doc, max_n), stopwords, 5)
}

/// TF-IDF keyphrase extractor.
#[derive(Debug, Clone)]
pub struct TfIdf {
    stopwords: HashSet<String>,
    max_n: usize,
}

impl Default for TfIdf {
    fn default() -> Self {
        Self::new()
    }
}

impl TfIdf {
    /// N-grams of up to three words, English stopwords.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            max_n: 3,
        }
    }

    /// Stopwords for the candidate filter.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Longest candidate in words (at least 1).
    pub fn with_max_words(mut self, n: usize) -> Self {
        self.max_n = n.max(1);
        self
    }

    /// Up to `max` keyphrases, best first.
    pub fn rank(&self, doc: &Document, df: &DocumentFrequency, max: usize) -> Vec<(String, f64)> {
        let candidates = ngram_candidates(doc, &self.stopwords, self.max_n);
        let scored = candidates
            .iter()
            .map(|c| (c, c.offsets.len() as f64 * df.idf(c)));
        rank(scored, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "graph" occurs twice in a two-document corpus where it appears in one
    /// other document: w = 2 * log2(3 / 2). "ranking" is unseen elsewhere:
    /// w = 1 * log2(3 / 1).
    #[test]
    fn weight_is_tf_times_smoothed_log2_idf() {
        let doc = Document::from_text("graph ranking. graph");
        let df = DocumentFrequency::new(2, HashMap::from([("graph".to_string(), 1)]));
        let got: HashMap<String, f64> = TfIdf::new().rank(&doc, &df, 10).into_iter().collect();
        assert!(
            (got["graph"] - 2.0 * 1.5f64.log2()).abs() < 1e-12,
            "{got:?}"
        );
        assert!((got["ranking"] - 3.0f64.log2()).abs() < 1e-12, "{got:?}");
        assert!(
            (got["graph ranking"] - 3.0f64.log2()).abs() < 1e-12,
            "{got:?}"
        );
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn document_frequency_counts_documents_not_occurrences() {
        let docs = [
            Document::from_text("graph graph graph"),
            Document::from_text("graph models"),
        ];
        let df = DocumentFrequency::from_documents(&docs, &default_stopwords(), 3);
        assert_eq!(df.documents(), 2);
        assert_eq!(df.get("graph"), 2);
        assert_eq!(df.get("graph model"), 1);
        assert_eq!(df.get("graph graph"), 1);
    }
}
