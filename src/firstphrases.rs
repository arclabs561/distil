//! FirstPhrases: the position baseline.
//!
//! The first candidates of a document are its keyphrases. There is no paper;
//! this is pke's `FirstPhrases` baseline, which this module matches exactly
//! (`tests/pke_parity.rs`).
//!
//! # Method
//!
//! 1. Candidates are the longest sequences of nouns, proper nouns and
//!    adjectives (untagged text: stopword- and punctuation-delimited phrases
//!    of up to five words).
//! 2. A candidate scores minus the token offset of its first occurrence, so
//!    the earliest candidate ranks first.

use crate::graph::rank;
use crate::singlerank::noun_adjective_runs;
use crate::text::Document;
use crate::{default_stopwords, KeywordExtractor};
use std::collections::HashSet;

/// FirstPhrases keyphrase extractor.
#[derive(Debug, Clone)]
pub struct FirstPhrases {
    stopwords: HashSet<String>,
}

impl Default for FirstPhrases {
    fn default() -> Self {
        Self::new()
    }
}

impl FirstPhrases {
    /// English stopwords for untagged text.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
        }
    }

    /// Stopwords used for untagged text.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Up to `max` keyphrases, earliest first, scored `-offset`.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        let candidates = noun_adjective_runs(doc, &self.stopwords);
        rank(
            candidates.iter().map(|c| (c, -(c.first_offset() as f64))),
            max,
        )
    }
}

impl KeywordExtractor for FirstPhrases {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.rank(&Document::from_text(text), max_keywords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earliest_candidate_first() {
        let got = FirstPhrases::new().extract("The red fox jumps over the lazy dog.", 10);
        assert_eq!(
            got,
            vec![
                ("red fox jumps".to_string(), -1.0),
                ("lazy dog".to_string(), -6.0)
            ]
        );
    }
}
