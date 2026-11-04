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
use crate::text::{Document, Roles};
use crate::KeywordExtractor;
use std::collections::HashSet;

/// FirstPhrases keyphrase extractor.
#[derive(Debug, Clone)]
pub struct FirstPhrases {
    stopwords: Roles,
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
            stopwords: Roles::default(),
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

    /// Up to `max` keyphrases, earliest first, scored `-offset`.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        let candidates = noun_adjective_runs(doc, &self.stopwords.delimiting(doc));
        rank(
            candidates.iter().map(|c| (c, -(c.first_offset() as f64))),
            max,
        )
    }
}

impl KeywordExtractor for FirstPhrases {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.rank(&self.stopwords.document(text), max_keywords)
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
