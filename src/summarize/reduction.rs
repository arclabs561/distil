//! Reduction (sumy): a sentence's score is the sum of its similarities to
//! every other sentence, the TextRank similarity without the random walk.
//!
//! The similarity of sentences with content-word lists `S_i`, `S_j` counts
//! matching pairs (a stem occurring twice in one and three times in the
//! other gives 6) over `ln |S_i| + ln |S_j|`, and is 0 when that is 0.
//! Uses the shared [`graph::similarity_matrix`](super::graph::similarity_matrix).
//!
//! From sumy's `ReductionSummarizer` (after github.com/adamfabish/Reduction);
//! no paper. Parity: sumy.

use super::graph::similarity_matrix;
use super::{by_score, content_stems, SentenceExtractor};
use crate::text::{Document, Roles};
use std::collections::HashSet;

/// sumy's Reduction.
#[derive(Debug, Clone)]
pub struct Reduction {
    stopwords: Roles,
}

impl Default for Reduction {
    fn default() -> Self {
        Self::new()
    }
}

impl Reduction {
    /// English stopwords.
    pub fn new() -> Self {
        Reduction {
            stopwords: Roles::default(),
        }
    }

    /// Replace the stopword list.
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
}

impl SentenceExtractor for Reduction {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        let terms = content_stems(doc, self.stopwords.language());
        let m = similarity_matrix(terms.len(), false, |i, j| {
            let pairs = terms[i]
                .iter()
                .map(|a| terms[j].iter().filter(|b| a == *b).count())
                .sum::<usize>() as f64;
            let norm = (terms[i].len() as f64).ln() + (terms[j].len() as f64).ln();
            if pairs == 0.0 || norm == 0.0 {
                0.0
            } else {
                pairs / norm
            }
        });
        by_score(m.iter().map(|row| row.iter().sum()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "a a b", "a c", "d": sentences 0 and 1 match 2 pairs over
    /// ln 3 + ln 2; sentence 2 matches nothing.
    #[test]
    fn counts_matching_pairs() {
        let doc = Document::from_text("Alpha alpha beta. Alpha gamma. Delta.");
        let r = Reduction::new().rank(&doc);
        let want = 2.0 / (3f64.ln() + 2f64.ln());
        assert!((r[0].1 - want).abs() < 1e-12 && (r[1].1 - want).abs() < 1e-12);
        assert_eq!(r[2], (2, 0.0));
    }
}
