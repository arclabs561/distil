//! Luhn (1958): sentences with dense clusters of significant words.
//!
//! A word is significant when it is a content word whose stem occurs at
//! least twice in the document (sumy's rule; Luhn used a frequency band). A
//! cluster starts at a significant word and ends after `max_gap`
//! consecutive non-significant words; trailing non-significant words are
//! dropped (stem counts come from the shared term statistics of
//! [`crate::summarize`]). A cluster with `s` significant words over `n` words scores
//! `s^2 / n`, or 0 when `s = 1`, and a sentence scores its best cluster.
//!
//! H. P. Luhn, "The automatic creation of literature abstracts", IBM Journal
//! of Research and Development 2(2), 1958. Parity: sumy `LuhnSummarizer`.

use super::{by_score, content_stems, term_counts, SentenceExtractor};
use crate::text::{is_content, Document};
use std::collections::HashSet;

/// Luhn's significance-cluster score.
#[derive(Debug, Clone)]
pub struct Luhn {
    stopwords: HashSet<String>,
    max_gap: usize,
    min_count: usize,
}

impl Default for Luhn {
    fn default() -> Self {
        Self::new()
    }
}

impl Luhn {
    /// English stopwords, a gap of 4, stems occurring at least twice.
    pub fn new() -> Self {
        Luhn {
            stopwords: crate::default_stopwords(),
            max_gap: 4,
            min_count: 2,
        }
    }

    /// Replace the stopword list.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// End a cluster after this many consecutive non-significant words.
    pub fn with_max_gap(mut self, max_gap: usize) -> Self {
        self.max_gap = max_gap.max(1);
        self
    }

    /// A stem is significant when it occurs at least this often.
    pub fn with_min_count(mut self, min_count: usize) -> Self {
        self.min_count = min_count;
        self
    }
}

impl SentenceExtractor for Luhn {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        let freq = term_counts(&content_stems(doc, &self.stopwords));
        let scores = doc
            .sentences()
            .iter()
            .map(|s| {
                let flags: Vec<bool> = s
                    .iter()
                    .filter(|t| !t.is_punct())
                    .map(|t| {
                        is_content(t, &self.stopwords)
                            && freq
                                .get(t.stem.as_str())
                                .is_some_and(|&c| c >= self.min_count)
                    })
                    .collect();
                best_cluster(&flags, self.max_gap)
            })
            .collect();
        by_score(scores)
    }
}

/// Best `s^2 / n` over the clusters of a sentence's significance flags.
fn best_cluster(flags: &[bool], max_gap: usize) -> f64 {
    let mut best = 0.0f64;
    let mut i = 0;
    while i < flags.len() {
        if !flags[i] {
            i += 1;
            continue;
        }
        let (mut last, mut count, mut j) = (i, 0usize, i);
        while j < flags.len() && j - last <= max_gap {
            if flags[j] {
                last = j;
                count += 1;
            }
            j += 1;
        }
        if count > 1 {
            best = best.max((count * count) as f64 / (last - i + 1) as f64);
        }
        i = j.max(last + 1);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flags S S . . S: one cluster of 3 significant words over 5 words,
    /// 9 / 5. With four gaps (S . . . . S) the cluster closes and both
    /// halves have one significant word, so 0.
    #[test]
    fn cluster_score_follows_the_definition() {
        let f = |s: &str| s.chars().map(|c| c == 'S').collect::<Vec<_>>();
        assert_eq!(best_cluster(&f("SS..S"), 4), 9.0 / 5.0);
        assert_eq!(best_cluster(&f("S....S"), 4), 0.0);
        assert_eq!(best_cluster(&f("S...S"), 4), 4.0 / 5.0);
        assert_eq!(best_cluster(&f(".SS...."), 4), 2.0);
    }

    #[test]
    fn sentence_with_repeated_terms_wins() {
        let doc = Document::from_text(
            "Solar panels convert light. Solar panels need solar light. Cats sleep.",
        );
        let r = Luhn::new().rank(&doc);
        // S S . S scores 9/4; S S . S S scores 16/5.
        assert_eq!(r[0], (1, 16.0 / 5.0));
        assert_eq!(r[1], (0, 9.0 / 4.0));
        assert_eq!(r[2], (2, 0.0));
    }
}
