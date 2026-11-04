//! Baselines: the first sentences (Lead-k) and a seeded random choice.
//!
//! Lead-k is a strong baseline on news, where writers put the main facts
//! first: on CNN/DailyMail, Lead-3 is hard for unsupervised methods to beat.

use super::{by_score, SentenceExtractor};
use crate::text::Document;

/// The first sentences, in order. Sentence `i` of `n` scores `n - i`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lead;

impl Lead {
    /// Lead-k; the `k` comes from the [`Budget`](super::Budget).
    pub fn new() -> Self {
        Lead
    }
}

impl SentenceExtractor for Lead {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        let n = doc.sentences().len();
        (0..n).map(|i| (i, (n - i) as f64)).collect()
    }
}

/// Sentences in a random order fixed by a seed: each sentence gets a score
/// uniform in [0, 1) from a SplitMix64 stream.
#[derive(Debug, Clone, Copy)]
pub struct Random {
    seed: u64,
}

impl Default for Random {
    fn default() -> Self {
        Self::new()
    }
}

impl Random {
    /// Seed 0.
    pub fn new() -> Self {
        Random { seed: 0 }
    }

    /// Use another seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl SentenceExtractor for Random {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        let mut state = self.seed;
        let scores = (0..doc.sentences().len())
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                (z >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect();
        by_score(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::summarize::Budget;

    const TEXT: &str = "One. Two. Three. Four. Five. Six.";

    #[test]
    fn lead_keeps_the_first_sentences() {
        let doc = Document::from_text(TEXT);
        assert_eq!(
            Lead::new().summarize(&doc, Budget::Sentences(3)),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn random_is_fixed_by_its_seed() {
        let doc = Document::from_text(TEXT);
        let a = Random::new().with_seed(7).rank(&doc);
        assert_eq!(a, Random::new().with_seed(7).rank(&doc));
        let mut idx: Vec<usize> = a.iter().map(|x| x.0).collect();
        idx.sort_unstable();
        assert_eq!(idx, (0..6).collect::<Vec<_>>());
        let orders: std::collections::HashSet<Vec<usize>> = (0..20)
            .map(|s| {
                Random::new()
                    .with_seed(s)
                    .rank(&doc)
                    .iter()
                    .map(|x| x.0)
                    .collect()
            })
            .collect();
        assert!(orders.len() > 10, "seeds should give different orders");
    }
}
