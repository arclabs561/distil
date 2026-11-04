//! KP-Miner (El-Beltagy and Rafea, 2010).
//!
//! El-Beltagy, S. R. and Rafea, A. (2010). KP-Miner: Participation in
//! SemEval-2. Proceedings of the 5th International Workshop on Semantic
//! Evaluation, pages 190-193. ACL Anthology S10-1041.
//!
//! # Method
//!
//! 1. Candidates are n-grams of up to five tokens that pass pke's filter
//!    (as in [`crate::tfidf`]), first seen within the first `cutoff = 400`
//!    tokens, and seen at least `lasf = 3` times (least allowable seen
//!    frequency).
//! 2. With `D` corpus documents, single words get
//!    `idf = log2((1 + D) / (1 + df))`; a compound phrase is assumed to occur
//!    in one document only, `idf = log2(1 + D)`.
//! 3. A boosting factor favours compound phrases:
//!
//!    $$B = \min\left(\sigma, \frac{|N_d|}{|P_d| \cdot \alpha}\right)$$
//!
//!    with `|N_d|` the candidate occurrences in the document, `|P_d|` those
//!    of compound candidates (at least 1), `alpha = 2.3`, `sigma = 3`.
//! 4. `w = tf * idf` for single words and `w = tf * B * idf` for compound
//!    phrases (the paper's position factor `P_f` is 1).
//!
//! # Deviations
//!
//! - Steps 1-4 follow pke's `KPMiner`, which this module matches exactly
//!   (`tests/pke_parity.rs`): `|N_d|` and `|P_d|` count occurrences, the
//!   cutoff counts tokens rather than words, and idf has pke's `+ 1`s.
//! - The paper's final refinement (subtract a top phrase's count from each
//!   top phrase it contains, then re-weight) is off by default, as in pke;
//!   [`KpMiner::with_refinement`] turns it on.
//! - Ties are broken by first occurrence.

use crate::default_stopwords;
use crate::graph::rank;
use crate::text::{Candidate, Document};
use crate::tfidf::{ngram_candidates, DocumentFrequency};
use std::collections::HashSet;

/// KP-Miner keyphrase extractor.
#[derive(Debug, Clone)]
pub struct KpMiner {
    stopwords: HashSet<String>,
    lasf: usize,
    cutoff: usize,
    alpha: f64,
    sigma: f64,
    refine_top: Option<usize>,
}

impl Default for KpMiner {
    fn default() -> Self {
        Self::new()
    }
}

impl KpMiner {
    /// `lasf = 3`, `cutoff = 400`, `alpha = 2.3`, `sigma = 3`, English
    /// stopwords, no refinement.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            lasf: 3,
            cutoff: 400,
            alpha: 2.3,
            sigma: 3.0,
            refine_top: None,
        }
    }

    /// Stopwords for the candidate filter.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Least allowable seen frequency.
    pub fn with_lasf(mut self, lasf: usize) -> Self {
        self.lasf = lasf;
        self
    }

    /// Candidates first seen after this token offset are dropped.
    pub fn with_cutoff(mut self, cutoff: usize) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Boosting constants `alpha` and `sigma`.
    pub fn with_boost(mut self, alpha: f64, sigma: f64) -> Self {
        self.alpha = alpha;
        self.sigma = sigma;
        self
    }

    /// Apply the paper's refinement to the top `n` phrases.
    pub fn with_refinement(mut self, n: usize) -> Self {
        self.refine_top = Some(n);
        self
    }

    /// Up to `max` keyphrases, best first.
    pub fn rank(&self, doc: &Document, df: &DocumentFrequency, max: usize) -> Vec<(String, f64)> {
        let mut candidates = ngram_candidates(doc, &self.stopwords, 5);
        candidates.retain(|c| c.first_offset() <= self.cutoff && c.offsets.len() >= self.lasf);
        let n_d: usize = candidates.iter().map(|c| c.offsets.len()).sum();
        let p_d: usize = candidates
            .iter()
            .filter(|c| c.len() > 1)
            .map(|c| c.offsets.len())
            .sum();
        let boost = (n_d as f64 / (p_d.max(1) as f64 * self.alpha)).min(self.sigma);
        let corpus = DocumentFrequency::new(df.documents(), Default::default());
        let weight = |c: &Candidate, tf: f64| {
            if c.len() == 1 {
                tf * df.idf(c)
            } else {
                tf * boost * corpus.idf(c)
            }
        };
        let mut scored: Vec<(&Candidate, f64)> = candidates
            .iter()
            .map(|c| (c, weight(c, c.offsets.len() as f64)))
            .collect();
        if let Some(n) = self.refine_top {
            let top = rank_refs(&scored, n);
            for (c, w) in scored.iter_mut() {
                let c: &Candidate = c;
                if !top.iter().any(|t| std::ptr::eq(*t, c)) {
                    continue;
                }
                let subtract: usize = top
                    .iter()
                    .filter(|t| t.len() > c.len() && contains(&t.lexical_form, &c.lexical_form))
                    .map(|t| t.offsets.len())
                    .sum();
                *w = weight(c, c.offsets.len().saturating_sub(subtract) as f64);
            }
        }
        rank(scored.into_iter(), max)
    }
}

/// The `n` best candidates by score (ties to the earlier first occurrence).
fn rank_refs<'a>(scored: &[(&'a Candidate, f64)], n: usize) -> Vec<&'a Candidate> {
    let mut v = scored.to_vec();
    v.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| a.0.first_offset().cmp(&b.0.first_offset()))
    });
    v.into_iter().take(n).map(|(c, _)| c).collect()
}

/// True when `inner` is a contiguous run of `outer`.
fn contains(outer: &[String], inner: &[String]) -> bool {
    outer.windows(inner.len()).any(|w| w == inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn scores(k: &KpMiner, text: &str, df: &DocumentFrequency) -> HashMap<String, f64> {
        k.rank(&Document::from_text(text), df, usize::MAX)
            .into_iter()
            .collect()
    }

    /// "neural network" x2 and its words x2 each: |N_d| = 6, |P_d| = 2, so
    /// B = min(3, 6 / (2 * 2.3)). With D = 3 and df(network) = 1:
    /// w(network) = 2 log2(4 / 2), w(neural) = 2 log2(4), and the phrase
    /// w = 2 * B * log2(4).
    #[test]
    fn weights_follow_the_boosted_tf_idf() {
        let df = DocumentFrequency::new(3, HashMap::from([("network".to_string(), 1)]));
        let k = KpMiner::new().with_lasf(1);
        let got = scores(&k, "neural network. neural network.", &df);
        let b = 6.0 / (2.0 * 2.3);
        assert!((got["network"] - 2.0).abs() < 1e-12, "{got:?}");
        assert!((got["neural"] - 4.0).abs() < 1e-12, "{got:?}");
        assert!(
            (got["neural network"] - 2.0 * b * 2.0).abs() < 1e-12,
            "{got:?}"
        );
    }

    #[test]
    fn lasf_and_cutoff_filter_candidates() {
        let df = DocumentFrequency::new(1, HashMap::new());
        let text = "graph graph graph. tree. graph tree";
        let got = scores(&KpMiner::new(), text, &df);
        assert_eq!(got.keys().collect::<Vec<_>>(), vec!["graph"]);
        // "tree" is first seen at token 4; a cutoff of 3 drops it.
        let k = KpMiner::new().with_lasf(1).with_cutoff(3);
        let got = scores(&k, text, &df);
        assert!(
            !got.contains_key("tree") && got.contains_key("graph"),
            "{got:?}"
        );
    }

    /// Refinement subtracts "neural network"'s count (2) from "network"
    /// (seen 3 times) when both are in the top n.
    #[test]
    fn refinement_discounts_contained_phrases() {
        let df = DocumentFrequency::new(3, HashMap::new());
        let text = "neural network. neural network. network.";
        let k = KpMiner::new().with_lasf(1);
        assert!((scores(&k, text, &df)["network"] - 3.0 * 2.0).abs() < 1e-12);
        let got = scores(&k.with_refinement(10), text, &df);
        assert!((got["network"] - 2.0).abs() < 1e-12, "{got:?}");
    }
}
