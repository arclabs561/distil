//! KL-Sum (Haghighi and Vanderwende 2009): greedily add the sentence that
//! brings the summary's word distribution closest to the document's.
//!
//! `P_D` is the distribution of content-word stems in the document and
//! `P_S` that of the summary with the candidate sentence added. Each step
//! adds the sentence minimizing
//!
//! `KL(P_D || P_S) = sum over w with P_S(w) > 0 of P_D(w) ln(P_D(w) / P_S(w))`
//!
//! Words missing from the summary are left out of the sum rather than
//! smoothed (sumy's convention); a candidate summary with no content words
//! scores infinity. Scores are `-KL` at selection, in selection order.
//! Uses the shared term distribution of [`crate::summarize`] and the greedy
//! loop of [`select`](super::select).
//!
//! A. Haghighi and L. Vanderwende, "Exploring content models for
//! multi-document summarization", NAACL-HLT 2009. Parity: sumy
//! `KLSummarizer`, which compares lowercased words rather than stems (give
//! it lowercased words as stems), counts the summary's stopwords and
//! original-case words, and filters stopwords before lowercasing; the
//! fixture generator patches the last two.

use super::select::greedy;
use super::{content_stems, distribution, Budget, SentenceExtractor};
use crate::text::{Document, Roles};
use std::collections::{BTreeMap, HashSet};

/// KL-Sum.
#[derive(Debug, Clone)]
pub struct KlSum {
    stopwords: Roles,
}

impl Default for KlSum {
    fn default() -> Self {
        Self::new()
    }
}

impl KlSum {
    /// English stopwords.
    pub fn new() -> Self {
        KlSum {
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

impl SentenceExtractor for KlSum {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        let terms = content_stems(doc, self.stopwords.language());
        let p_doc = distribution(&terms);
        greedy(
            terms.len(),
            &[],
            Budget::Sentences(terms.len()),
            |i, picked| {
                let mut joint: BTreeMap<&str, f64> = BTreeMap::new();
                for &w in picked.iter().chain([&i]).flat_map(|&j| terms[j].iter()) {
                    *joint.entry(w).or_insert(0.0) += 1.0;
                }
                let n: f64 = joint.values().sum();
                if n == 0.0 {
                    return f64::NEG_INFINITY;
                }
                -joint
                    .iter()
                    .map(|(w, c)| p_doc[w] * (p_doc[w] / (c / n)).ln())
                    .sum::<f64>()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Document words a a b c (P_D = 1/2, 1/4, 1/4) in sentences "a b",
    /// "a c". Either sentence alone gives P_S = 1/2, 1/2 on its two words:
    /// KL = 1/2 ln(1) + 1/4 ln(1/2) = -ln(2)/4 for both, so the first wins.
    /// With both, P_S = P_D and KL = 0.
    #[test]
    fn matches_hand_computed_divergence() {
        let doc = Document::from_text("Alpha beta. Alpha gamma.");
        let r = KlSum::new().rank(&doc);
        assert_eq!(r[0].0, 0);
        assert!((r[0].1 - 2f64.ln() / 4.0).abs() < 1e-12, "{r:?}");
        assert!(r[1].1.abs() < 1e-12, "{r:?}");
    }
}
