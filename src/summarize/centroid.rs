//! Centroid-based summarization (Radev, Jing, Stys and Tam 2004, MEAD).
//!
//! The centroid gives each content-word stem the value `tf(w) idf(w)` over
//! the document, keeping values of at least `threshold` (the paper keeps
//! "words with high TF*IDF" above a predefined threshold whose value it does
//! not give; the default 0 keeps every word). Sentence `i` of `n`
//! (0-based) scores
//!
//! `SCORE(i) = w_c C_i + w_p P_i + w_f F_i`
//!
//! with `C_i` the sum of the centroid values of its distinct stems,
//! `P_i = (n - i) / n * max C`, and `F_i` the dot product of its stem counts
//! with the first sentence's. The weights default to 1 ("equal weight for
//! all three parameters"). Selection subtracts the redundancy penalty
//! `w_R R(i, j)` for the most similar sentence already chosen, where
//! `R = 2 |shared stems| / (|stems_i| + |stems_j|)` and `w_R` is the highest
//! score; the paper reranks until the extract stops changing, which this
//! greedy pass approximates. Uses the shared term statistics of
//! [`crate::summarize`] and the greedy loop of [`select`](super::select).
//!
//! D. R. Radev, H. Jing, M. Stys and D. Tam, "Centroid-based summarization
//! of multiple documents", Information Processing and Management 40(6),
//! 2004. No maintained reference implementation; tested on hand-computed
//! inputs.

use super::select::greedy;
use super::{content_stems, counts, dot, sentence_idf, word_count, Budget, SentenceExtractor};
use crate::text::Document;
use std::collections::{BTreeSet, HashMap, HashSet};

/// MEAD's centroid, position and first-sentence score with a redundancy
/// penalty.
#[derive(Debug, Clone)]
pub struct Centroid {
    stopwords: HashSet<String>,
    weights: [f64; 3],
    threshold: f64,
    idf: Option<(HashMap<String, f64>, f64)>,
}

impl Default for Centroid {
    fn default() -> Self {
        Self::new()
    }
}

impl Centroid {
    /// Weights 1, threshold 0, idf over the document's sentences, English
    /// stopwords.
    pub fn new() -> Self {
        Centroid {
            stopwords: crate::default_stopwords(),
            weights: [1.0; 3],
            threshold: 0.0,
            idf: None,
        }
    }

    /// Replace the stopword list.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Weights of the centroid, position and first-sentence scores.
    pub fn with_weights(mut self, centroid: f64, position: f64, first: f64) -> Self {
        self.weights = [centroid, position, first];
        self
    }

    /// Keep centroid words whose value is at least `t`.
    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    /// Idf per stem (for example from a background corpus), and the value
    /// for stems not in the map.
    pub fn with_idf(mut self, idf: HashMap<String, f64>, unseen: f64) -> Self {
        self.idf = Some((idf, unseen));
        self
    }

    /// `SCORE(i)` of each sentence, in document order, before the
    /// redundancy penalty.
    pub fn scores(&self, doc: &Document) -> Vec<f64> {
        let terms = content_stems(doc, &self.stopwords);
        let n = terms.len();
        let own = sentence_idf(&terms);
        let idf = |w: &str| match &self.idf {
            Some((map, unseen)) => map.get(w).copied().unwrap_or(*unseen),
            None => own.get(w).copied().unwrap_or(0.0),
        };
        let mut centroid: HashMap<&str, f64> = HashMap::new();
        for &w in terms.iter().flatten() {
            *centroid.entry(w).or_insert(0.0) += idf(w);
        }
        centroid.retain(|_, v| *v >= self.threshold);
        let c: Vec<f64> = terms
            .iter()
            .map(|s| {
                let distinct: BTreeSet<&str> = s.iter().copied().collect();
                distinct.iter().filter_map(|w| centroid.get(w)).sum()
            })
            .collect();
        let c_max = c.iter().copied().fold(0.0, f64::max);
        let tf = counts(&terms);
        let [wc, wp, wf] = self.weights;
        (0..n)
            .map(|i| {
                let p = (n - i) as f64 / n as f64 * c_max;
                wc * c[i] + wp * p + wf * dot(&tf[0], &tf[i])
            })
            .collect()
    }

    fn select(&self, doc: &Document, budget: Budget) -> Vec<(usize, f64)> {
        let score = self.scores(doc);
        let sets: Vec<HashSet<&str>> = content_stems(doc, &self.stopwords)
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect();
        let w_r = score.iter().copied().fold(0.0, f64::max);
        let lengths: Vec<usize> = doc.sentences().iter().map(|s| word_count(s)).collect();
        greedy(score.len(), &lengths, budget, |i, picked| {
            let r = picked
                .iter()
                .map(|&j| {
                    let total = sets[i].len() + sets[j].len();
                    if total == 0 {
                        0.0
                    } else {
                        2.0 * sets[i].intersection(&sets[j]).count() as f64 / total as f64
                    }
                })
                .fold(0.0, f64::max);
            score[i] - w_r * r
        })
    }
}

impl SentenceExtractor for Centroid {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        self.select(doc, Budget::Sentences(doc.sentences().len()))
    }

    fn summarize(&self, doc: &Document, budget: Budget) -> Vec<usize> {
        let mut out: Vec<usize> = self.select(doc, budget).iter().map(|x| x.0).collect();
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sentences "a b", "a c", "d". Idf: a ln(3/2), b c d ln 3. Centroid
    /// values: a 2 ln 1.5, b c d ln 3. C = (2 ln 1.5 + ln 3) for sentences 0
    /// and 1 and ln 3 for 2; P = (3, 2, 1)/3 * C_max; F = (2, 1, 0).
    #[test]
    fn scores_follow_the_definition() {
        let doc = Document::from_text("Alpha beta. Alpha gamma. Delta.");
        let s = Centroid::new().scores(&doc);
        let (a, b) = (2.0 * 1.5f64.ln(), 3f64.ln());
        let want = [
            (a + b) + (a + b) + 2.0,
            (a + b) + 2.0 / 3.0 * (a + b) + 1.0,
            b + 1.0 / 3.0 * (a + b),
        ];
        for (g, w) in s.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "{s:?}");
        }
    }

    /// A copy of the best sentence has R = 1 and loses the whole top score.
    #[test]
    fn penalizes_repeats() {
        let doc = Document::from_text("Alpha beta. Alpha beta. Gamma delta.");
        let r = Centroid::new().rank(&doc);
        assert_eq!(r[0].0, 0);
        assert_eq!(r[1].0, 2, "{r:?}");
    }
}
