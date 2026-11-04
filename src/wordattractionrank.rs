//! WordAttractionRank (Wang, Liu and McDonald, 2014).
//!
//! Wang, R., Liu, W. and McDonald, C. (2014). Corpus-independent generic
//! keyphrase extraction using word embedding vectors. Software Engineering
//! Research Conference, volume 39.
//!
//! # Method
//!
//! SingleRank's word graph (nouns and adjectives co-occurring within a
//! window of `w = 10` tokens), with each edge weighted by an attraction
//! score that combines caller-supplied word vectors with local counts:
//!
//! $$f(w_i, w_j) = \frac{freq(w_i) \cdot freq(w_j)}{d(w_i, w_j)^2}, \qquad
//! dice(w_i, w_j) = \frac{2 \, freq(w_i, w_j)}{freq(w_i) + freq(w_j)}$$
//!
//! $$attr(w_i, w_j) = f(w_i, w_j) \cdot dice(w_i, w_j)$$
//!
//! where `freq(w)` counts the word in the document, `freq(w_i, w_j)` counts
//! their co-occurrences, and `d` is the Euclidean distance between their
//! vectors. Word scores are weighted PageRank (`0.85`); a candidate (longest
//! noun/adjective sequence, untagged text: delimited phrase) scores the sum
//! of its word scores.
//!
//! # Deviations
//!
//! - The equations follow secondary descriptions of the paper (its PDF was
//!   not available to check); the window size is SingleRank's.
//! - Vectors are looked up by lowercased word; a stem takes the vector of
//!   its first word form that has one. When either word has no vector, or
//!   the vectors coincide, the distance is taken as 1.
//! - Ties are broken by first occurrence.

use crate::default_stopwords;
use crate::graph::{is_graph_word, rank_by_word_sum, WordGraph};
use crate::singlerank::noun_adjective_runs;
use crate::text::Document;
use std::collections::{HashMap, HashSet};

/// WordAttractionRank keyphrase extractor.
#[derive(Debug, Clone)]
pub struct WordAttractionRank {
    stopwords: HashSet<String>,
    window: usize,
}

impl Default for WordAttractionRank {
    fn default() -> Self {
        Self::new()
    }
}

impl WordAttractionRank {
    /// Window 10, English stopwords for untagged text.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            window: 10,
        }
    }

    /// Stopwords used for untagged text.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Co-occurrence window in tokens (at least 2).
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(2);
        self
    }

    /// Up to `max` keyphrases, best first. `vectors` maps lowercased words
    /// to embeddings of one dimension.
    pub fn rank(
        &self,
        doc: &Document,
        vectors: &HashMap<String, Vec<f64>>,
        max: usize,
    ) -> Vec<(String, f64)> {
        let accept = |t: &crate::text::Token| is_graph_word(t, &self.stopwords);
        let mut freq: HashMap<&str, f64> = HashMap::new();
        let mut vec_of: HashMap<&str, &[f64]> = HashMap::new();
        for t in doc.tokens().filter(|t| accept(t)) {
            *freq.entry(&t.stem).or_insert(0.0) += 1.0;
            if let Some(v) = vectors.get(&t.word.to_lowercase()) {
                vec_of.entry(&t.stem).or_insert(v);
            }
        }
        let mut g = WordGraph::default();
        g.add_document(doc, &accept, self.window, 1.0);
        let nodes = g.nodes.clone();
        g.reweight(|i, j, cooc| {
            let (a, b) = (nodes[i].as_str(), nodes[j].as_str());
            let (fa, fb) = (freq[a], freq[b]);
            let d2 = match (vec_of.get(a), vec_of.get(b)) {
                (Some(x), Some(y)) => x.iter().zip(*y).map(|(p, q)| (p - q).powi(2)).sum(),
                _ => 1.0,
            };
            attraction(fa, fb, cooc, if d2 > 0.0 { d2 } else { 1.0 })
        });
        let candidates = noun_adjective_runs(doc, &self.stopwords);
        rank_by_word_sum(&candidates, &g.scores(None), max)
    }
}

/// `f * dice` for word frequencies `fa`, `fb`, co-occurrence count `cooc`
/// and squared vector distance `d2`.
fn attraction(fa: f64, fb: f64, cooc: f64, d2: f64) -> f64 {
    let force = fa * fb / d2;
    let dice = 2.0 * cooc / (fa + fb);
    force * dice
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three single-word candidates linked "graph - model - tree" (window 3
    /// spans the commas). Unit frequencies make every dice 1, so
    /// attr(graph, model) = 1 / 1^2 and attr(model, tree) = 1 / 3^2. PageRank
    /// on that path: s_m = 0.135 / 0.2775, s_g = 0.05 + 0.85 * 0.9 * s_m and
    /// s_t = 0.05 + 0.85 * 0.1 * s_m. Without vectors the path is symmetric.
    /// freq 2 and 3, distance 2, 4 co-occurrences: f = 6 / 4, dice = 8 / 5.
    #[test]
    fn attraction_is_force_times_dice() {
        assert!((attraction(2.0, 3.0, 4.0, 4.0) - 1.5 * 1.6).abs() < 1e-12);
    }

    #[test]
    fn attraction_weights_follow_distances() {
        let doc = Document::from_text("graph, model, tree");
        let wa = WordAttractionRank::new().with_window(3);
        let flat: HashMap<String, f64> = wa.rank(&doc, &HashMap::new(), 10).into_iter().collect();
        assert!((flat["graph"] - flat["tree"]).abs() < 1e-12, "{flat:?}");
        let vectors = HashMap::from([
            ("graph".to_string(), vec![0.0, 0.0]),
            ("model".to_string(), vec![1.0, 0.0]),
            ("tree".to_string(), vec![4.0, 0.0]),
        ]);
        let got: HashMap<String, f64> = wa.rank(&doc, &vectors, 10).into_iter().collect();
        let m = 0.135 / 0.2775;
        assert!((got["model"] - m).abs() < 1e-9, "{got:?}");
        assert!((got["graph"] - (0.05 + 0.765 * m)).abs() < 1e-9, "{got:?}");
        assert!((got["tree"] - (0.05 + 0.085 * m)).abs() < 1e-9, "{got:?}");
    }
}
