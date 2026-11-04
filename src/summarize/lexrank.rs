//! LexRank (Erkan and Radev 2004) and Biased LexRank (Otterbacher, Erkan
//! and Radev 2005): random walks over an idf-modified-cosine graph.
//!
//! Sentences are vectors of content-word stem counts times idf, and two
//! sentences are joined by their cosine (the idf-modified cosine; a
//! sentence's similarity to itself is 1). Idf defaults to `ln(N / n_w)`
//! with the document's `N` sentences as documents, or comes from the
//! caller. Both methods are configurations of the shared graph core in
//! [`graph`](super::graph):
//!
//! - [`LexRank`] binarizes the graph at a threshold (default 0.1, the
//!   paper's best) or keeps the cosines (continuous LexRank), and scores
//!   sentences by the stationary distribution of the walk that jumps
//!   uniformly with probability `jump` (default 0.15).
//! - [`BiasedLexRank`] uses the same graph and jumps toward sentences
//!   relevant to a query: `rel(s|q) = sum over w in q of
//!   ln(tf(w, s) + 1) ln(tf(w, q) + 1) idf(w)`, with jump probability
//!   `bias`. The defaults are the paper's best system, LR[0.20, 0.95]:
//!   threshold 0.2 and bias 0.95.
//!
//! The LexRank paper calls its jump probability `d` and suggests
//! [0.1, 0.2]; here it is `jump`, the probability of a uniform jump (the
//! complement of TextRank's damping).
//!
//! G. Erkan and D. R. Radev, "LexRank: Graph-based lexical centrality as
//! salience in text summarization", JAIR 22, 2004. J. Otterbacher, G. Erkan
//! and D. R. Radev, "Using random walks for question-focused sentence
//! retrieval", HLT/EMNLP 2005. Parity: sumy `LexRankSummarizer` and the
//! `lexrank` package, both with `jump = 0` (they do not jump), on the
//! idf each computes (sumy's `ln(N / (1 + n_w))` over sentences, the
//! package's from a document collection), passed through
//! [`LexRank::with_idf`]. sumy stops its power iteration at 0.1; the
//! fixture generator tightens it.

use super::graph::{similarity_matrix, stationary, threshold};
use super::{by_score, content_stems, cosine, counts, sentence_idf, weighted, SentenceExtractor};
use crate::text::{is_content, Document};
use std::collections::{HashMap, HashSet};

/// LexRank, thresholded or continuous.
#[derive(Debug, Clone)]
pub struct LexRank {
    stopwords: HashSet<String>,
    threshold: Option<f64>,
    jump: f64,
    idf: Option<(HashMap<String, f64>, f64)>,
}

impl Default for LexRank {
    fn default() -> Self {
        Self::new()
    }
}

impl LexRank {
    /// Threshold 0.1, jump 0.15, idf over the document's sentences,
    /// English stopwords.
    pub fn new() -> Self {
        LexRank {
            stopwords: crate::default_stopwords(),
            threshold: Some(0.1),
            jump: 0.15,
            idf: None,
        }
    }

    /// Replace the stopword list.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Keep edges whose cosine exceeds `t`, with weight 1.
    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = Some(t);
        self
    }

    /// Continuous LexRank: edges weigh their cosine.
    pub fn continuous(mut self) -> Self {
        self.threshold = None;
        self
    }

    /// Probability of a uniform jump at each step (0 for none).
    pub fn with_jump(mut self, jump: f64) -> Self {
        self.jump = jump.clamp(0.0, 1.0);
        self
    }

    /// Idf per stem, and the value for stems not in the map.
    pub fn with_idf(mut self, idf: HashMap<String, f64>, unseen: f64) -> Self {
        self.idf = Some((idf, unseen));
        self
    }

    /// The graph's edge weights (after the threshold, if any).
    pub fn similarity(&self, doc: &Document) -> Vec<Vec<f64>> {
        let terms = content_stems(doc, &self.stopwords);
        let own = sentence_idf(&terms);
        let idf = |w: &str| self.idf_of(w, &own);
        let vectors = weighted(&counts(&terms), &idf);
        let mut m = similarity_matrix(terms.len(), true, |i, j| {
            if i == j {
                1.0
            } else {
                cosine(&vectors[i], &vectors[j])
            }
        });
        if let Some(t) = self.threshold {
            threshold(&mut m, t);
        }
        m
    }

    /// Each sentence's LexRank score, in document order; they sum to 1.
    pub fn centrality(&self, doc: &Document) -> Vec<f64> {
        stationary(&self.similarity(doc), self.jump, None)
    }

    fn idf_of(&self, w: &str, own: &HashMap<&str, f64>) -> f64 {
        match &self.idf {
            Some((map, unseen)) => map.get(w).copied().unwrap_or(*unseen),
            None => own.get(w).copied().unwrap_or(0.0),
        }
    }
}

impl SentenceExtractor for LexRank {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        by_score(self.centrality(doc))
    }
}

/// LexRank biased toward a query.
#[derive(Debug, Clone)]
pub struct BiasedLexRank {
    lexrank: LexRank,
    query: Vec<String>,
    bias: f64,
}

impl BiasedLexRank {
    /// Threshold 0.2, bias 0.95, toward the query's content words.
    pub fn new(query: &Document) -> Self {
        BiasedLexRank::from_lexrank(LexRank::new().with_threshold(0.2), query)
    }

    /// Use `lexrank`'s graph (threshold, stopwords, idf); its jump is
    /// replaced by the bias.
    pub fn from_lexrank(lexrank: LexRank, query: &Document) -> Self {
        let query = query
            .tokens()
            .filter(|t| is_content(t, &lexrank.stopwords))
            .map(|t| t.stem.clone())
            .collect();
        BiasedLexRank {
            lexrank,
            query,
            bias: 0.95,
        }
    }

    /// Probability of jumping toward query-relevant sentences.
    pub fn with_bias(mut self, bias: f64) -> Self {
        self.bias = bias.clamp(0.0, 1.0);
        self
    }

    /// `rel(s|q)` of each sentence, in document order.
    pub fn relevance(&self, doc: &Document) -> Vec<f64> {
        let terms = content_stems(doc, &self.lexrank.stopwords);
        let own = sentence_idf(&terms);
        let q: Vec<&str> = self.query.iter().map(String::as_str).collect();
        let qtf = &counts(&[q])[0];
        counts(&terms)
            .iter()
            .map(|tf| {
                qtf.iter()
                    .map(|(w, n)| {
                        let t = tf.get(w).copied().unwrap_or(0.0);
                        (t + 1.0).ln() * (n + 1.0).ln() * self.lexrank.idf_of(w, &own)
                    })
                    .sum()
            })
            .collect()
    }

    /// Each sentence's score, in document order; they sum to 1. Without
    /// any relevant sentence the jump is uniform, as in [`LexRank`].
    pub fn centrality(&self, doc: &Document) -> Vec<f64> {
        let rel = self.relevance(doc);
        stationary(&self.lexrank.similarity(doc), self.bias, Some(&rel))
    }
}

impl SentenceExtractor for BiasedLexRank {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        by_score(self.centrality(doc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without jumps on a thresholded graph the walk's stationary
    /// distribution is proportional to degree (self-loops included). In
    /// "a b", "a c", "b c", "d e" the first three form a triangle (cosines
    /// 1/2 > 0.1), degree 3 each; the fourth is alone. Each component keeps
    /// its share of nodes: 3/4 split evenly, and 1/4.
    #[test]
    fn threshold_graph_without_jumps_is_proportional_to_degree() {
        let doc = Document::from_text("Alpha beta. Alpha gamma. Beta gamma. Delta epsilon.");
        let c = LexRank::new().with_jump(0.0).centrality(&doc);
        for x in &c[..3] {
            assert!((x - 0.25).abs() < 1e-9, "{c:?}");
        }
        assert!((c[3] - 0.25).abs() < 1e-9, "{c:?}");
        let sim = LexRank::new().similarity(&doc);
        assert_eq!(sim[0], vec![1.0, 1.0, 1.0, 0.0]);
    }

    /// Idf ln(4/2) for a, b, c and ln 4 for d, e. Sentences 0 and 1 share
    /// a: cos = idf_a^2 / (idf_a^2 + idf_b^2) = 1/2.
    #[test]
    fn continuous_weights_are_idf_cosines() {
        let doc = Document::from_text("Alpha beta. Alpha gamma. Beta gamma. Delta epsilon.");
        let sim = LexRank::new().continuous().similarity(&doc);
        assert!((sim[0][1] - 0.5).abs() < 1e-12 && sim[0][3] == 0.0);
    }

    /// A query word only in sentence 3 pulls the walk there.
    #[test]
    fn query_bias_moves_mass_to_relevant_sentences() {
        let doc = Document::from_text("Alpha beta. Alpha gamma. Beta gamma. Delta epsilon.");
        let b = BiasedLexRank::new(&Document::from_text("delta"));
        let rel = b.relevance(&doc);
        assert!((rel[3] - 2f64.ln() * 2f64.ln() * 4f64.ln()).abs() < 1e-12);
        assert_eq!(&rel[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(b.rank(&doc)[0].0, 3);
    }
}
