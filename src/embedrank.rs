//! EmbedRank and EmbedRank++ (Bennani-Smires et al., 2018).
//!
//! Bennani-Smires, K., Musat, C., Hossmann, A., Baeriswyl, M. and Jaggi, M.
//! (2018). Simple Unsupervised Keyphrase Extraction using Sentence
//! Embeddings. Proceedings of CoNLL 2018, pages 221-229. ACL Anthology
//! K18-1022.
//!
//! # Method
//!
//! Candidates are noun phrases `(ADJ)*(NOUN|PROPN)+` (untagged text:
//! stopword- and punctuation-delimited phrases of up to five words). The
//! caller embeds each candidate ([`EmbedRank::candidates`] lists them) and
//! the document in one vector space (the paper uses Sent2Vec or Doc2Vec).
//!
//! - EmbedRank scores a candidate by `cos(C_i, doc)`.
//! - EmbedRank++ selects greedily by maximal marginal relevance,
//!
//!   $$MMR = \arg\max_{C_i \in C \setminus K} \left[\lambda \, \widetilde{cos}(C_i, doc) - (1 - \lambda) \max_{C_j \in K} \widetilde{cos}(C_i, C_j)\right]$$
//!
//!   with `K` the phrases already selected and the normalized similarity
//!
//!   $$\widetilde{cos}(C_i, doc) = 0.5 + \frac{n(C_i) - \overline{n}}{\sigma(n)}, \qquad n(C_i) = \frac{cos(C_i, doc) - \min_j cos(C_j, doc)}{\max_j cos(C_j, doc)}$$
//!
//!   and the same transformation of each column of the candidate-candidate
//!   similarity matrix (diagonal excluded). The default `lambda = 0.5` gives
//!   relevance and diversity equal weight.
//!
//! # Deviations
//!
//! - Candidates without a vector are skipped. A zero standard deviation
//!   makes every normalized similarity 0.5.
//! - EmbedRank++ reports each phrase's MMR value at the time it was
//!   selected, so scores need not decrease.
//! - Ties are broken by first occurrence.

use crate::default_stopwords;
use crate::graph::rank;
use crate::text::{delimited_phrases, noun_phrases, Candidate, Document};
use std::collections::{HashMap, HashSet};

/// EmbedRank keyphrase extractor; EmbedRank++ with [`EmbedRank::with_diversity`].
#[derive(Debug, Clone)]
pub struct EmbedRank {
    stopwords: HashSet<String>,
    lambda: Option<f64>,
}

impl Default for EmbedRank {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbedRank {
    /// EmbedRank (no diversity), English stopwords for untagged text.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            lambda: None,
        }
    }

    /// Stopwords used for untagged text.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// EmbedRank++: select by MMR with relevance weight `lambda` in `[0, 1]`.
    pub fn with_diversity(mut self, lambda: f64) -> Self {
        self.lambda = Some(lambda.clamp(0.0, 1.0));
        self
    }

    /// The candidates to embed, in order of first occurrence.
    pub fn candidates(&self, doc: &Document) -> Vec<Candidate> {
        if doc.is_tagged() {
            noun_phrases(doc, usize::MAX)
        } else {
            delimited_phrases(doc, &self.stopwords, 5)
        }
    }

    /// Up to `max` keyphrases, best first. `phrase_vectors` maps candidate
    /// surfaces (as [`Candidate::surface`]) to embeddings in the space of
    /// `doc_vector`.
    pub fn rank(
        &self,
        doc: &Document,
        phrase_vectors: &HashMap<String, Vec<f64>>,
        doc_vector: &[f64],
        max: usize,
    ) -> Vec<(String, f64)> {
        let all = self.candidates(doc);
        let (cands, vecs): (Vec<&Candidate>, Vec<&[f64]>) = all
            .iter()
            .filter_map(|c| phrase_vectors.get(&c.surface).map(|v| (c, v.as_slice())))
            .unzip();
        let to_doc: Vec<f64> = vecs.iter().map(|v| cosine(v, doc_vector)).collect();
        let Some(lambda) = self.lambda else {
            return rank(cands.into_iter().zip(to_doc), max);
        };
        let n = cands.len();
        let rel = normalize(&to_doc);
        let pair: Vec<Vec<f64>> = (0..n)
            .map(|j| {
                let col: Vec<f64> = (0..n)
                    .filter(|&i| i != j)
                    .map(|i| cosine(vecs[i], vecs[j]))
                    .collect();
                let mut col = normalize(&col);
                col.insert(j, f64::NAN);
                col
            })
            .collect();
        let mut chosen: Vec<usize> = Vec::new();
        let mut out = Vec::new();
        while chosen.len() < n.min(max) {
            let mmr = |i: usize| {
                let redundancy = chosen
                    .iter()
                    .map(|&j| pair[j][i])
                    .fold(f64::NEG_INFINITY, f64::max);
                let redundancy = if chosen.is_empty() { 0.0 } else { redundancy };
                lambda * rel[i] - (1.0 - lambda) * redundancy
            };
            let best = (0..n)
                .filter(|i| !chosen.contains(i))
                .map(|i| (i, mmr(i)))
                .reduce(|a, b| if b.1 > a.1 { b } else { a })
                .expect("a candidate remains");
            chosen.push(best.0);
            out.push((cands[best.0].surface.clone(), best.1));
        }
        out
    }
}

/// `0.5 + (n - mean(n)) / std(n)` with `n = (x - min) / max`.
fn normalize(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let n: Vec<f64> = xs.iter().map(|x| (x - min) / max).collect();
    let mean = n.iter().sum::<f64>() / n.len() as f64;
    let std = (n.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n.len() as f64).sqrt();
    n.iter()
        .map(|x| {
            if std > 0.0 {
                0.5 + (x - mean) / std
            } else {
                0.5
            }
        })
        .collect()
}

pub(crate) fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Document, HashMap<String, Vec<f64>>) {
        let doc = Document::from_text("deep nets, neural nets, graph kernels");
        let v = HashMap::from([
            ("deep nets".to_string(), vec![1.0, 0.0]),
            ("neural nets".to_string(), vec![0.99, 0.141]),
            ("graph kernels".to_string(), vec![0.6, 0.8]),
        ]);
        (doc, v)
    }

    #[test]
    fn embedrank_orders_by_cosine_to_the_document() {
        let (doc, v) = setup();
        let got = EmbedRank::new().rank(&doc, &v, &[1.0, 0.0], 10);
        let names: Vec<&str> = got.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(names, vec!["deep nets", "neural nets", "graph kernels"]);
        assert!((got[2].1 - 0.6).abs() < 1e-12);
    }

    /// Column "deep nets" of the candidate similarities is [0.99, 0.6] for
    /// ("neural nets", "graph kernels"), normalized to [1.5, -0.5]. With
    /// lambda = 0.25, after "deep nets" the near-copy "neural nets" scores
    /// 0.25 rel - 0.75 * 1.5 and "graph kernels" 0.25 rel + 0.75 * 0.5, so the
    /// distant phrase is taken second. lambda = 1 is plain EmbedRank.
    #[test]
    fn embedrank_plus_plus_prefers_diverse_phrases() {
        let (doc, v) = setup();
        let d = [1.0, 0.0];
        let got = EmbedRank::new().with_diversity(0.25).rank(&doc, &v, &d, 10);
        let names: Vec<&str> = got.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(names, vec!["deep nets", "graph kernels", "neural nets"]);
        let cos: Vec<f64> = ["deep nets", "neural nets", "graph kernels"]
            .iter()
            .map(|p| cosine(&v[*p], &d))
            .collect();
        let rel = normalize(&cos);
        assert!((got[0].1 - 0.25 * rel[0]).abs() < 1e-12, "{got:?}");
        assert!(
            (got[1].1 - (0.25 * rel[2] + 0.75 * 0.5)).abs() < 1e-9,
            "{got:?}"
        );
        let plain = EmbedRank::new().rank(&doc, &v, &d, 10);
        let same = EmbedRank::new().with_diversity(1.0).rank(&doc, &v, &d, 10);
        let order = |r: &[(String, f64)]| r.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>();
        assert_eq!(order(&plain), order(&same));
    }

    /// n = (x - min) / max for x = [0, 1, 2]: [0, 0.5, 1]; mean 0.5,
    /// std sqrt(1/6).
    #[test]
    fn normalization_matches_the_paper() {
        let got = normalize(&[0.0, 1.0, 2.0]);
        let s = (1.0f64 / 6.0).sqrt();
        assert!((got[0] - (0.5 - 0.5 / s)).abs() < 1e-12);
        assert!((got[1] - 0.5).abs() < 1e-12);
        assert!((got[2] - (0.5 + 0.5 / s)).abs() < 1e-12);
    }
}
