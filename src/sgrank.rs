//! SGRank (Danesh, Sumner and Martin, 2015).
//!
//! Danesh, S., Sumner, T. and Martin, J. H. (2015). SGRank: Combining
//! Statistical and Graphical Methods to Improve the State of the Art in
//! Unsupervised Keyphrase Extraction. Proceedings of *SEM 2015, pages
//! 117-126. ACL Anthology S15-1013.
//!
//! # Method
//!
//! 1. Candidates are n-grams of one to six words with no stopword or
//!    punctuation (tagged text: only nouns, proper nouns, adjectives and
//!    verbs). Documents of 1,500 to 4,000 words drop candidates seen fewer
//!    than 2 times, longer documents fewer than 3.
//! 2. Candidates are ranked by `tf * idf`, with `idf = ln(D / df)` for single
//!    words and `ln(D)` for longer candidates (assumed to occur in one
//!    document, as in KP-Miner).
//! 3. The top `T = 100` are re-weighted:
//!
//!    $$w_s(t) = (tf(t) - subSumCount(t)) \cdot idf(t) \cdot PFO(t) \cdot TL(t), \qquad PFO(t) = \ln\frac{cutoff}{p(t)}$$
//!
//!    where `subSumCount` sums the frequencies of top-`T` candidates that
//!    contain `t`, `p(t)` is the 1-based token position of its first
//!    occurrence, `cutoff = 3000`, and `TL` is its length in words.
//! 4. Candidates with `w_s > 0` become nodes. Two occurrences closer than
//!    `winSize = 1500` tokens co-occur; the distance weight is the mean of
//!    `ln(winSize / |pos_i - pos_j|)` over co-occurrences, and the edge is
//!    `w_d(t_i, t_j) * w_s(t_i) * w_s(t_j)`. The final score is weighted
//!    PageRank (damping 0.85).
//!
//! # Deviations
//!
//! - Without document frequencies every `idf` is 1, which ranks stage 2 by
//!   term frequency; pass a [`DocumentFrequency`] for the paper's weights.
//! - Overlapping occurrences (distance 0) do not co-occur.
//! - The paper writes PageRank's damping as 0.15 in a form where it is the
//!   teleport probability; this uses the usual 0.85 continuation weight.
//! - Ties are broken by first occurrence.

use crate::graph::{pagerank, rank};
use crate::text::{is_content, ngrams_where, Candidate, Document, Pos};
use crate::tfidf::DocumentFrequency;
use crate::{default_stopwords, KeywordExtractor};
use std::collections::HashSet;

/// SGRank keyphrase extractor.
#[derive(Debug, Clone)]
pub struct SgRank {
    stopwords: HashSet<String>,
    top: usize,
    cutoff: f64,
    window: usize,
}

impl Default for SgRank {
    fn default() -> Self {
        Self::new()
    }
}

impl SgRank {
    /// `T = 100`, `cutoff = 3000`, `winSize = 1500`, English stopwords.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            top: 100,
            cutoff: 3000.0,
            window: 1500,
        }
    }

    /// Stopwords excluded from candidates.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Up to `max` keyphrases, best first, without corpus statistics.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        self.rank_with(doc, None, max)
    }

    /// Up to `max` keyphrases, best first, with document frequencies for
    /// single-word idf.
    pub fn rank_with(
        &self,
        doc: &Document,
        df: Option<&DocumentFrequency>,
        max: usize,
    ) -> Vec<(String, f64)> {
        let tagged = doc.is_tagged();
        let keep = |t: &crate::text::Token| {
            is_content(t, &self.stopwords)
                && (!tagged
                    || t.pos
                        .is_some_and(|p| p.is_noun_or_adjective() || p == Pos::Verb))
        };
        let words = doc.tokens().filter(|t| !t.is_punct()).count();
        let min_tf = match words {
            0..=1499 => 0,
            1500..=4000 => 2,
            _ => 3,
        };
        let mut cands = ngrams_where(doc, keep, 6);
        cands.retain(|c| c.offsets.len() >= min_tf);
        let idf = |c: &Candidate| match df {
            None => 1.0,
            Some(df) if c.len() == 1 => {
                (df.documents() as f64 / df.get(&c.lexical_form[0]).max(1) as f64).ln()
            }
            Some(df) => (df.documents() as f64).ln(),
        };
        // Stage 2: the top T by tf * idf.
        let mut order: Vec<(usize, f64)> = cands
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.offsets.len() as f64 * idf(c)))
            .collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        order.truncate(self.top);
        let top: Vec<&Candidate> = order.iter().map(|&(i, _)| &cands[i]).collect();
        // Stage 3: statistical weights.
        let weighted: Vec<(&Candidate, f64)> = top
            .iter()
            .map(|&c| {
                let subsumed: usize = top
                    .iter()
                    .filter(|o| {
                        o.len() > c.len()
                            && o.lexical_form.windows(c.len()).any(|w| w == c.lexical_form)
                    })
                    .map(|o| o.offsets.len())
                    .sum();
                let tf = c.offsets.len() as f64 - subsumed as f64;
                let pfo = (self.cutoff / (c.first_offset() + 1) as f64).ln();
                (c, tf * idf(c) * pfo * c.len() as f64)
            })
            .filter(|&(_, w)| w > 0.0)
            .collect();
        // Stage 4: distance-weighted graph.
        let n = weighted.len();
        let mut adj = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i + 1..n {
                let (mut sum, mut count) = (0.0, 0.0);
                for &a in &weighted[i].0.offsets {
                    for &b in &weighted[j].0.offsets {
                        let d = a.abs_diff(b);
                        if d > 0 && d < self.window {
                            sum += (self.window as f64 / d as f64).ln();
                            count += 1.0;
                        }
                    }
                }
                if count > 0.0 {
                    let w = sum / count * weighted[i].1 * weighted[j].1;
                    adj[i][j] = w;
                    adj[j][i] = w;
                }
            }
        }
        let scores = pagerank(&adj, None);
        rank(weighted.iter().map(|&(c, _)| c).zip(scores), max)
    }
}

impl KeywordExtractor for SgRank {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.rank(&Document::from_text(text), max_keywords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "graph model" x2 subsumes "graph" and "model" (x2 each), whose
    /// weights drop to 0 and leave the graph; "tree" (first at token 7)
    /// keeps w_s = ln(3000 / 8). With two nodes PageRank is symmetric.
    #[test]
    fn subsumed_candidates_drop_out() {
        let doc = Document::from_text("graph model, graph model, the tree");
        let got = SgRank::new().rank(&doc, 10);
        let names: HashSet<&str> = got.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(names, HashSet::from(["graph model", "tree"]));
        assert!(
            (got[0].1 - 0.5).abs() < 1e-9 && (got[1].1 - 0.5).abs() < 1e-9,
            "{got:?}"
        );
    }

    /// Nodes alpha, beta, gamma at tokens 0, 2 and 4 with
    /// w_s = ln(3000 / p) for p = 1, 3, 5. The edge between two nodes d
    /// tokens apart is ln(1500 / d) * w_s * w_s; the scores are PageRank
    /// over that matrix, built here from the equations.
    #[test]
    fn edges_follow_the_distance_and_statistical_weights() {
        let doc = Document::from_text("alpha, beta. gamma");
        let got: std::collections::HashMap<String, f64> =
            SgRank::new().rank(&doc, 10).into_iter().collect();
        let ws = [3000f64.ln(), 1000f64.ln(), 600f64.ln()];
        let pos = [0.0, 2.0, 4.0];
        let adj: Vec<Vec<f64>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        if i == j {
                            0.0
                        } else {
                            (1500.0 / f64::abs(pos[i] - pos[j])).ln() * ws[i] * ws[j]
                        }
                    })
                    .collect()
            })
            .collect();
        let want = pagerank(&adj, None);
        for (k, name) in ["alpha", "beta", "gamma"].iter().enumerate() {
            assert!((got[*name] - want[k]).abs() < 1e-12, "{got:?} {want:?}");
        }
    }
}
