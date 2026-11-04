//! TopicRank (Bougouin, Boudin and Daille, 2013).
//!
//! Bougouin, A., Boudin, F. and Daille, B. (2013). TopicRank: Graph-Based
//! Topic Ranking for Keyphrase Extraction. Proceedings of IJCNLP 2013, pages
//! 543-551. <https://aclanthology.org/I13-1062/>
//!
//! # Method
//!
//! 1. Candidates are the longest sequences of nouns and adjectives, filtered
//!    by [`crate::text::filter_candidates`] (untagged text: stopword- and
//!    punctuation-delimited phrases).
//! 2. Candidates are clustered into topics by hierarchical agglomerative
//!    clustering with average linkage. The paper calls two candidates
//!    similar when they share at least 25% of their stemmed words; following
//!    pke, the distance is the Jaccard distance between stem count vectors
//!    and clusters merge while their distance is at most 0.74.
//! 3. Topics form a complete graph. The weight between topics `t_i` and
//!    `t_j` sums the reciprocal distances between all occurrences of their
//!    candidates:
//!
//!    $$w_{i,j} = \sum_{c_i \in t_i} \sum_{c_j \in t_j} \sum_{p_i \in pos(c_i)} \sum_{p_j \in pos(c_j)} \frac{1}{|p_i - p_j|}$$
//!
//!    where the gap is measured from the end of the earlier candidate (pke's
//!    `compute_gap`).
//! 4. Topics are ranked by weighted PageRank, `lambda = 0.85`, and each
//!    topic is represented by its first-occurring candidate.
//!
//! # Deviations
//!
//! - The Jaccard distance on count vectors (scipy's definition: a word
//!   counts as differing when its counts differ) and the 0.74 threshold are
//!   pke's reading of "25% overlap".
//! - Gaps are clamped to at least 1; pke clamps only 0, which matters only
//!   for overlapping candidates, and the selections here never overlap.
//! - Ties are broken by first occurrence.

use crate::graph::{pagerank, rank};
use crate::text::{
    delimited_phrases, filter_candidates, longest_sequences, Candidate, Document, Pos,
};
use crate::{default_stopwords, KeywordExtractor};
use std::collections::{HashMap, HashSet};

/// TopicRank keyphrase extractor.
#[derive(Debug, Clone)]
pub struct TopicRank {
    stopwords: HashSet<String>,
    threshold: f64,
}

impl Default for TopicRank {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicRank {
    /// Clustering threshold 0.74, English stopwords.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            threshold: 0.74,
        }
    }

    /// Stopwords for candidate filtering (and selection in untagged text).
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Largest average-linkage distance at which clusters merge.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        if threshold.is_finite() {
            self.threshold = threshold;
        }
        self
    }

    /// Candidates grouped into topics, each topic a list of candidates in
    /// order of first occurrence; topics are ordered the same way.
    pub fn topics(&self, doc: &Document) -> Vec<Vec<Candidate>> {
        let candidates = topic_candidates(doc, &self.stopwords);
        let clusters = cluster(&candidates, self.threshold);
        clusters
            .into_iter()
            .map(|c| c.into_iter().map(|i| candidates[i].clone()).collect())
            .collect()
    }

    /// Up to `max` keyphrases (one per topic), best first.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        let topics = self.topics(doc);
        let n = topics.len();
        let mut adj = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i + 1..n {
                let w: f64 = topics[i]
                    .iter()
                    .flat_map(|a| topics[j].iter().map(move |b| proximity(a, b)))
                    .sum();
                adj[i][j] = w;
                adj[j][i] = w;
            }
        }
        let scores = pagerank(&adj, None);
        rank(topics.iter().map(|t| &t[0]).zip(scores), max)
    }
}

/// Longest noun/adjective sequences (tagged) or delimited phrases, then
/// pke's candidate filter with at most five words.
pub(crate) fn topic_candidates(doc: &Document, stopwords: &HashSet<String>) -> Vec<Candidate> {
    let raw = if doc.is_tagged() {
        longest_sequences(doc, |t| t.pos.is_some_and(Pos::is_noun_or_adjective))
    } else {
        delimited_phrases(doc, stopwords, 5)
    };
    filter_candidates(raw, stopwords, 5)
}

/// Sum of `1 / gap` over all occurrence pairs of two candidates.
pub(crate) fn proximity(a: &Candidate, b: &Candidate) -> f64 {
    let mut w = 0.0;
    for &pa in &a.offsets {
        for &pb in &b.offsets {
            let gap = if pa < pb {
                pb - pa - (a.len() - 1).min(pb - pa)
            } else {
                pa - pb - (b.len() - 1).min(pa - pb)
            };
            w += 1.0 / gap.max(1) as f64;
        }
    }
    w
}

/// Jaccard distance between stem count vectors: the share of words, among
/// those in either candidate, whose counts differ.
fn distance(a: &Candidate, b: &Candidate) -> f64 {
    fn count(c: &Candidate) -> HashMap<&str, usize> {
        let mut m = HashMap::new();
        for w in &c.lexical_form {
            *m.entry(w.as_str()).or_insert(0) += 1;
        }
        m
    }
    let (ca, cb) = (count(a), count(b));
    let words: HashSet<&str> = ca.keys().chain(cb.keys()).copied().collect();
    let differ = words.iter().filter(|w| ca.get(*w) != cb.get(*w)).count();
    differ as f64 / words.len() as f64
}

/// Average-linkage agglomerative clustering cut at `threshold`: candidates
/// end up together when the dendrogram joins them at a distance of at most
/// `threshold`. Returns clusters of candidate indices, each sorted, ordered
/// by their first index.
///
/// The dendrogram is built with the nearest-neighbour chain algorithm in the
/// same way as scipy's `linkage(method="average")`, on candidates sorted by
/// lexical form as pke sorts them, so that ties between equal distances
/// (common with short candidates) resolve as they do in pke.
pub(crate) fn cluster(candidates: &[Candidate], threshold: f64) -> Vec<Vec<usize>> {
    let n = candidates.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| candidates[i].lexical_form.join(" "));
    let mut d: Vec<Vec<f64>> = order
        .iter()
        .map(|&a| {
            order
                .iter()
                .map(|&b| distance(&candidates[a], &candidates[b]))
                .collect()
        })
        .collect();
    let mut size = vec![1usize; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut group: Vec<usize> = (0..n).collect();
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    for _ in 1..n {
        if chain.is_empty() {
            chain.extend((0..n).find(|&i| size[i] > 0));
        }
        let (x, y, dist) = loop {
            let x = chain[chain.len() - 1];
            // Prefer the previous chain element on ties, to avoid cycles.
            let (mut y, mut min) = match chain.len() {
                1 => (x, f64::INFINITY),
                len => (chain[len - 2], d[x][chain[len - 2]]),
            };
            for i in 0..n {
                if size[i] > 0 && i != x && d[x][i] < min {
                    min = d[x][i];
                    y = i;
                }
            }
            if chain.len() > 1 && y == chain[chain.len() - 2] {
                chain.truncate(chain.len() - 2);
                break (x.min(y), x.max(y), min);
            }
            chain.push(y);
        };
        let (nx, ny) = (size[x] as f64, size[y] as f64);
        size[y] += size[x];
        size[x] = 0;
        for i in 0..n {
            if size[i] > 0 && i != y {
                let v = (nx * d[i][x] + ny * d[i][y]) / (nx + ny);
                d[i][y] = v;
                d[y][i] = v;
            }
        }
        let moved = std::mem::take(&mut members[x]);
        if dist <= threshold {
            // Average linkage never merges below a child's height, so the
            // merges at or under the cut are exactly the flat clusters.
            let target = group[members[y][0]];
            for &m in &moved {
                let from = group[m];
                for g in group.iter_mut().filter(|g| **g == from) {
                    *g = target;
                }
            }
        }
        members[y].extend(moved);
    }
    let mut by_group: HashMap<usize, Vec<usize>> = HashMap::new();
    for (sorted_pos, &g) in group.iter().enumerate() {
        by_group.entry(g).or_default().push(order[sorted_pos]);
    }
    let mut out: Vec<Vec<usize>> = by_group
        .into_values()
        .map(|mut c| {
            c.sort_unstable();
            c
        })
        .collect();
    out.sort_by_key(|c| c[0]);
    out
}

impl KeywordExtractor for TopicRank {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.rank(&Document::from_text(text), max_keywords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(words: &[&str], offsets: &[usize]) -> Candidate {
        Candidate {
            lexical_form: words.iter().map(|w| w.to_string()).collect(),
            surface: words.join(" "),
            offsets: offsets.to_vec(),
        }
    }

    #[test]
    fn gap_is_measured_from_the_end_of_the_earlier_candidate() {
        // "neural network" at 0-1, "training" at 5: gap 5 - 0 - 1 = 4.
        let a = cand(&["neural", "network"], &[0]);
        let b = cand(&["train"], &[5]);
        assert!((proximity(&a, &b) - 0.25).abs() < 1e-12);
        assert!((proximity(&b, &a) - 0.25).abs() < 1e-12);
        // Two occurrences of b: 1/4 + 1/1.
        let b2 = cand(&["train"], &[5, 2]);
        assert!((proximity(&a, &b2) - 1.25).abs() < 1e-12);
    }

    #[test]
    fn jaccard_on_counts() {
        let a = cand(&["neural", "network"], &[0]);
        let b = cand(&["network"], &[3]);
        assert!((distance(&a, &b) - 0.5).abs() < 1e-12);
        let c = cand(&["network", "network"], &[6]);
        assert!((distance(&b, &c) - 1.0).abs() < 1e-12);
    }

    /// Distances: ab-a 0.5, ab-c 1, a-c 1. "ab" and "a" merge at 0.5; the
    /// merged cluster is at distance 1 from "c", above the threshold.
    #[test]
    fn average_linkage_with_threshold() {
        let c = vec![
            cand(&["a", "b"], &[0]),
            cand(&["c"], &[3]),
            cand(&["a"], &[5]),
        ];
        assert_eq!(cluster(&c, 0.74), vec![vec![0, 2], vec![1]]);
        assert_eq!(cluster(&c, 0.4), vec![vec![0], vec![1], vec![2]]);
        assert_eq!(cluster(&c, 1.0), vec![vec![0, 1, 2]]);
    }

    /// A = "neural networks model language", B = "neural network needs
    /// data", C = "language models predict words": d(A, B) = d(A, C) = 4/6,
    /// d(B, C) = 1. Sorted by lexical form the order is C, A, B; the chain
    /// starts at C, whose nearest neighbour is A, and A prefers C on the tie,
    /// so A and C merge (scipy's result), and B stays alone at 5/6.
    #[test]
    fn ties_resolve_like_scipy_nn_chain() {
        let text = "Neural networks model language. A neural network needs data. \
                    Language models predict words.";
        let topics: Vec<Vec<String>> = TopicRank::new()
            .topics(&Document::from_text(text))
            .into_iter()
            .map(|t| t.into_iter().map(|c| c.surface).collect())
            .collect();
        assert_eq!(
            topics,
            vec![
                vec![
                    "neural networks model language",
                    "language models predict words"
                ],
                vec!["neural network needs data"],
            ]
        );
    }
}
