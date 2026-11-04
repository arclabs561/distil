//! CommunityCluster (Grineva, Grinev and Lizorkin, 2009).
//!
//! Grineva, M., Grinev, M. and Lizorkin, D. (2009). Extracting Key Terms
//! From Noisy and Multi-theme Documents. Proceedings of WWW 2009, pages
//! 661-670. doi:10.1145/1526709.1526798
//!
//! # Method
//!
//! 1. Terms are the candidates (noun phrases `(ADJ)*(NOUN|PROPN)+`; untagged
//!    text: stopword- and punctuation-delimited phrases of up to five
//!    words). Two terms are linked when their relatedness is positive, with
//!    the relatedness as edge weight. The paper uses Wikipedia-based
//!    relatedness; the default here counts occurrences of the two terms
//!    starting fewer than `w = 10` tokens apart, and
//!    [`CommunityCluster::rank_with`] takes the caller's measure.
//! 2. Communities come from the Girvan-Newman algorithm (Newman and Girvan,
//!    2004): repeatedly remove the edge of highest betweenness and keep the
//!    partition of highest modularity `Q = sum_c [L_c / m - (d_c / 2m)^2]`.
//! 3. A community's rank is its density (summed inner edge weights over its
//!    size) times its informativeness (mean keyphraseness of its terms; the
//!    paper's Wikipedia link probability, caller-supplied here, 1 by
//!    default).
//! 4. Communities are sorted by rank and cut at the largest drop between
//!    neighbours; terms of the communities above the drop are the
//!    keyphrases, each scored with its community's rank.
//!
//! # Deviations
//!
//! - graphops has Louvain and Leiden but not Girvan-Newman, which the paper
//!   uses, so edge betweenness (Brandes) and the removal loop are local.
//!   Betweenness and modularity ignore edge weights, as in Newman and
//!   Girvan's algorithm; ties remove the edge with the smallest node pair.
//! - Ties between terms are broken by first occurrence.

use crate::default_stopwords;
use crate::graph::rank;
use crate::text::{delimited_phrases, noun_phrases, Candidate, Document};
use std::collections::{HashSet, VecDeque};

/// CommunityCluster keyphrase extractor.
#[derive(Debug, Clone)]
pub struct CommunityCluster {
    stopwords: HashSet<String>,
    window: usize,
}

impl Default for CommunityCluster {
    fn default() -> Self {
        Self::new()
    }
}

impl CommunityCluster {
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

    /// Co-occurrence window in tokens for the default relatedness.
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(2);
        self
    }

    /// The candidate terms, in order of first occurrence.
    pub fn candidates(&self, doc: &Document) -> Vec<Candidate> {
        if doc.is_tagged() {
            noun_phrases(doc, usize::MAX)
        } else {
            delimited_phrases(doc, &self.stopwords, 5)
        }
    }

    /// Up to `max` keyphrases with co-occurrence relatedness and uniform
    /// keyphraseness, best first.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        let w = self.window;
        let rel = |a: &Candidate, b: &Candidate| {
            let pairs = a
                .offsets
                .iter()
                .flat_map(|&x| b.offsets.iter().map(move |&y| x.abs_diff(y)));
            pairs.filter(|&d| d > 0 && d < w).count() as f64
        };
        self.rank_with(doc, &rel, &|_| 1.0, max)
    }

    /// Up to `max` keyphrases with caller-supplied relatedness between
    /// candidates and keyphraseness of a candidate, best first.
    pub fn rank_with(
        &self,
        doc: &Document,
        relatedness: &dyn Fn(&Candidate, &Candidate) -> f64,
        keyphraseness: &dyn Fn(&Candidate) -> f64,
        max: usize,
    ) -> Vec<(String, f64)> {
        let cands = self.candidates(doc);
        let n = cands.len();
        let mut weight = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i + 1..n {
                let r = relatedness(&cands[i], &cands[j]);
                if r > 0.0 {
                    weight[i][j] = r;
                    weight[j][i] = r;
                }
            }
        }
        let mut communities: Vec<(Vec<usize>, f64)> = girvan_newman(&weight)
            .into_iter()
            .map(|c| {
                let inner: f64 = c
                    .iter()
                    .flat_map(|&i| c.iter().map(move |&j| (i, j)))
                    .filter(|(i, j)| i < j)
                    .map(|(i, j)| weight[i][j])
                    .sum();
                let info =
                    c.iter().map(|&i| keyphraseness(&cands[i])).sum::<f64>() / c.len() as f64;
                let r = inner / c.len() as f64 * info;
                (c, r)
            })
            .collect();
        communities.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0[0].cmp(&b.0[0])));
        let keep = (0..communities.len().saturating_sub(1))
            .map(|i| (i, communities[i].1 - communities[i + 1].1))
            .fold(None, |best: Option<(usize, f64)>, (i, d)| match best {
                Some((_, bd)) if bd >= d => best,
                _ => Some((i, d)),
            })
            .map_or(communities.len(), |(i, _)| i + 1);
        let scored = communities[..keep]
            .iter()
            .flat_map(|(c, r)| c.iter().map(|&i| (&cands[i], *r)));
        rank(scored, max)
    }
}

/// Girvan-Newman communities of the graph with positive `weight` entries,
/// each sorted, in order of their smallest node.
fn girvan_newman(weight: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = weight.len();
    let mut adj: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..n).filter(|&j| weight[i][j] > 0.0).collect())
        .collect();
    let degree: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
    let m: f64 = degree.iter().sum::<f64>() / 2.0;
    let original = adj.clone();
    let mut best = (f64::NEG_INFINITY, components(&adj));
    loop {
        let parts = components(&adj);
        if m > 0.0 {
            let q: f64 = parts
                .iter()
                .map(|c| {
                    let set: HashSet<usize> = c.iter().copied().collect();
                    let l = c
                        .iter()
                        .map(|&i| original[i].iter().filter(|j| set.contains(j)).count())
                        .sum::<usize>() as f64
                        / 2.0;
                    let d: f64 = c.iter().map(|&i| degree[i]).sum();
                    l / m - (d / (2.0 * m)).powi(2)
                })
                .sum();
            if q > best.0 + 1e-12 {
                best = (q, parts);
            }
        }
        let Some((a, b)) = max_betweenness_edge(&adj) else {
            break;
        };
        adj[a].retain(|&x| x != b);
        adj[b].retain(|&x| x != a);
    }
    best.1
}

fn components(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut seen = vec![false; adj.len()];
    let mut out = Vec::new();
    for s in 0..adj.len() {
        if seen[s] {
            continue;
        }
        seen[s] = true;
        let mut comp = vec![s];
        let mut queue = VecDeque::from([s]);
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    comp.push(v);
                    queue.push_back(v);
                }
            }
        }
        comp.sort_unstable();
        out.push(comp);
    }
    out
}

/// Unweighted edge betweenness (Brandes 2001); the edge `(a, b)`, `a < b`,
/// with the highest value, smallest pair first among ties.
fn max_betweenness_edge(adj: &[Vec<usize>]) -> Option<(usize, usize)> {
    let n = adj.len();
    let mut eb = std::collections::BTreeMap::<(usize, usize), f64>::new();
    for s in 0..n {
        let mut order = Vec::new();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0.0; n];
        let mut dist = vec![usize::MAX; n];
        sigma[s] = 1.0;
        dist[s] = 0;
        let mut queue = VecDeque::from([s]);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in &adj[u] {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    queue.push_back(v);
                }
                if dist[v] == dist[u] + 1 {
                    sigma[v] += sigma[u];
                    preds[v].push(u);
                }
            }
        }
        let mut delta = vec![0.0; n];
        for &w in order.iter().rev() {
            for &v in &preds[w] {
                let c = sigma[v] / sigma[w] * (1.0 + delta[w]);
                *eb.entry((v.min(w), v.max(w))).or_insert(0.0) += c;
                delta[v] += c;
            }
        }
    }
    eb.into_iter()
        .fold(
            None,
            |best: Option<((usize, usize), f64)>, (e, b)| match best {
                Some((_, bb)) if bb >= b - 1e-12 => best,
                _ => Some((e, b)),
            },
        )
        .map(|(e, _)| e)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangles joined by one bridge: the bridge has the highest
    /// betweenness (9 shortest paths cross it), and removing it gives
    /// Q = 2 * (3/7 - (7/14)^2) = 5/14, the best partition.
    #[test]
    fn girvan_newman_splits_two_triangles() {
        let mut w = vec![vec![0.0; 6]; 6];
        for (a, b) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)] {
            w[a][b] = 1.0;
            w[b][a] = 1.0;
        }
        let adj: Vec<Vec<usize>> = (0..6)
            .map(|i| (0..6).filter(|&j| w[i][j] > 0.0).collect())
            .collect();
        assert_eq!(max_betweenness_edge(&adj), Some((2, 3)));
        assert_eq!(girvan_newman(&w), vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    /// Communities {a, b, c} (dense, inner weight 3 / 3 = 1) and {d, e}
    /// (inner weight 0.1 / 2): the drop after the first community cuts the
    /// second away.
    #[test]
    fn keeps_communities_above_the_largest_drop() {
        let doc = Document::from_text("alpha, beta, gamma, delta, epsilon");
        let rel = |x: &Candidate, y: &Candidate| {
            let g1 = ["alpha", "beta", "gamma"];
            let (a, b) = (x.surface.as_str(), y.surface.as_str());
            if g1.contains(&a) && g1.contains(&b) {
                1.0
            } else if !g1.contains(&a) && !g1.contains(&b) {
                0.1
            } else {
                0.0
            }
        };
        let got = CommunityCluster::new().rank_with(&doc, &rel, &|_| 1.0, 10);
        assert_eq!(
            got,
            vec![
                ("alpha".to_string(), 1.0),
                ("beta".to_string(), 1.0),
                ("gamma".to_string(), 1.0)
            ]
        );
    }
}
