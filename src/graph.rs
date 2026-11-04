//! Word graphs and PageRank shared by the graph keyphrase methods.

use crate::text::{is_content, Candidate, Document, Token};
use std::collections::{HashMap, HashSet};

/// PageRank damping factor used by every method in this crate, as in all of
/// their papers.
pub(crate) const DAMPING: f64 = 0.85;

/// Convergence threshold on the L1 change between iterations. Tighter than
/// the papers' 1e-4 so that rankings reflect the stationary distribution
/// rather than where the iteration stopped.
const TOLERANCE: f64 = 1e-10;
const MAX_ITERATIONS: usize = 1000;

/// Graph-node filter: nouns, proper nouns and adjectives for tagged tokens,
/// non-stopword non-punctuation tokens otherwise.
pub(crate) fn is_graph_word(token: &Token, stopwords: &HashSet<String>) -> bool {
    match token.pos {
        Some(p) => p.is_noun_or_adjective(),
        None => is_content(token, stopwords),
    }
}

/// An undirected weighted graph over stems, nodes in order of first
/// occurrence.
#[derive(Default)]
pub(crate) struct WordGraph {
    pub nodes: Vec<String>,
    index: HashMap<String, usize>,
    edges: HashMap<(usize, usize), f64>,
}

impl WordGraph {
    fn node(&mut self, stem: &str) -> usize {
        if let Some(&i) = self.index.get(stem) {
            return i;
        }
        self.index.insert(stem.to_string(), self.nodes.len());
        self.nodes.push(stem.to_string());
        self.nodes.len() - 1
    }

    /// Add one document's co-occurrences. Two accepted tokens co-occur when
    /// fewer than `window` tokens apart, counting every token and ignoring
    /// sentence boundaries (as pke does); each co-occurrence of two
    /// different stems adds `weight` to their edge.
    pub fn add_document(
        &mut self,
        doc: &Document,
        accept: &dyn Fn(&Token) -> bool,
        window: usize,
        weight: f64,
    ) {
        let tokens: Vec<(usize, Option<usize>)> = doc
            .tokens()
            .enumerate()
            .map(|(i, t)| (i, accept(t).then(|| self.node(&t.stem))))
            .collect();
        let accepted: Vec<(usize, usize)> = tokens
            .iter()
            .filter_map(|&(i, n)| n.map(|n| (i, n)))
            .collect();
        for (a, &(pos_a, node_a)) in accepted.iter().enumerate() {
            for &(pos_b, node_b) in &accepted[a + 1..] {
                if pos_b - pos_a >= window {
                    break;
                }
                if node_a != node_b {
                    let key = (node_a.min(node_b), node_a.max(node_b));
                    *self.edges.entry(key).or_insert(0.0) += weight;
                }
            }
        }
    }

    /// Set every edge weight to 1 (TextRank's unweighted graph).
    pub fn unweighted(mut self) -> Self {
        self.edges.values_mut().for_each(|w| *w = 1.0);
        self
    }

    fn matrix(&self) -> Vec<Vec<f64>> {
        let n = self.nodes.len();
        let mut adj = vec![vec![0.0; n]; n];
        for (&(i, j), &w) in &self.edges {
            adj[i][j] = w;
            adj[j][i] = w;
        }
        adj
    }

    /// PageRank score of each stem; with `personalization` (a weight per
    /// node, in node order) the teleport and dangling mass follow it.
    pub fn scores(&self, personalization: Option<&[f64]>) -> HashMap<String, f64> {
        let ranks = pagerank(&self.matrix(), personalization);
        self.nodes.iter().cloned().zip(ranks).collect()
    }
}

/// Weighted PageRank over a dense adjacency matrix (`adj[i][j]` is the weight
/// of the edge from `i` to `j`).
///
/// Without personalization this is `graphops::pagerank_weighted`: teleport
/// and dangling-node mass are spread uniformly. graphops (through 0.5) has no
/// weighted personalized PageRank, so that case is computed here with
/// networkx's convention: teleport and dangling mass both follow the
/// normalized personalization vector.
pub(crate) fn pagerank(adj: &[Vec<f64>], personalization: Option<&[f64]>) -> Vec<f64> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    let Some(p) = personalization.filter(|p| p.iter().sum::<f64>() > 0.0) else {
        let config = graphops::PageRankConfig {
            damping: DAMPING,
            max_iterations: MAX_ITERATIONS,
            tolerance: TOLERANCE,
        };
        return graphops::pagerank_weighted(&graphops::AdjacencyMatrix(adj), config);
    };
    let total: f64 = p.iter().sum();
    let p: Vec<f64> = p.iter().map(|x| x / total).collect();
    let out: Vec<f64> = adj.iter().map(|row| row.iter().sum()).collect();
    let mut s = vec![1.0 / n as f64; n];
    for _ in 0..MAX_ITERATIONS {
        let dangling: f64 = (0..n).filter(|&i| out[i] == 0.0).map(|i| s[i]).sum();
        let mut next: Vec<f64> = p
            .iter()
            .map(|pi| (1.0 - DAMPING) * pi + DAMPING * dangling * pi)
            .collect();
        for (u, row) in adj.iter().enumerate() {
            if out[u] > 0.0 {
                let share = DAMPING * s[u] / out[u];
                for (v, &w) in row.iter().enumerate() {
                    if w > 0.0 {
                        next[v] += share * w;
                    }
                }
            }
        }
        let diff: f64 = s.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
        s = next;
        if diff < TOLERANCE {
            break;
        }
    }
    s
}

/// Score each candidate by the sum of its words' scores (words missing from
/// `words` count 0), then rank.
pub(crate) fn rank_by_word_sum(
    candidates: &[Candidate],
    words: &HashMap<String, f64>,
    max: usize,
) -> Vec<(String, f64)> {
    let scored = candidates.iter().map(|c| {
        let s = c.lexical_form.iter().filter_map(|w| words.get(w)).sum();
        (c, s)
    });
    rank(scored, max)
}

/// Sort by descending score; ties go to the earlier first occurrence.
pub(crate) fn rank<'a>(
    scored: impl Iterator<Item = (&'a Candidate, f64)>,
    max: usize,
) -> Vec<(String, f64)> {
    let mut v: Vec<_> = scored.collect();
    v.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| a.0.first_offset().cmp(&b.0.first_offset()))
    });
    v.into_iter()
        .take(max)
        .map(|(c, s)| (c.surface.clone(), s))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path a - b - c with unit weights. By symmetry s_a = s_c and
    /// s_a = 0.05 + 0.85 * s_b / 2, s_b = 0.05 + 0.85 * 2 * s_a, so
    /// s_a = 0.07125 / 0.2775 and s_b = 0.05 + 1.7 * s_a.
    #[test]
    fn pagerank_on_a_path_matches_the_closed_form() {
        let adj = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];
        let sa = 0.07125 / 0.2775;
        let sb = 0.05 + 1.7 * sa;
        for p in [None, Some(&[1.0, 1.0, 1.0][..])] {
            let s = pagerank(&adj, p);
            assert!(
                (s[0] - sa).abs() < 1e-9 && (s[2] - sa).abs() < 1e-9,
                "{s:?}"
            );
            assert!((s[1] - sb).abs() < 1e-9, "{s:?}");
        }
    }

    /// Two nodes a - b plus an isolated node c, teleporting only to a:
    /// c is dangling and gets nothing, so s_c = 0, and
    /// s_a = 0.15 + 0.85 s_b + 0.85 s_c, s_b = 0.85 s_a, giving
    /// s_a = 0.15 / (1 - 0.7225).
    #[test]
    fn personalized_pagerank_sends_teleport_and_dangling_mass_to_the_bias() {
        let adj = vec![
            vec![0.0, 2.0, 0.0],
            vec![2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        let s = pagerank(&adj, Some(&[1.0, 0.0, 0.0]));
        let sa = 0.15 / (1.0 - 0.7225);
        assert!((s[0] - sa).abs() < 1e-9, "{s:?}");
        assert!((s[1] - 0.85 * sa).abs() < 1e-9, "{s:?}");
        assert!(s[2].abs() < 1e-12, "{s:?}");
    }

    #[test]
    fn window_counts_all_tokens() {
        // "graph" and "model" are 2 tokens apart: linked at window 3, not 2.
        let doc = Document::from_text("graph of model");
        let stop = crate::default_stopwords();
        let accept = |t: &Token| is_graph_word(t, &stop);
        let mut g = WordGraph::default();
        g.add_document(&doc, &accept, 2, 1.0);
        assert!(g.edges.is_empty());
        let mut g = WordGraph::default();
        g.add_document(&doc, &accept, 3, 1.0);
        assert_eq!(g.edges.get(&(0, 1)), Some(&1.0));
    }
}
