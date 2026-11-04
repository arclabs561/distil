//! CollabRank (Wan and Xiao, 2008).
//!
//! Wan, X. and Xiao, J. (2008). CollabRank: Towards a Collaborative Approach
//! to Single-Document Keyphrase Extraction. Proceedings of COLING 2008,
//! pages 969-976. <https://aclanthology.org/C08-1122/>
//!
//! # Method
//!
//! The caller supplies a cluster of topic-related documents (the paper
//! clusters a collection with k-means or agglomerative clustering).
//!
//! 1. Cluster-level word evaluation: one graph over the noun and adjective
//!    stems of every document in the cluster, edges weighted by the
//!    co-occurrence count within a window of `k = 10` tokens summed over
//!    the documents (Eq. 1):
//!
//!    $$aff(v_i, v_j) = \sum_{d \in C} count_d(v_i, v_j)$$
//!
//!    Word saliency is weighted PageRank on it, `mu = 0.85` (Eq. 4).
//! 2. Document-level extraction: for each document, candidates are runs of
//!    adjacent nouns and adjectives ending in a noun, scored by the sum of
//!    their words' cluster-level saliency (Eq. 6).
//!
//! This is ExpandRank with every similarity set to 1, computed once per
//! cluster. Untagged text uses non-stopword, non-punctuation graph words and
//! stopword-delimited candidates of up to five words.
//!
//! # Deviations
//!
//! - Graph nodes are stems; the paper uses unstemmed words.
//! - The window counts all tokens and ignores sentence boundaries. Iteration
//!   runs to an L1 change of 1e-10 (the paper: 1e-4).

use crate::expandrank::{paper_candidates, word_scores};
use crate::graph::rank_by_word_sum;
use crate::text::{Document, Roles};
use crate::KeywordExtractor;
use std::collections::{HashMap, HashSet};

/// CollabRank keyphrase extractor.
#[derive(Debug, Clone)]
pub struct CollabRank {
    stopwords: Roles,
    window: usize,
}

impl Default for CollabRank {
    fn default() -> Self {
        Self::new()
    }
}

impl CollabRank {
    /// Window 10, English stopwords for untagged text.
    pub fn new() -> Self {
        Self {
            stopwords: Roles::default(),
            window: 10,
        }
    }

    /// Stopwords used for untagged text.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords.set_stopwords(stopwords);
        self
    }

    /// Stopword language: its list replaces the current one, and text given
    /// to `extract` is stemmed with its stemmer. Sentences of a document in
    /// another language use that language's stopwords
    /// ([`Stoplist`](crate::text::Stoplist)).
    pub fn with_language(mut self, language: crate::Language) -> Self {
        self.stopwords.set_language(language);
        self
    }

    /// Replace the extra words that end candidate phrases in untagged text
    /// but still count as content words for scoring. The default is the
    /// SMART list (570 words) in English sentences and nothing in other
    /// languages. A list given here applies to sentences in the method's
    /// language; include SMART yourself to extend it with a domain list
    /// ("et", "al", "fig"), or pass an empty list for NLTK's list alone.
    pub fn with_delimiters<I, S>(mut self, words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.stopwords.set_delimiters(words);
        self
    }

    /// Co-occurrence window in tokens (at least 2).
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(2);
        self
    }

    /// Cluster-level saliency of each word.
    pub fn word_scores(&self, cluster: &[Document]) -> HashMap<String, f64> {
        let Some((first, rest)) = cluster.split_first() else {
            return HashMap::new();
        };
        let rest: Vec<(&Document, f64)> = rest.iter().map(|d| (d, 1.0)).collect();
        word_scores(first, &rest, self.stopwords.language(), self.window)
    }

    /// Up to `max` keyphrases for each document of the cluster, in cluster
    /// order, from one ranking of the whole cluster.
    pub fn rank_cluster(&self, cluster: &[Document], max: usize) -> Vec<Vec<(String, f64)>> {
        let scores = self.word_scores(cluster);
        cluster
            .iter()
            .map(|d| {
                rank_by_word_sum(
                    &paper_candidates(d, &self.stopwords.delimiting(d)),
                    &scores,
                    max,
                )
            })
            .collect()
    }
}

impl KeywordExtractor for CollabRank {
    /// A single document is a cluster of one.
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.rank_cluster(&[self.stopwords.document(text)], max_keywords)
            .pop()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expandrank::ExpandRank;

    #[test]
    fn cluster_equals_expandrank_with_unit_similarities() {
        let a = Document::from_text("Solar panels convert sunlight. Panel efficiency matters.");
        let b = Document::from_text("Sunlight intensity limits the output of a solar panel.");
        let c = Document::from_text("Panel cleaning raises efficiency.");
        let cluster = [a.clone(), b.clone(), c.clone()];
        let collab = CollabRank::new().word_scores(&cluster);
        let expand = ExpandRank::new().word_scores(&a, &[(&b, 1.0), (&c, 1.0)]);
        assert_eq!(collab.len(), expand.len());
        for (w, s) in &collab {
            assert!((s - expand[w]).abs() < 1e-12, "{w}");
        }
        let ranked = CollabRank::new().rank_cluster(&cluster, 3);
        assert_eq!(ranked.len(), 3);
        assert!(ranked.iter().all(|r| !r.is_empty()));
    }

    #[test]
    fn empty_cluster() {
        assert!(CollabRank::new().rank_cluster(&[], 5).is_empty());
        assert!(CollabRank::new().extract("", 5).is_empty());
    }
}
