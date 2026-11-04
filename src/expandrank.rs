//! ExpandRank (Wan and Xiao, 2008).
//!
//! Wan, X. and Xiao, J. (2008). Single Document Keyphrase Extraction Using
//! Neighborhood Knowledge. Proceedings of AAAI 2008, pages 855-860.
//!
//! # Method
//!
//! The document `d0` is expanded with `k` neighbour documents `d1..dk`
//! (found by the caller, in the paper by TF-IDF cosine similarity), each
//! with a similarity `sim(d0, dp)`; `sim(d0, d0) = 1`.
//!
//! 1. One global graph over the noun and adjective stems of all documents.
//!    Co-occurrences within a window of `w = 10` tokens are counted per
//!    document and weighted by that document's similarity (Eq. 2):
//!
//!    $$aff(v_i, v_j) = \sum_{d_p \in D} sim(d_0, d_p) \cdot count_{d_p}(v_i, v_j)$$
//!
//! 2. Word saliency is weighted PageRank on that graph, `mu = 0.85` (Eq. 5).
//! 3. Candidates in `d0` are runs of adjacent nouns and adjectives that end
//!    in a noun; a phrase scores the sum of its words' saliency (Eq. 7).
//!
//! With no neighbours this is SingleRank with the paper's candidate rule.
//! Untagged text uses non-stopword, non-punctuation words as graph words and
//! stopword-delimited phrases of up to five words as candidates.
//!
//! # Deviations
//!
//! - The paper does not stem graph words; nodes here are stems, as in the
//!   other graph methods.
//! - The window counts all tokens and ignores sentence boundaries. Iteration
//!   runs to an L1 change of 1e-10 (the paper: 1e-4).

use crate::graph::{graph_words, rank_by_word_sum, WordGraph};
use crate::text::{
    delimited_phrases, longest_sequences, Candidate, Document, Pos, Roles, Stoplist,
};
use crate::KeywordExtractor;
use std::collections::{HashMap, HashSet};

/// ExpandRank keyphrase extractor.
#[derive(Debug, Clone)]
pub struct ExpandRank {
    stopwords: Roles,
    window: usize,
}

impl Default for ExpandRank {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpandRank {
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
    /// ([`Stoplist`]).
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

    /// Saliency of each word in the graph of `doc` plus its neighbours.
    /// Neighbours with a non-positive or non-finite similarity are ignored.
    pub fn word_scores(
        &self,
        doc: &Document,
        neighbours: &[(&Document, f64)],
    ) -> HashMap<String, f64> {
        word_scores(doc, neighbours, self.stopwords.language(), self.window)
    }

    /// Up to `max` keyphrases of `doc`, best first.
    pub fn rank(
        &self,
        doc: &Document,
        neighbours: &[(&Document, f64)],
        max: usize,
    ) -> Vec<(String, f64)> {
        let candidates = paper_candidates(doc, &self.stopwords.delimiting(doc));
        rank_by_word_sum(&candidates, &self.word_scores(doc, neighbours), max)
    }
}

pub(crate) fn word_scores(
    doc: &Document,
    neighbours: &[(&Document, f64)],
    stopwords: &(impl Stoplist + ?Sized),
    window: usize,
) -> HashMap<String, f64> {
    let accept = graph_words(stopwords);
    let mut g = WordGraph::default();
    g.add_document(doc, &accept, window, 1.0);
    for &(n, sim) in neighbours {
        if sim.is_finite() && sim > 0.0 {
            g.add_document(n, &accept, window, sim);
        }
    }
    g.scores(None)
}

/// Runs of nouns and adjectives ending in a noun (tagged), otherwise
/// delimited phrases of at most five words.
pub(crate) fn paper_candidates(
    doc: &Document,
    stopwords: &(impl Stoplist + ?Sized),
) -> Vec<Candidate> {
    if !doc.is_tagged() {
        return delimited_phrases(doc, stopwords, 5);
    }
    let mut out = longest_sequences(doc, |t| t.pos.is_some_and(Pos::is_noun_or_adjective));
    let tokens: Vec<_> = doc.tokens().collect();
    out.retain(|c| tokens[c.first_offset() + c.len() - 1].pos != Some(Pos::Adj));
    out
}

impl KeywordExtractor for ExpandRank {
    /// Without neighbour documents ExpandRank is SingleRank.
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.rank(&self.stopwords.document(text), &[], max_keywords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::singlerank::SingleRank;

    const DOC: &str = "Wind turbines convert wind energy. Offshore wind turbines \
                       need strong foundations. Turbine blades fail under storms.";

    #[test]
    fn no_neighbours_matches_singlerank_word_scores() {
        let doc = Document::from_text(DOC);
        let a = ExpandRank::new().word_scores(&doc, &[]);
        let b = SingleRank::new().word_scores(&doc);
        assert_eq!(a.len(), b.len());
        for (w, s) in &a {
            assert!((s - b[w]).abs() < 1e-12, "{w}");
        }
    }

    /// A copy of the document at similarity 1 doubles every edge weight,
    /// which leaves the transition matrix, and so the scores, unchanged.
    #[test]
    fn identical_neighbour_changes_nothing() {
        let doc = Document::from_text(DOC);
        let er = ExpandRank::new();
        let a = er.word_scores(&doc, &[]);
        let b = er.word_scores(&doc, &[(&doc, 1.0)]);
        for (w, s) in &a {
            assert!((s - b[w]).abs() < 1e-9, "{w}");
        }
    }

    #[test]
    fn neighbour_vocabulary_shifts_the_ranking() {
        let doc = Document::from_text(DOC);
        let other = Document::from_text("Storms damage turbine blades. Blades crack in storms.");
        let er = ExpandRank::new();
        let alone = er.word_scores(&doc, &[]);
        let with = er.word_scores(&doc, &[(&other, 0.8)]);
        assert!(with["storm"] / with["wind"] > alone["storm"] / alone["wind"]);
        // A neighbour at similarity 0 is ignored.
        let ignored = er.word_scores(&doc, &[(&other, 0.0)]);
        assert_eq!(ignored.len(), alone.len());
    }

    #[test]
    fn tagged_phrases_must_end_in_a_noun() {
        let s: Vec<(&str, Pos)> = vec![
            ("Mad", Pos::Adj),
            ("cow", Pos::Noun),
            ("disease", Pos::Noun),
            ("killed", Pos::Verb),
            ("cattle", Pos::Noun),
            ("today", Pos::Noun),
            ("is", Pos::Aux),
            ("bad", Pos::Adj),
        ];
        let doc = Document::from_tagged(&[s]);
        let got: Vec<_> = paper_candidates(&doc, &HashSet::new())
            .into_iter()
            .map(|c| c.surface)
            .collect();
        assert_eq!(got, vec!["mad cow disease", "cattle today"]);
    }
}
