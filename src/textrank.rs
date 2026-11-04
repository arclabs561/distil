//! TextRank keyphrases (Mihalcea and Tarau, 2004).
//!
//! Mihalcea, R. and Tarau, P. (2004). TextRank: Bringing Order into Text.
//! Proceedings of EMNLP 2004, pages 404-411. <https://aclanthology.org/W04-3252/>
//!
//! # Method
//!
//! 1. Nodes are the stems of nouns and adjectives (untagged text: words that
//!    are neither stopwords nor punctuation). Two nodes share an unweighted
//!    edge when their words occur within a window of `N = 2` tokens, i.e.
//!    next to each other.
//! 2. Node scores are PageRank with damping `d = 0.85`:
//!
//!    $$S(V_i) = (1 - d) / |V| + d \sum_{j \in In(V_i)} S(V_j) / |Out(V_j)|$$
//!
//! 3. The top `T` words are kept, `T` a third of the number of nodes (the
//!    paper's setting for abstracts).
//! 4. Kept words are marked in the text and runs of adjacent marked words
//!    become phrases. A phrase scores the sum of its words' scores (the
//!    paper does not score phrases; this is pke's rule).
//!
//! # Deviations
//!
//! - Following pke, the window is measured over all tokens, so two nouns
//!   separated by a stopword are not linked, and sentence boundaries are
//!   ignored.
//! - The paper stops iterating at a change below 0.0001; this runs to 1e-10.
//! - Words tied at the top-`T` cutoff (scores within 1e-12) are kept in
//!   order of first occurrence, which is what pke's stable sort does.
//! - Phrase ties are broken by first occurrence (pke adds `1e-8 * offset`,
//!   which favours later candidates).

use crate::graph::{graph_words, rank_by_word_sum, WordGraph};
use crate::text::{longest_sequences, Document, Roles};
use crate::KeywordExtractor;
use std::collections::{HashMap, HashSet};

/// TextRank keyphrase extractor with phrase merging.
#[derive(Debug, Clone)]
pub struct TextRank {
    stopwords: Roles,
    window: usize,
    top_fraction: f64,
}

impl Default for TextRank {
    fn default() -> Self {
        Self::new()
    }
}

impl TextRank {
    /// Window 2, top third of the words, English stopwords for untagged text.
    pub fn new() -> Self {
        Self {
            stopwords: Roles::default(),
            window: 2,
            top_fraction: 1.0 / 3.0,
        }
    }

    /// Stopwords used to pick graph words in untagged text.
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

    /// Co-occurrence window in tokens (at least 2).
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(2);
        self
    }

    /// Fraction of graph words kept as keywords, clamped to `[0, 1]`.
    pub fn with_top_fraction(mut self, fraction: f64) -> Self {
        self.top_fraction = if fraction.is_nan() {
            1.0 / 3.0
        } else {
            fraction.clamp(0.0, 1.0)
        };
        self
    }

    /// Score of each graph word (stem).
    pub fn word_scores(&self, doc: &Document) -> HashMap<String, f64> {
        let mut g = WordGraph::default();
        g.add_document(
            doc,
            &graph_words(self.stopwords.language()),
            self.window,
            1.0,
        );
        g.unweighted().scores(None)
    }

    /// Up to `max` keyphrases, best first.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        let scores = self.word_scores(doc);
        let mut first: HashMap<&str, usize> = HashMap::new();
        for (i, t) in doc.tokens().enumerate() {
            first.entry(t.stem.as_str()).or_insert(i);
        }
        // Words in symmetric parts of the graph tie exactly in theory but
        // differ in the last bits; scores equal to 1e-12 count as tied, and
        // ties go to the word that occurs first (pke's order).
        let mut words: Vec<(&String, f64)> = scores.iter().map(|(w, &s)| (w, s)).collect();
        words.sort_by_key(|&(w, s)| {
            (
                std::cmp::Reverse((s * 1e12).round() as i64),
                first[w.as_str()],
            )
        });
        // The small epsilon keeps `n * (1/3)` from flooring to one below an
        // integer.
        let keep = ((words.len() as f64 * self.top_fraction) + 1e-9).floor() as usize;
        let top: HashSet<&str> = words[..keep].iter().map(|(w, _)| w.as_str()).collect();
        let candidates = longest_sequences(doc, |t| top.contains(t.stem.as_str()));
        rank_by_word_sum(&candidates, &scores, max)
    }
}

impl KeywordExtractor for TextRank {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.rank(&self.stopwords.document(text), max_keywords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::Pos;

    #[test]
    fn adjacent_keywords_merge_into_phrases() {
        // With every graph word kept, phrases are the maximal runs of nouns
        // and adjectives.
        let s: Vec<(&str, Pos)> = vec![
            ("Neural", Pos::Adj),
            ("networks", Pos::Noun),
            ("need", Pos::Verb),
            ("training", Pos::Noun),
            ("data", Pos::Noun),
            (".", Pos::Punct),
        ];
        let doc = Document::from_tagged(&[s]);
        let got = TextRank::new().with_top_fraction(1.0).rank(&doc, 10);
        let phrases: Vec<&str> = got.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(phrases.len(), 2, "{got:?}");
        assert!(phrases.contains(&"neural networks") && phrases.contains(&"training data"));
    }

    #[test]
    fn keeps_a_third_of_the_words() {
        // Six graph words, so two are keywords.
        let doc = Document::from_text("alpha beta gamma delta epsilon zeta");
        let tr = TextRank::new();
        let kept: usize = tr
            .rank(&doc, 100)
            .iter()
            .map(|(p, _)| p.split(' ').count())
            .sum();
        assert_eq!(kept, 2);
    }

    #[test]
    fn empty_input() {
        assert!(TextRank::new().extract("", 5).is_empty());
        assert!(TextRank::new().extract("the of and", 5).is_empty());
    }
}
