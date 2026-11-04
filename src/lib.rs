#![warn(missing_docs)]
//! Keyword and keyphrase extraction.
//!
//! distil implements the main unsupervised keyphrase-extraction methods:
//! graph-based ranking, statistical scoring, topic clustering, embedding
//! similarity and language-model scoring. Each method lives in its own module,
//! whose documentation gives the paper, the equations used, the defaults and
//! any deviation. Where a maintained reference implementation exists (pke,
//! the `yake` package), tests check that distil returns the same candidates
//! and scores on the same tokens.
//!
//! # Example
//!
//! ```
//! use distil::{KeywordExtractor, PositionRank};
//!
//! let text = "Keyphrase extraction finds the phrases that describe a document. \
//!             Graph-based keyphrase extraction ranks words with PageRank.";
//!
//! for (phrase, score) in PositionRank::new().extract(text, 3) {
//!     println!("{score:.3}  {phrase}");
//! }
//! ```
//!
//! Every method that needs only the text implements [`KeywordExtractor`].
//! Methods that need more (document frequencies, neighbour documents, topic
//! distributions, vectors or a background corpus) take it in their own
//! `rank` functions.
//!
//! # Methods
//!
//! | Method | Family | Extra input | Checked against |
//! |--------|--------|-------------|-----------------|
//! | [`TextRank`] | graph | | pke |
//! | [`SingleRank`] | graph | | pke |
//! | [`PositionRank`] | graph | | pke |
//! | [`TopicRank`] | graph over topics | | pke |
//! | [`MultipartiteRank`] | graph over topics | | pke |
//! | [`TopicalPageRank`] | graph | word-topic and document-topic distributions | pke |
//! | [`ExpandRank`] | graph | neighbour documents with similarities | identities with SingleRank |
//! | [`CollabRank`] | graph | a cluster of documents | identities with ExpandRank |
//! | [`SgRank`] | graph over candidates | document frequencies (optional) | paper equations |
//! | [`WordAttractionRank`] | graph | word vectors | paper equations |
//! | [`TfIdf`] | statistical | [`DocumentFrequency`] | pke |
//! | [`KpMiner`] | statistical | [`DocumentFrequency`] | pke |
//! | [`Yake`] | statistical | | `yake` package |
//! | [`RakeExtractor`] | statistical | | unit tests |
//! | [`FirstPhrases`] | baseline | | pke |
//! | [`KeyCluster`] | clustering | relatedness (optional) | paper equations |
//! | [`CommunityCluster`] | clustering | relatedness (optional) | paper equations |
//! | [`EmbedRank`] | embedding | phrase and document vectors | paper equations |
//! | [`TomokiyoHurst`] | language model | background word counts | paper equations |
//!
//! [`KpMiner`] and [`SgRank`] build on the same TF-IDF weights as [`TfIdf`]
//! ([`DocumentFrequency`]); the other methods do not use corpus statistics.
//! The README has F1 scores for each method on the Inspec test set.
//!
//! # Candidates
//!
//! Methods work on a [`text::Document`]. Given part-of-speech tags
//! ([`text::Document::from_tagged`]) they select the noun-phrase candidates
//! their papers use; from plain text they fall back to phrases delimited by
//! stopwords and punctuation. Phrases come back lowercased, best first.
//!
//! # Languages
//!
//! Tokenization splits on whitespace and punctuation, and English stopwords
//! are the default. [`stopwords::for_language`] has lists for German, French,
//! Spanish, Portuguese, Italian, Dutch, Russian and Arabic. Pre-tokenize CJK
//! or Thai text and pass space-separated tokens.

pub mod collabrank;
pub mod communitycluster;
pub mod embedrank;
pub mod expandrank;
pub mod firstphrases;
mod graph;
pub mod keycluster;
pub mod kpminer;
pub mod multipartiterank;
pub mod positionrank;
mod rake_extractor;
pub mod sgrank;
pub mod singlerank;
pub mod stopwords;
pub mod summarize;
pub mod text;
pub mod textrank;
pub mod tfidf;
pub mod tomokiyohurst;
pub mod topicalpagerank;
pub mod topicrank;
pub mod wordattractionrank;
pub mod yake;

pub use collabrank::CollabRank;
pub use communitycluster::CommunityCluster;
pub use embedrank::EmbedRank;
pub use expandrank::ExpandRank;
pub use firstphrases::FirstPhrases;
pub use keycluster::KeyCluster;
pub use kpminer::KpMiner;
pub use multipartiterank::MultipartiteRank;
pub use positionrank::PositionRank;
pub use rake_extractor::RakeExtractor;
pub use sgrank::SgRank;
pub use singlerank::SingleRank;
#[doc(inline)]
pub use stopwords::{default_stopwords, STOPWORDS};
pub use summarize::{Budget, SentenceExtractor};
pub use textrank::TextRank;
pub use tfidf::{DocumentFrequency, TfIdf};
pub use tomokiyohurst::TomokiyoHurst;
pub use topicalpagerank::TopicalPageRank;
pub use topicrank::TopicRank;
pub use wordattractionrank::WordAttractionRank;
pub use yake::Yake;

// Compile and run the README's Rust examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

// =============================================================================
// Common Trait
// =============================================================================

/// Trait for keyword extraction algorithms.
///
/// Most implementations assume whitespace-delimited text. For languages
/// without whitespace word boundaries, pre-tokenize the text with an
/// appropriate tokenizer before calling these methods.
pub trait KeywordExtractor: Send + Sync {
    /// Extract keywords from text.
    ///
    /// Returns keywords paired with scores, sorted descending by score.
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)>;

    /// Extract all keywords without limit.
    fn extract_all(&self, text: &str) -> Vec<(String, f64)> {
        self.extract(text, usize::MAX)
    }

    /// Extract keywords from pre-tokenized text.
    ///
    /// Joins tokens with spaces, then delegates to [`extract`](KeywordExtractor::extract).
    /// Use this for languages that require specialized tokenization.
    fn extract_from_tokens(&self, tokens: &[&str], max_keywords: usize) -> Vec<(String, f64)> {
        let text = tokens.join(" ");
        self.extract(&text, max_keywords)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const TEST_TEXT: &str = "Machine learning is a subset of artificial intelligence. \
        Deep learning uses neural networks for machine learning tasks. \
        Neural networks are inspired by biological neural networks.";

    /// Every type that implements [`KeywordExtractor`], with defaults.
    pub(crate) fn all_extractors() -> Vec<(&'static str, Box<dyn KeywordExtractor>)> {
        vec![
            ("RAKE", Box::new(RakeExtractor::new())),
            ("YAKE", Box::new(Yake::new())),
            ("TextRank", Box::new(TextRank::new())),
            ("SingleRank", Box::new(SingleRank::new())),
            ("PositionRank", Box::new(PositionRank::new())),
            ("TopicRank", Box::new(TopicRank::new())),
            ("MultipartiteRank", Box::new(MultipartiteRank::new())),
            ("ExpandRank", Box::new(ExpandRank::new())),
            ("CollabRank", Box::new(CollabRank::new())),
            ("SGRank", Box::new(SgRank::new())),
            ("FirstPhrases", Box::new(FirstPhrases::new())),
        ]
    }

    #[test]
    fn test_empty_text() {
        for (name, ex) in all_extractors() {
            assert!(ex.extract("", 5).is_empty(), "{name}");
        }
    }

    /// Tied scores must come back in the same order on every call, not in
    /// hash-map iteration order.
    #[test]
    fn tied_scores_are_ordered_the_same_on_every_call() {
        let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet";
        for (name, ex) in all_extractors() {
            let first = ex.extract(text, 5);
            for _ in 0..30 {
                assert_eq!(ex.extract(text, 5), first, "{name} is not deterministic");
            }
        }
    }

    #[test]
    fn test_single_word_text() {
        let rake_kw = RakeExtractor::new().extract("algorithm", 5);
        assert_eq!(rake_kw.len(), 1, "RAKE single word: {rake_kw:?}");
        assert_eq!(rake_kw[0].0, "algorithm");
        assert!(!Yake::new().extract("algorithm", 5).is_empty());
    }

    #[test]
    fn test_all_stopwords_all_extractors() {
        let text = "the and or but is are was were been";
        for (name, ex) in all_extractors() {
            assert!(
                ex.extract(text, 5).is_empty(),
                "{name} on all-stopword text"
            );
        }
    }

    #[test]
    fn test_extract_all() {
        let extractor = RakeExtractor::new();
        let all = extractor.extract_all(TEST_TEXT);
        let limited = extractor.extract(TEST_TEXT, 2);
        assert!(all.len() >= limited.len());
    }

    #[test]
    fn test_extract_from_tokens() {
        let extractor = RakeExtractor::new();
        let tokens = &["algorithm", "data", "algorithm", "structure"];
        let keywords = extractor.extract_from_tokens(tokens, 5);
        assert!(!keywords.is_empty());
    }

    // =========================================================================
    // Golden example tests
    // =========================================================================

    pub(crate) const GOLDEN_TEXT: &str =
        "Machine learning is a subset of artificial intelligence. \
        Deep learning uses neural networks for machine learning tasks. \
        Neural networks are inspired by biological neural networks.";

    /// Helper: collect keyword strings as lowercase for assertions.
    pub(crate) fn keyword_strings(results: &[(String, f64)]) -> Vec<String> {
        results.iter().map(|(k, _)| k.to_lowercase()).collect()
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn edge_punctuation_only() {
        let text = "... !!! ???  ---";
        for (name, ex) in all_extractors() {
            assert!(ex.extract(text, 5).is_empty(), "{name}");
        }
    }

    #[test]
    fn edge_whitespace_only() {
        let text = "   \t\n  ";
        for (name, ex) in all_extractors() {
            assert!(ex.extract(text, 5).is_empty(), "{name}");
        }
    }

    #[test]
    fn edge_max_keywords_zero() {
        for (name, ex) in all_extractors() {
            assert!(ex.extract(GOLDEN_TEXT, 0).is_empty(), "{name}");
        }
    }

    #[test]
    fn edge_repeated_word() {
        let text = "algorithm algorithm algorithm algorithm algorithm";
        let rake_kw = RakeExtractor::new().extract(text, 5);
        assert_eq!(rake_kw.len(), 1, "RAKE repeated word: {rake_kw:?}");
        assert!(
            rake_kw[0].0.contains("algorithm"),
            "RAKE repeated word should contain 'algorithm': {:?}",
            rake_kw[0].0
        );
    }

    #[test]
    fn edge_very_short_text() {
        let text = "graph theory";
        // TextRank keeps the top third of the words (as in the paper and pke),
        // which rounds down to none for a two-word text.
        for (name, ex) in all_extractors() {
            if name == "TextRank" {
                continue;
            }
            assert!(!ex.extract(text, 5).is_empty(), "{name} on 2-word text");
        }
    }

    // =========================================================================
    // Property tests -- scores are finite, positive, keywords non-empty
    // =========================================================================

    /// Assert basic score properties for any extractor output.
    pub(crate) fn assert_score_properties(name: &str, results: &[(String, f64)]) {
        for (kw, score) in results {
            assert!(!kw.is_empty(), "{name}: keyword string must be non-empty");
            assert!(
                kw.trim() == kw,
                "{name}: keyword '{kw}' has leading/trailing whitespace"
            );
            assert!(
                score.is_finite(),
                "{name}: score for '{kw}' is not finite: {score}"
            );
            assert!(
                *score >= 0.0,
                "{name}: score for '{kw}' is negative: {score}"
            );
        }
    }

    #[test]
    fn property_all_extractors_descending_order() {
        let extractors = all_extractors();
        for (name, ext) in &extractors {
            let kw = ext.extract(GOLDEN_TEXT, 10);
            for pair in kw.windows(2) {
                assert!(
                    pair[0].1 >= pair[1].1,
                    "{name} scores not descending: {} ({}) vs {} ({})",
                    pair[0].0,
                    pair[0].1,
                    pair[1].0,
                    pair[1].1,
                );
            }
        }
    }

    #[test]
    fn property_max_keywords_respected() {
        let extractors = all_extractors();
        for (name, ext) in &extractors {
            let kw = ext.extract(GOLDEN_TEXT, 3);
            assert!(
                kw.len() <= 3,
                "{name} returned {} keywords, expected at most 3",
                kw.len()
            );
        }
    }
}
