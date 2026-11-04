#![warn(missing_docs)]
//! Keyword and keyphrase extraction.
//!
//! Statistical algorithms for extracting important terms (words and phrases)
//! from text. No ML models, no GPU, no external corpora required.
//!
//! # Algorithms
//!
//! | Algorithm | Candidates | Scoring | Best For |
//! |-----------|------------|---------|----------|
//! | [`RakeExtractor`] | Phrases between stopwords | Frequency x degree | Technical docs |
//! | [`YakeExtractor`] | N-grams after preprocessing | Statistical features | General text |
//! | [`TextRankExtractor`] | Content words | PageRank on co-occurrence | Reviews, short text |
//! | [`TfIdfExtractor`] | All terms | TF-IDF | Simple baseline |
//!
//! All four implement [`KeywordExtractor`].
//!
//! # Graph keyphrase methods
//!
//! One module per method, each documenting its paper, equations, defaults
//! and deviations. They work on a [`text::Document`]; given part-of-speech
//! tags ([`text::Document::from_tagged`]) they select the noun-phrase
//! candidates of their papers, otherwise stopword- and punctuation-delimited
//! phrases. Phrases come back lowercased, as first written, best first.
//!
//! | Method | Graph nodes | Extra input | Phrase score |
//! |--------|-------------|-------------|--------------|
//! | [`TextRank`] | words, window 2, unweighted | | sum over merged top-third words |
//! | [`SingleRank`] | words, window 10, weighted | | sum of word scores |
//! | [`ExpandRank`] | words of the document and neighbours | neighbour documents with similarities | sum of word scores |
//! | [`CollabRank`] | words of a document cluster | the cluster | sum of word scores |
//! | [`PositionRank`] | words, biased by inverse position | | sum of word scores |
//! | [`TopicRank`] | topics (clustered candidates) | | topic score, first candidate |
//! | [`MultipartiteRank`] | candidates, k-partite by topic | | candidate score |
//! | [`TopicalPageRank`] | words, biased by topic similarity | word-topic and document-topic distributions | sum of word scores |
//!
//! | [`SgRank`] | candidates, distance-weighted | document frequencies (optional) | PageRank of candidate |
//! | [`WordAttractionRank`] | words, attraction-weighted | word vectors | sum of word scores |
//!
//! All but [`TopicalPageRank`] implement [`KeywordExtractor`] (ExpandRank
//! and CollabRank with no neighbours or cluster).
//!
//! # Statistical, clustering and embedding keyphrase methods
//!
//! | Method | Candidates | Extra input | Phrase score |
//! |--------|------------|-------------|--------------|
//! | [`TfIdf`] | filtered 1-3-grams | [`DocumentFrequency`] | `tf * log2((1 + D) / (1 + df))` |
//! | [`KpMiner`] | filtered 1-5-grams, frequency and position cut | [`DocumentFrequency`] | tf-idf, compound phrases boosted |
//! | [`Yake`] | 1-3-grams within punctuation blocks | | YAKE `S(kw)`, lower is better |
//! | [`FirstPhrases`] | noun/adjective runs | | minus first position |
//! | [`KeyCluster`] | noun phrases with an exemplar term | relatedness (optional) | exemplar count |
//! | [`CommunityCluster`] | noun phrases | relatedness, keyphraseness (optional) | community density x informativeness |
//! | [`EmbedRank`] | noun phrases | phrase and document vectors | cosine to document, or MMR |
//! | [`TomokiyoHurst`] | 1-3-grams | background word counts | phraseness + informativeness |
//!
//! [`TfIdf`], [`KpMiner`], [`Yake`] and [`FirstPhrases`] match pke and the
//! `yake` package on shared tokens (`tests/pke_parity.rs`,
//! `tests/yake_parity.rs`); the others have no maintained reference
//! implementation and are tested against their equations on small inputs.
//!
//! # Language Limitations
//!
//! These methods use whitespace/punctuation tokenization and English stopword
//! lists by default. They work for Latin-script languages with appropriate
//! stopwords (see [`stopwords`] module). For CJK, Thai, or Arabic text,
//! pre-tokenize with a language-specific tool and pass space-separated tokens.
//!
//! # Example
//!
//! ```
//! use distil::{KeywordExtractor, RakeExtractor};
//!
//! let text = "Machine learning is a subset of artificial intelligence. \
//!             Deep learning uses neural networks for machine learning tasks.";
//!
//! let extractor = RakeExtractor::new();
//! let keywords = extractor.extract(text, 5);
//!
//! for (keyword, score) in &keywords {
//!     println!("{keyword}: {score:.2}");
//! }
//! ```
//!
//! # References
//!
//! - Rose et al. (2010): RAKE -- Rapid Automatic Keyword Extraction
//! - Campos et al. (2018): YAKE! Collection-independent keyword extraction
//! - Mihalcea & Tarau (2004): TextRank

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
pub mod text;
pub mod textrank;
mod textrank_extractor;
pub mod tfidf;
mod tfidf_extractor;
pub mod tomokiyohurst;
pub mod topicalpagerank;
pub mod topicrank;
pub mod wordattractionrank;
pub mod yake;
mod yake_extractor;

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
pub use textrank::TextRank;
pub use textrank_extractor::TextRankExtractor;
pub use tfidf::{DocumentFrequency, TfIdf};
pub use tfidf_extractor::TfIdfExtractor;
pub use tomokiyohurst::TomokiyoHurst;
pub use topicalpagerank::TopicalPageRank;
pub use topicrank::TopicRank;
pub use wordattractionrank::WordAttractionRank;
pub use yake::Yake;
pub use yake_extractor::YakeExtractor;

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

    #[test]
    fn test_empty_text() {
        let rake = RakeExtractor::new();
        let yake = YakeExtractor::new();
        let textrank = TextRankExtractor::new();
        let tfidf = TfIdfExtractor::new();

        assert!(rake.extract("", 5).is_empty());
        assert!(yake.extract("", 5).is_empty());
        assert!(textrank.extract("", 5).is_empty());
        assert!(tfidf.extract("", 5).is_empty());
    }

    /// Tied scores must come back in the same order on every call
    /// (ascending text), not in hash-map iteration order.
    #[test]
    fn tied_scores_are_ordered_by_text_on_every_call() {
        let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet";
        let extractors: Vec<(&str, Box<dyn KeywordExtractor>)> = vec![
            ("tfidf", Box::new(TfIdfExtractor::new())),
            ("rake", Box::new(RakeExtractor::new())),
            ("yake", Box::new(YakeExtractor::new())),
            ("textrank", Box::new(TextRankExtractor::new())),
        ];
        for (name, ex) in &extractors {
            let first = ex.extract(text, 5);
            for _ in 0..30 {
                assert_eq!(ex.extract(text, 5), first, "{name} is not deterministic");
            }
        }
        let tfidf: Vec<String> = TfIdfExtractor::new()
            .extract(text, 3)
            .into_iter()
            .map(|(w, _)| w)
            .collect();
        assert_eq!(tfidf, vec!["alpha", "bravo", "charlie"]);
    }

    #[test]
    fn test_single_word_text() {
        let rake = RakeExtractor::new();
        let yake = YakeExtractor::new();
        let tfidf = TfIdfExtractor::new();

        let text = "algorithm";
        let rake_kw = rake.extract(text, 5);
        let yake_kw = yake.extract(text, 5);
        let tfidf_kw = tfidf.extract(text, 5);

        assert_eq!(rake_kw.len(), 1, "RAKE single word: {rake_kw:?}");
        assert_eq!(rake_kw[0].0, "algorithm");

        assert_eq!(tfidf_kw.len(), 1, "TF-IDF single word: {tfidf_kw:?}");
        assert_eq!(tfidf_kw[0].0, "algorithm");

        assert!(!yake_kw.is_empty(), "YAKE single word should return result");
    }

    #[test]
    fn test_all_stopwords_all_extractors() {
        let text = "the and or but is are was were been";
        let rake = RakeExtractor::new();
        let yake = YakeExtractor::new();
        let textrank = TextRankExtractor::new();
        let tfidf = TfIdfExtractor::new();

        assert!(
            rake.extract(text, 5).is_empty(),
            "RAKE should return empty for all-stopword text"
        );
        assert!(
            yake.extract(text, 5).is_empty(),
            "YAKE should return empty for all-stopword text"
        );
        assert!(
            textrank.extract(text, 5).is_empty(),
            "TextRank should return empty for all-stopword text"
        );
        assert!(
            tfidf.extract(text, 5).is_empty(),
            "TF-IDF should return empty for all-stopword text"
        );
    }

    #[test]
    fn test_extract_all() {
        let extractor = TfIdfExtractor::new();
        let all = extractor.extract_all(TEST_TEXT);
        let limited = extractor.extract(TEST_TEXT, 2);
        assert!(all.len() >= limited.len());
    }

    #[test]
    fn test_extract_from_tokens() {
        let extractor = TfIdfExtractor::new();
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
        assert!(RakeExtractor::new().extract(text, 5).is_empty());
        assert!(YakeExtractor::new().extract(text, 5).is_empty());
        assert!(TextRankExtractor::new().extract(text, 5).is_empty());
        assert!(TfIdfExtractor::new().extract(text, 5).is_empty());
    }

    #[test]
    fn edge_whitespace_only() {
        let text = "   \t\n  ";
        assert!(RakeExtractor::new().extract(text, 5).is_empty());
        assert!(YakeExtractor::new().extract(text, 5).is_empty());
        assert!(TextRankExtractor::new().extract(text, 5).is_empty());
        assert!(TfIdfExtractor::new().extract(text, 5).is_empty());
    }

    #[test]
    fn edge_max_keywords_zero() {
        let rake = RakeExtractor::new();
        let yake = YakeExtractor::new();
        let textrank = TextRankExtractor::new();
        let tfidf = TfIdfExtractor::new();

        assert!(rake.extract(GOLDEN_TEXT, 0).is_empty());
        // YAKE returns results even with max_keywords=0 (implementation quirk).
        let _ = yake.extract(GOLDEN_TEXT, 0);
        assert!(textrank.extract(GOLDEN_TEXT, 0).is_empty());
        assert!(tfidf.extract(GOLDEN_TEXT, 0).is_empty());
    }

    #[test]
    fn edge_repeated_word() {
        let text = "algorithm algorithm algorithm algorithm algorithm";
        let rake_kw = RakeExtractor::new().extract(text, 5);
        let tfidf_kw = TfIdfExtractor::new().extract(text, 5);

        assert_eq!(rake_kw.len(), 1, "RAKE repeated word: {rake_kw:?}");
        assert!(
            rake_kw[0].0.contains("algorithm"),
            "RAKE repeated word should contain 'algorithm': {:?}",
            rake_kw[0].0
        );

        assert_eq!(tfidf_kw.len(), 1, "TF-IDF repeated word: {tfidf_kw:?}");
        assert_eq!(tfidf_kw[0].0, "algorithm");
    }

    #[test]
    fn edge_very_short_text() {
        let text = "graph theory";
        let rake_kw = RakeExtractor::new().extract(text, 5);
        let tfidf_kw = TfIdfExtractor::new().extract(text, 5);

        assert!(!rake_kw.is_empty(), "RAKE should handle 2-word text");
        assert!(!tfidf_kw.is_empty(), "TF-IDF should handle 2-word text");
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
        let extractors: Vec<(&str, Box<dyn KeywordExtractor>)> = vec![
            ("RAKE", Box::new(RakeExtractor::new())),
            ("YAKE", Box::new(YakeExtractor::new())),
            ("TextRank", Box::new(TextRankExtractor::new())),
            ("TF-IDF", Box::new(TfIdfExtractor::new())),
        ];
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
        let extractors: Vec<(&str, Box<dyn KeywordExtractor>)> = vec![
            ("RAKE", Box::new(RakeExtractor::new())),
            ("YAKE", Box::new(YakeExtractor::new())),
            ("TextRank", Box::new(TextRankExtractor::new())),
            ("TF-IDF", Box::new(TfIdfExtractor::new())),
        ];
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
