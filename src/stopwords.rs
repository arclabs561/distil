//! Stopword lists.
//!
//! The lists are NLTK's stopword corpus (`nltk_data/corpora/stopwords`),
//! from [`textprep::stopwords`]; this module is a thin layer over it. Every
//! language uses the NLTK list rather than the larger stopwords-iso list
//! because NLTK's lists hold function words only. stopwords-iso merges
//! several sources; its English list has 1,298 words, including content
//! words such as "information", "system", "web" and "computer". On the
//! Inspec test set, 482 of the 3,283 gold keyphrases (counted per abstract)
//! contain one of its words and so can never be a stopword-delimited
//! candidate (NLTK's list: 68), even though it raises F1@10 for most
//! untagged methods by removing generic words from candidates.
//!
//! English, the default, has 198 words. It lacks a few modals and
//! contractions that other lists have ("could", "would", "that's"), so
//! those words can appear inside phrases from untagged text.
//!
//! # Roles
//!
//! Stopwords do two jobs in keyphrase extraction. In untagged text they
//! delimit candidates: a phrase ends at a stopword. They also decide which
//! words count in term weights and similarities (graph words, term
//! frequencies). Each job is served best by a different list: a larger
//! general list such as SMART (570 words) gives better candidate
//! boundaries, while weighting keeps the small function-word list. So in untagged English
//! text, candidates end at words of NLTK's list or SMART's
//! ([`textprep::stopwords::smart_english`]), and weighting uses NLTK's list
//! alone. Other languages use NLTK's list for both, and tagged text selects
//! candidates by part of speech with NLTK's list as the filter. Methods
//! that select candidates by stopwords take `with_delimiters` to replace
//! the extension: a domain list ("et", "al", "fig") added to SMART, or an
//! empty list for NLTK's list alone (`RakeExtractor::nltk`).
//!
//! On the Inspec test set, the SMART extension raised F1@10 (exact match)
//! for every method that delimits candidates: RAKE 0.235 to 0.290,
//! PositionRank 0.257 to 0.285, SGRank 0.239 to 0.299, TopicRank 0.215 to
//! 0.238 (`examples/eval_inspec.rs`).
//!
//! # Reference tools
//!
//! The reference tools each ship their own English list: pke uses spaCy's
//! (326 words), sumy its own (580 words), and YAKE its own (574). For parity
//! with one of them, pass its list through a method's `with_stopwords`, as
//! the parity tests do.

use std::collections::HashSet;

use crate::Language;

/// NLTK's English stopwords, the default for every method.
pub fn default_stopwords() -> HashSet<String> {
    Language::english().stopwords().clone()
}

/// Stopwords for a language, from a code or name such as `"de"`, `"deu"`,
/// `"german"` or `"de-AT"` (see [`textprep::stopwords::canonical_code`]).
/// [`languages`] lists every supported code. Returns `None` for other
/// languages. Words are lowercase, with `'` for U+2018 and U+2019.
pub fn for_language(lang: &str) -> Option<HashSet<String>> {
    Language::from_code(lang).map(|l| l.stopwords().clone())
}

/// The language codes [`for_language`] accepts, sorted: ISO 639-1 codes
/// plus NLTK's `"hinglish"`. The same as [`textprep::stopwords::LANGUAGES`].
pub fn languages() -> &'static [&'static str] {
    textprep::stopwords::LANGUAGES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_and_aliases() {
        for lang in ["en", "de", "fr", "es", "pt", "it", "nl", "ru", "ar"] {
            assert!(for_language(lang).is_some(), "{lang}");
        }
        assert_eq!(for_language("German"), for_language("de"));
        assert_eq!(for_language("eng"), Some(default_stopwords()));
        assert!(for_language("xx").is_none());
        assert!(languages().contains(&"sv"));
        assert_eq!(languages().len(), 33);
        assert!(languages().windows(2).all(|w| w[0] < w[1]));
        for lang in languages() {
            assert!(!for_language(lang).unwrap().is_empty(), "{lang}");
        }
    }

    #[test]
    fn english_is_nltk() {
        let en = default_stopwords();
        assert_eq!(en.len(), 198);
        for w in ["the", "of", "and", "don't", "should've"] {
            assert!(en.contains(w), "{w}");
        }
        assert!(!en.contains("would"));
    }
}
