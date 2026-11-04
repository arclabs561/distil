//! Stopword lists.
//!
//! The lists are NLTK's stopword corpus (`nltk_data/corpora/stopwords`), as
//! packaged by the `stop-words` crate with its `nltk` feature. Every
//! language uses the NLTK list rather than the larger stopwords-iso list
//! because NLTK's lists hold function words only. stopwords-iso merges
//! several sources; its English list has 1,298 words, including content
//! words such as "information", "system", "web" and "computer". On the
//! Inspec test set, 482 of the 3,283 gold keyphrases (counted per abstract)
//! contain one of its words and so can never be a stopword-delimited
//! candidate (NLTK's list: 68), even though it raises F1@10 for most
//! untagged methods by removing generic words from candidates. If another
//! crate in the build enables `stop-words`' `iso` feature, the NLTK list
//! still wins for every language NLTK covers.
//!
//! English, the default, has 198 words. It lacks a few modals and
//! contractions that other lists have ("could", "would", "that's"), so
//! those words can appear inside phrases from untagged text.
//!
//! The reference tools each ship their own English list: pke uses spaCy's
//! (326 words), sumy its own (580 words), and YAKE its own. For parity with
//! one of them, pass its list through a method's `with_stopwords`, as the
//! parity tests do.

use std::collections::HashSet;

/// NLTK's English stopwords, the default for every method.
pub fn default_stopwords() -> HashSet<String> {
    for_language("en").unwrap_or_default()
}

/// Stopwords for a language, from an ISO 639-1 code such as `"de"`.
///
/// English, German, French, Spanish, Portuguese, Italian, Dutch, Russian and
/// Arabic also accept their ISO 639-2 code (`"deu"`) or English name
/// (`"german"`). [`languages`] lists every supported code. Returns `None`
/// for other languages.
pub fn for_language(lang: &str) -> Option<HashSet<String>> {
    let lang = lang.to_lowercase();
    let code = match lang.as_str() {
        "eng" | "english" => "en",
        "deu" | "german" => "de",
        "fra" | "french" => "fr",
        "spa" | "spanish" => "es",
        "por" | "portuguese" => "pt",
        "ita" | "italian" => "it",
        "nld" | "dutch" => "nl",
        "rus" | "russian" => "ru",
        "ara" | "arabic" => "ar",
        other => other,
    };
    if !NLTK_LANGUAGES.contains(&code) {
        return None;
    }
    Some(
        stop_words::get(code)
            .iter()
            .map(|w| w.to_string())
            .collect(),
    )
}

/// The language codes [`for_language`] accepts, sorted.
pub fn languages() -> &'static [&'static str] {
    NLTK_LANGUAGES
}

/// The codes of `stop-words`' NLTK lists. `stop-words` 0.10.0 has no lookup
/// that returns `None` for an unknown code (its `get` panics), so the codes
/// are listed here.
const NLTK_LANGUAGES: &[&str] = &[
    "ar", "az", "be", "bn", "ca", "da", "de", "el", "en", "es", "eu", "fi", "fr", "he", "hinglish",
    "hu", "id", "it", "kk", "ne", "nl", "no", "pt", "ro", "ru", "sl", "sq", "sv", "ta", "tg", "tr",
    "uz", "zh",
];

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
            assert!(!stop_words::get(lang).is_empty(), "{lang}");
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
