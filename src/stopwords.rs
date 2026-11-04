//! Stopword lists for various languages.
//!
//! These are minimal lists covering common function words. For production use
//! with non-English text, consider more comprehensive sources.

use std::collections::HashSet;

/// Default English stopwords for keyword extraction.
pub const STOPWORDS: &[&str] = &[
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
];

/// Create a stopword set from the default English list.
///
/// For other languages, use [`for_language`] or construct your own set.
pub fn default_stopwords() -> HashSet<String> {
    STOPWORDS.iter().map(|s| s.to_string()).collect()
}

/// German stopwords (common function words).
pub fn german() -> HashSet<String> {
    [
        "der", "die", "das", "und", "in", "zu", "den", "ist", "nicht", "von", "sie", "mit", "auf",
        "es", "ein", "eine", "dem", "für", "sich", "an", "als", "auch", "er", "hat", "aus", "bei",
        "war", "so", "werden", "ich", "ihr", "wir", "aber", "wie", "nur", "oder", "nach", "noch",
        "kann", "über",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// French stopwords.
pub fn french() -> HashSet<String> {
    [
        "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "à", "au", "aux", "que",
        "qui", "ne", "pas", "pour", "sur", "ce", "cette", "il", "elle", "nous", "vous", "ils",
        "elles", "son", "sa", "ses", "leur", "leurs", "mais", "ou", "donc", "car", "avec", "dans",
        "par",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Spanish stopwords.
pub fn spanish() -> HashSet<String> {
    [
        "el", "la", "los", "las", "de", "del", "en", "y", "a", "que", "es", "un", "una", "por",
        "con", "no", "para", "se", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o",
        "este", "si", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Portuguese stopwords.
pub fn portuguese() -> HashSet<String> {
    [
        "o", "a", "os", "as", "de", "da", "do", "das", "dos", "em", "um", "uma", "e", "é", "que",
        "no", "na", "nos", "nas", "por", "para", "com", "não", "se", "mais", "como", "mas", "ao",
        "ele", "ela", "seu", "sua", "ou", "ser", "quando", "muito", "há", "foi", "são",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Italian stopwords.
pub fn italian() -> HashSet<String> {
    [
        "il", "lo", "la", "i", "gli", "le", "di", "a", "da", "in", "con", "su", "per", "tra",
        "fra", "un", "uno", "una", "e", "che", "non", "è", "si", "come", "più", "ma", "o", "anche",
        "questo", "quello", "essere", "sono", "sono", "suo", "sua", "loro", "chi", "cui", "dove",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Dutch stopwords.
pub fn dutch() -> HashSet<String> {
    [
        "de", "het", "een", "van", "en", "in", "is", "op", "te", "dat", "die", "voor", "zijn",
        "met", "niet", "aan", "om", "ook", "als", "dan", "maar", "of", "door", "over", "bij",
        "uit", "naar", "nog", "wel", "kan", "meer", "was", "worden", "tot", "er", "al", "worden",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Russian stopwords (Cyrillic).
pub fn russian() -> HashSet<String> {
    [
        "и",
        "в",
        "не",
        "на",
        "я",
        "с",
        "он",
        "что",
        "это",
        "по",
        "но",
        "они",
        "к",
        "у",
        "же",
        "вы",
        "за",
        "бы",
        "так",
        "от",
        "все",
        "как",
        "она",
        "его",
        "только",
        "или",
        "мы",
        "ещё",
        "из",
        "для",
        "если",
        "уже",
        "при",
        "их",
        "во",
        "когда",
        "до",
        "ни",
        "чтобы",
        "да",
        "был",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Arabic stopwords.
pub fn arabic() -> HashSet<String> {
    [
        "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "التي", "الذي", "أن", "كان", "قد",
        "ما", "لم", "لا", "و", "أو", "ثم", "بين", "كل", "بعد", "قبل", "حتى", "إذا", "هو", "هي",
        "هم", "أنت", "نحن",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Get stopwords for a language by ISO 639-1 code.
///
/// Returns `None` if the language is not supported.
pub fn for_language(lang: &str) -> Option<HashSet<String>> {
    match lang.to_lowercase().as_str() {
        "en" | "eng" | "english" => Some(default_stopwords()),
        "de" | "deu" | "german" => Some(german()),
        "fr" | "fra" | "french" => Some(french()),
        "es" | "spa" | "spanish" => Some(spanish()),
        "pt" | "por" | "portuguese" => Some(portuguese()),
        "it" | "ita" | "italian" => Some(italian()),
        "nl" | "nld" | "dutch" => Some(dutch()),
        "ru" | "rus" | "russian" => Some(russian()),
        "ar" | "ara" | "arabic" => Some(arabic()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stopwords_for_language() {
        assert!(for_language("en").is_some());
        assert!(for_language("de").is_some());
        assert!(for_language("fr").is_some());
        assert!(for_language("es").is_some());
        assert!(for_language("pt").is_some());
        assert!(for_language("it").is_some());
        assert!(for_language("nl").is_some());
        assert!(for_language("ru").is_some());
        assert!(for_language("ar").is_some());
        assert!(for_language("xx").is_none());
    }
}
