//! RAKE keyword extractor ([`RakeExtractor`]).

use std::collections::{HashMap, HashSet};

use crate::text::{Document, Roles, Stoplist};
use crate::{KeywordExtractor, Language};

/// RAKE keyword extractor.
///
/// Identifies candidate keywords as sequences of words between stopwords
/// or punctuation, then scores them using word frequency and degree (co-occurrence).
///
/// # Algorithm
///
/// 1. Split text on stopwords and punctuation into candidate phrases
/// 2. For each word, compute `freq(w)` and `deg(w)` (sum of phrase lengths containing w)
/// 3. Word score = `deg(w) / freq(w)`
/// 4. Phrase score = sum of word scores
///
/// # Reference
///
/// Rose, S., Engel, D., Cramer, N., & Cowley, W. (2010).
/// "Automatic Keyword Extraction from Individual Documents"
#[derive(Debug, Clone)]
pub struct RakeExtractor {
    stopwords: Roles,
    min_word_length: usize,
    /// Minimum number of words in a candidate phrase.
    pub min_phrase_length: usize,
    max_phrase_length: usize,
}

impl Default for RakeExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl RakeExtractor {
    /// Create a new RAKE extractor with English stopwords: NLTK's list
    /// plus SMART (570 words) as delimiters, the list the RAKE paper's
    /// widely used implementation splits on.
    pub fn new() -> Self {
        Self {
            stopwords: Roles::default(),
            min_word_length: 1,
            min_phrase_length: 1,
            max_phrase_length: 5,
        }
    }

    /// rake-nltk's setup: NLTK's English list (and punctuation) as the only
    /// delimiters, without the SMART extension that [`RakeExtractor::new`]
    /// adds for English. For parity with rake-nltk.
    pub fn nltk() -> Self {
        Self::new().with_delimiters([""; 0])
    }

    /// Set custom stopwords.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords.set_stopwords(stopwords);
        self
    }

    /// Stopword language: its list replaces the current one. Sentences of a
    /// document in another language use that language's stopwords
    /// ([`Stoplist`]).
    pub fn with_language(mut self, language: crate::Language) -> Self {
        self.stopwords.set_language(language);
        self
    }

    /// Replace the extra words that end candidate phrases, on top of the
    /// stopwords. The default is the SMART list (570 words) in English and
    /// nothing in other languages; [`RakeExtractor::nltk`] uses none. A list
    /// given here applies to sentences in the method's language. RAKE uses
    /// stopwords only as delimiters, so this is the same as adding the
    /// words to the list.
    pub fn with_delimiters<I, S>(mut self, words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.stopwords.set_delimiters(words);
        self
    }

    /// Set minimum word length.
    pub fn with_min_word_length(mut self, len: usize) -> Self {
        self.min_word_length = len;
        self
    }

    /// Set maximum phrase length (in words).
    pub fn with_max_phrase_length(mut self, len: usize) -> Self {
        self.max_phrase_length = len;
        self
    }

    /// Up to `max` keyphrases of a document, best first. Each sentence is
    /// split with the stopwords of its language, and phrases end at
    /// sentence boundaries. `extract` instead reads the text as one run of
    /// words in the method's language.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        let mut words = Vec::new();
        for (sentence, language) in doc.sentences_in() {
            words.extend(sentence.iter().map(|t| Some((t.word.as_str(), language))));
            words.push(None);
        }
        self.score(&self.candidates(words), max)
    }

    /// Candidate phrases of `text`, whose words are those of
    /// `text::word_ranges` (UAX #29), all in the method's language.
    fn extract_candidates(&self, text: &str) -> Vec<Vec<String>> {
        let words = crate::text::word_ranges(text)
            .into_iter()
            .map(|r| Some((&text[r], None)));
        self.candidates(words)
    }

    /// Extract candidate phrases by splitting on stopwords and punctuation.
    /// `None` ends a phrase.
    fn candidates<'a>(
        &self,
        words: impl IntoIterator<Item = Option<(&'a str, Option<&'a Language>)>>,
    ) -> Vec<Vec<String>> {
        let mut candidates = Vec::new();
        let mut current_phrase = Vec::new();
        // RAKE has no part-of-speech filter, so the delimiter extension
        // applies to tagged text too.
        let delimiting = self.stopwords.delimiting_untagged(true);

        // A punctuation token ends a phrase, as in the RAKE paper's
        // delimiter set.
        for item in words {
            let Some((word, language)) = item else {
                self.flush_phrase(&mut current_phrase, &mut candidates);
                continue;
            };
            if !word.chars().any(char::is_alphanumeric) {
                self.flush_phrase(&mut current_phrase, &mut candidates);
                continue;
            }
            if word.len() < self.min_word_length {
                continue;
            }
            let word_lower = word.to_lowercase();
            if delimiting.is_stopword_in(word, language) {
                self.flush_phrase(&mut current_phrase, &mut candidates);
            } else {
                current_phrase.push(word_lower);
            }
        }
        self.flush_phrase(&mut current_phrase, &mut candidates);

        candidates
    }

    /// Move `phrase` into `candidates` if its length is in range, then clear it.
    fn flush_phrase(&self, phrase: &mut Vec<String>, candidates: &mut Vec<Vec<String>>) {
        if phrase.len() >= self.min_phrase_length && phrase.len() <= self.max_phrase_length {
            candidates.push(std::mem::take(phrase));
        } else {
            phrase.clear();
        }
    }

    /// Compute word scores using degree/frequency.
    fn compute_word_scores(&self, candidates: &[Vec<String>]) -> HashMap<String, f64> {
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        let mut word_degree: HashMap<String, usize> = HashMap::new();

        for phrase in candidates {
            let phrase_len = phrase.len();
            for word in phrase {
                *word_freq.entry(word.clone()).or_insert(0) += 1;
                *word_degree.entry(word.clone()).or_insert(0) += phrase_len;
            }
        }

        word_freq
            .keys()
            .map(|word| {
                let freq = word_freq[word] as f64;
                let deg = word_degree[word] as f64;
                (word.clone(), deg / freq)
            })
            .collect()
    }
}

impl KeywordExtractor for RakeExtractor {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.score(&self.extract_candidates(text), max_keywords)
    }
}

impl RakeExtractor {
    /// Score candidates by degree over frequency and keep the best `max`.
    fn score(&self, candidates: &[Vec<String>], max_keywords: usize) -> Vec<(String, f64)> {
        let word_scores = self.compute_word_scores(candidates);

        let mut phrase_scores: HashMap<String, f64> = HashMap::new();

        for phrase in candidates {
            let phrase_text = phrase.join(" ");
            let score: f64 = phrase
                .iter()
                .map(|w| word_scores.get(w).unwrap_or(&0.0))
                .sum();
            phrase_scores
                .entry(phrase_text)
                .and_modify(|s| *s = s.max(score))
                .or_insert(score);
        }

        let mut sorted: Vec<_> = phrase_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted.truncate(max_keywords);

        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_stopwords;
    use crate::tests::{assert_score_properties, keyword_strings, GOLDEN_TEXT, TEST_TEXT};

    #[test]
    fn nltk_drops_the_smart_extension() {
        let text = "Graph ranking provides keyphrases.";
        let smart = keyword_strings(&RakeExtractor::new().extract(text, 5));
        assert_eq!(smart, ["graph ranking", "keyphrases"]);
        let nltk = keyword_strings(&RakeExtractor::nltk().extract(text, 5));
        assert_eq!(nltk, ["graph ranking provides keyphrases"]);
    }

    #[test]
    fn test_rake_extraction() {
        let extractor = RakeExtractor::new();
        let keywords = extractor.extract(TEST_TEXT, 5);

        assert!(!keywords.is_empty());
        let phrases: Vec<_> = keywords.iter().map(|(k, _)| k.as_str()).collect();
        assert!(
            phrases.iter().any(|p| p.contains(' ')),
            "RAKE should extract multi-word phrases"
        );
    }

    #[test]
    fn test_stopwords_only() {
        let text = "the and or but if then";
        let extractor = RakeExtractor::new();
        let keywords = extractor.extract(text, 5);
        assert!(keywords.is_empty());
    }

    #[test]
    fn test_custom_stopwords() {
        let mut custom = HashSet::new();
        custom.insert("machine".to_string());
        custom.insert("learning".to_string());

        let extractor = RakeExtractor::new().with_stopwords(custom);
        let keywords = extractor.extract(TEST_TEXT, 10);

        for (keyword, _) in &keywords {
            assert!(
                !keyword
                    .split_whitespace()
                    .any(|w| w == "machine" || w == "learning"),
                "Custom stopwords should be filtered"
            );
        }
    }

    #[test]
    fn test_rake_extract_candidates_stopword_splitting() {
        let rake = RakeExtractor::new();
        let candidates =
            rake.extract_candidates("machine learning is a subset of artificial intelligence");
        let phrases: Vec<String> = candidates.iter().map(|p| p.join(" ")).collect();
        assert!(
            phrases.contains(&"machine learning".to_string()),
            "expected 'machine learning' in {phrases:?}"
        );
        assert!(
            phrases.contains(&"artificial intelligence".to_string()),
            "expected 'artificial intelligence' in {phrases:?}"
        );
        assert!(
            phrases.contains(&"subset".to_string()),
            "expected 'subset' in {phrases:?}"
        );
    }

    #[test]
    fn candidates_keep_hyphenated_words_and_decimals() {
        let rake = RakeExtractor::new();
        let candidates = rake.extract_candidates("A state-of-the-art model, version 3.5, works.");
        assert_eq!(
            candidates,
            vec![
                vec!["state-of-the-art".to_string(), "model".to_string()],
                vec!["version".to_string(), "3.5".to_string()],
                vec!["works".to_string()],
            ]
        );
    }

    #[test]
    fn rake_punctuation_ends_a_phrase() {
        let rake = RakeExtractor::new();
        let candidates = rake.extract_candidates("fast cars, slow trucks. red lorry; blue bus");
        let phrases: Vec<String> = candidates.iter().map(|p| p.join(" ")).collect();
        assert_eq!(
            phrases,
            vec!["fast cars", "slow trucks", "red lorry", "blue bus"]
        );
    }

    #[test]
    fn test_rake_extract_candidates_min_phrase_length() {
        let rake = RakeExtractor::new()
            .with_stopwords(default_stopwords())
            .with_max_phrase_length(5);
        let rake = RakeExtractor {
            min_phrase_length: 2,
            ..rake
        };
        let candidates =
            rake.extract_candidates("machine learning is a subset of artificial intelligence");
        let phrases: Vec<String> = candidates.iter().map(|p| p.join(" ")).collect();
        assert!(
            !phrases.contains(&"subset".to_string()),
            "single-word 'subset' should be filtered with min_phrase_length=2, got {phrases:?}"
        );
        assert!(phrases.contains(&"machine learning".to_string()));
        assert!(phrases.contains(&"artificial intelligence".to_string()));
    }

    #[test]
    fn test_rake_extract_candidates_max_phrase_length() {
        let rake = RakeExtractor::new().with_max_phrase_length(1);
        let candidates = rake.extract_candidates("machine learning is great");
        let phrases: Vec<String> = candidates.iter().map(|p| p.join(" ")).collect();
        assert!(
            !phrases.iter().any(|p| p.contains(' ')),
            "max_phrase_length=1 should block multi-word phrases, got {phrases:?}"
        );
    }

    #[test]
    fn test_rake_compute_word_scores_degree_frequency() {
        let rake = RakeExtractor::new();
        let candidates = vec![
            vec!["deep".to_string(), "learning".to_string()],
            vec!["learning".to_string()],
        ];
        let scores = rake.compute_word_scores(&candidates);
        let deep_score = scores["deep"];
        let learning_score = scores["learning"];
        assert!(
            (deep_score - 2.0).abs() < f64::EPSILON,
            "deep: expected 2.0, got {deep_score}"
        );
        assert!(
            (learning_score - 1.5).abs() < f64::EPSILON,
            "learning: expected 1.5, got {learning_score}"
        );
    }

    #[test]
    fn golden_rake_extracts_key_phrases() {
        let extractor = RakeExtractor::new();
        let keywords = extractor.extract(GOLDEN_TEXT, 10);
        let kw = keyword_strings(&keywords);

        // RAKE should surface multi-word keyphrases containing key terms.
        let has_ml = kw
            .iter()
            .any(|k| k.contains("machine") && k.contains("learning"));
        let has_nn = kw
            .iter()
            .any(|k| k.contains("neural") && k.contains("network"));

        assert!(
            has_ml,
            "RAKE should extract 'machine learning', got: {kw:?}"
        );
        assert!(has_nn, "RAKE should extract 'neural networks', got: {kw:?}");
    }

    #[test]
    fn golden_rake_scores_descending() {
        let extractor = RakeExtractor::new();
        let keywords = extractor.extract(GOLDEN_TEXT, 10);
        for pair in keywords.windows(2) {
            assert!(
                pair[0].1 >= pair[1].1,
                "RAKE scores not descending: {} ({}) vs {} ({})",
                pair[0].0,
                pair[0].1,
                pair[1].0,
                pair[1].1,
            );
        }
    }

    #[test]
    fn edge_single_content_word_rake() {
        // A single non-stopword should produce exactly one result.
        let kw = RakeExtractor::new().extract("optimization", 5);
        assert_eq!(kw.len(), 1);
        assert_eq!(kw[0].0, "optimization");
        assert!(kw[0].1 > 0.0);
    }

    #[test]
    fn property_rake_scores() {
        let texts = [
            GOLDEN_TEXT,
            "Rust is a systems programming language focused on safety and performance.",
            "a",
            "hello world",
        ];
        let extractor = RakeExtractor::new();
        for text in &texts {
            let kw = extractor.extract(text, 20);
            assert_score_properties("RAKE", &kw);
        }
    }
}
