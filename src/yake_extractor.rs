//! Simplified YAKE-style keyword extractor ([`YakeExtractor`]).

use std::collections::{HashMap, HashSet};

use crate::{default_stopwords, KeywordExtractor};

/// YAKE-style keyword extractor.
///
/// Statistical, unsupervised keyword extractor that uses multiple features
/// (casing, position, frequency, relatedness, sentence frequency) without
/// requiring external corpora.
///
/// This is a simplified scoring inspired by YAKE, not the published
/// algorithm: it has no context-relatedness feature, higher scores are
/// better (YAKE ranks lower scores first), and its scores will not match
/// the reference `yake` package.
///
/// # Reference
///
/// Campos, R., Mangaravite, V., Pasquali, A., Jorge, A., Nunes, C., & Jatowt, A. (2018).
/// "YAKE! Collection-Independent Automatic Keyword Extractor"
#[derive(Debug, Clone)]
pub struct YakeExtractor {
    stopwords: HashSet<String>,
    #[allow(dead_code)]
    window_size: usize,
    n_gram_max: usize,
}

impl Default for YakeExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl YakeExtractor {
    /// Create a new YAKE extractor with default English stopwords.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            window_size: 2,
            n_gram_max: 3,
        }
    }

    /// Set custom stopwords.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Set maximum n-gram length.
    pub fn with_ngram_max(mut self, n: usize) -> Self {
        self.n_gram_max = n;
        self
    }

    /// Tokenize text into words.
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect()
    }

    /// Split text into sentences (simple heuristic).
    fn split_sentences(&self, text: &str) -> Vec<String> {
        text.split(['.', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Compute YAKE features for each word.
    fn compute_word_features(&self, text: &str) -> HashMap<String, f64> {
        let sentences = self.split_sentences(text);
        let words = self.tokenize(text);
        let total_words = words.len() as f64;

        if total_words == 0.0 {
            return HashMap::new();
        }

        let mut word_freq: HashMap<String, usize> = HashMap::new();
        let mut word_first_pos: HashMap<String, usize> = HashMap::new();
        let mut word_uppercase: HashMap<String, usize> = HashMap::new();
        let mut word_sentence_freq: HashMap<String, HashSet<usize>> = HashMap::new();

        for (i, word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            *word_freq.entry(lower.clone()).or_insert(0) += 1;
            word_first_pos.entry(lower.clone()).or_insert(i);

            if i > 0
                && word
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
            {
                *word_uppercase.entry(lower.clone()).or_insert(0) += 1;
            }
        }

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            for word in self.tokenize(sentence) {
                let lower = word.to_lowercase();
                word_sentence_freq
                    .entry(lower)
                    .or_default()
                    .insert(sent_idx);
            }
        }

        let num_sentences = sentences.len().max(1) as f64;

        word_freq
            .keys()
            .filter(|w| !self.stopwords.contains(*w) && w.len() > 1)
            .map(|word| {
                let freq = word_freq[word] as f64;
                let first_pos = *word_first_pos.get(word).unwrap_or(&0) as f64;
                let uppercase = *word_uppercase.get(word).unwrap_or(&0) as f64;
                let sent_freq = word_sentence_freq.get(word).map(|s| s.len()).unwrap_or(0) as f64;

                let pos_score = (first_pos / total_words).ln_1p();
                let freq_norm = freq / total_words;
                let sent_spread = sent_freq / num_sentences;
                let case_score = uppercase / freq.max(1.0);

                let score = (1.0 + case_score) * (1.0 + sent_spread) / (1.0 + pos_score);

                (word.clone(), score * freq_norm.sqrt())
            })
            .collect()
    }
}

impl KeywordExtractor for YakeExtractor {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        let word_scores = self.compute_word_features(text);

        if word_scores.is_empty() {
            return vec![];
        }

        let words: Vec<String> = self
            .tokenize(text)
            .into_iter()
            .map(|w| w.to_lowercase())
            .collect();

        let mut ngram_scores: HashMap<String, f64> = HashMap::new();

        // 1-grams
        for word in &words {
            if let Some(&score) = word_scores.get(word) {
                ngram_scores
                    .entry(word.clone())
                    .and_modify(|s| *s = s.max(score))
                    .or_insert(score);
            }
        }

        // 2-grams through n_gram_max
        for n in 2..=self.n_gram_max {
            for window in words.windows(n) {
                // Every word needs a score. Stopwords and one-character tokens
                // have none, and skipping them in the product while dividing by
                // n would inflate the n-gram above its own scored words.
                if window.iter().any(|w| !word_scores.contains_key(w)) {
                    continue;
                }

                let ngram = window.join(" ");
                let score: f64 = window
                    .iter()
                    .map(|w| word_scores[w])
                    .product::<f64>()
                    .powf(1.0 / n as f64); // Geometric mean

                ngram_scores
                    .entry(ngram)
                    .and_modify(|s| *s = s.max(score))
                    .or_insert(score);
            }
        }

        let mut sorted: Vec<_> = ngram_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Deduplicate: skip keywords that are substrings of already-selected ones
        let mut result = Vec::new();
        for (keyword, score) in sorted {
            let dominated = result.iter().any(|(existing, _): &(String, f64)| {
                existing.contains(&keyword) || keyword.contains(existing)
            });

            if !dominated {
                result.push((keyword, score));
                if result.len() >= max_keywords {
                    break;
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_score_properties, keyword_strings, GOLDEN_TEXT, TEST_TEXT};

    #[test]
    fn test_yake_extraction() {
        let extractor = YakeExtractor::new();
        let keywords = extractor.extract(TEST_TEXT, 5);

        assert!(!keywords.is_empty());
        for (_, score) in &keywords {
            assert!(*score > 0.0);
        }
    }

    #[test]
    fn test_yake_word_features_nonempty() {
        let yake = YakeExtractor::new();
        let features =
            yake.compute_word_features("Rust programming language. Rust is fast and safe.");
        assert!(
            features.contains_key("rust"),
            "expected 'rust' in features: {features:?}"
        );
        assert!(features["rust"] > 0.0);
    }

    #[test]
    fn test_yake_empty_text_features() {
        let yake = YakeExtractor::new();
        let features = yake.compute_word_features("");
        assert!(features.is_empty());
    }

    #[test]
    fn test_yake_ngram_with_unscored_token_is_not_a_candidate() {
        // "2" and "4" are too short to get a word score. They used to drop out
        // of the geometric mean while still counting toward n, so "arms 2"
        // scored sqrt(score(arms)) and outranked "arms" itself.
        let yake = YakeExtractor::new();
        let keywords = yake.extract(
            "The robot has arms 2 and arms 4. Each robot arm carries a camera.",
            20,
        );
        for (keyword, _) in &keywords {
            assert!(
                keyword.split(' ').all(|w| w.len() > 1),
                "{keyword:?} contains a token with no word score: {keywords:?}"
            );
        }
    }

    #[test]
    fn test_yake_ngram_scoring() {
        let yake = YakeExtractor::new().with_ngram_max(2);
        let keywords = yake.extract(
            "graph algorithms solve graph problems. graph algorithms are efficient.",
            10,
        );
        let all_keywords: Vec<&str> = keywords.iter().map(|(k, _)| k.as_str()).collect();
        assert!(
            all_keywords
                .iter()
                .any(|k| k.contains("graph") || k.contains("algorithm")),
            "expected graph-related keyword in {all_keywords:?}"
        );
        for pair in keywords.windows(2) {
            assert!(
                pair[0].1 >= pair[1].1,
                "scores should be descending: {} >= {}",
                pair[0].1,
                pair[1].1,
            );
        }
    }

    #[test]
    fn golden_yake_extracts_key_terms() {
        let extractor = YakeExtractor::new();
        let keywords = extractor.extract(GOLDEN_TEXT, 10);
        let kw = keyword_strings(&keywords);

        let has_neural = kw.iter().any(|k| k.contains("neural"));
        let has_learning = kw.iter().any(|k| k.contains("learning"));

        assert!(
            has_neural,
            "YAKE should extract a keyword containing 'neural', got: {kw:?}"
        );
        assert!(
            has_learning,
            "YAKE should extract a keyword containing 'learning', got: {kw:?}"
        );
    }

    #[test]
    fn golden_yake_no_stopword_keywords() {
        let extractor = YakeExtractor::new();
        let keywords = extractor.extract(GOLDEN_TEXT, 20);
        let stops = default_stopwords();

        for (kw, _) in &keywords {
            // Single-token keywords should not be stopwords.
            if !kw.contains(' ') {
                assert!(
                    !stops.contains(kw),
                    "YAKE returned stopword as keyword: {kw}"
                );
            }
        }
    }

    #[test]
    fn property_yake_scores() {
        let texts = [
            GOLDEN_TEXT,
            "Rust is a systems programming language focused on safety and performance.",
            "hello world",
        ];
        let extractor = YakeExtractor::new();
        for text in &texts {
            let kw = extractor.extract(text, 20);
            assert_score_properties("YAKE", &kw);
        }
    }
}
