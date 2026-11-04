//! Single-document TF-IDF keyword extractor ([`TfIdfExtractor`]).

use std::collections::{HashMap, HashSet};

use crate::{default_stopwords, KeywordExtractor};

/// Single-document TF-IDF keyword extractor.
///
/// For single-document extraction, this reduces to term frequency
/// with optional logarithmic scaling.
#[derive(Debug, Clone, Default)]
pub struct TfIdfExtractor {
    stopwords: HashSet<String>,
    use_log_tf: bool,
}

impl TfIdfExtractor {
    /// Create a new TF-IDF extractor with default English stopwords and log-TF enabled.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            use_log_tf: true,
        }
    }

    /// Set whether to use `ln(1 + tf)` instead of raw term frequency.
    pub fn with_log_tf(mut self, use_log: bool) -> Self {
        self.use_log_tf = use_log;
        self
    }
}

impl KeywordExtractor for TfIdfExtractor {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        let mut freq: HashMap<String, usize> = HashMap::new();

        for word in text.split(|c: char| !c.is_alphanumeric()) {
            let lower = word.to_lowercase();
            if !lower.is_empty() && lower.len() > 2 && !self.stopwords.contains(&lower) {
                *freq.entry(lower).or_insert(0) += 1;
            }
        }

        let mut scored: Vec<_> = freq
            .into_iter()
            .map(|(word, count)| {
                let score = if self.use_log_tf {
                    (count as f64).ln_1p()
                } else {
                    count as f64
                };
                (word, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(max_keywords);

        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_score_properties, keyword_strings, GOLDEN_TEXT, TEST_TEXT};

    #[test]
    fn test_tfidf_extraction() {
        let extractor = TfIdfExtractor::new();
        let keywords = extractor.extract(TEST_TEXT, 5);

        assert!(!keywords.is_empty());
        assert!(
            keywords.iter().any(|(k, _)| k == "neural"),
            "TF-IDF should rank frequent words high"
        );
    }

    #[test]
    fn test_multilingual() {
        let text =
            "机器学习 is machine learning in Chinese. 人工智能 means artificial intelligence.";
        let extractor = TfIdfExtractor::new();
        let keywords = extractor.extract(text, 10);

        assert!(!keywords.is_empty());
    }

    #[test]
    fn test_tfidf_term_frequency_ordering() {
        let extractor = TfIdfExtractor::new();
        let text = "data data data algorithm";
        let keywords = extractor.extract(text, 10);
        assert!(keywords.len() == 2, "expected 2 terms, got {keywords:?}");
        assert_eq!(keywords[0].0, "data", "most frequent term should be first");
        assert!(
            keywords[0].1 > keywords[1].1,
            "data score ({}) should exceed algorithm score ({})",
            keywords[0].1,
            keywords[1].1,
        );
    }

    #[test]
    fn test_tfidf_log_tf_vs_raw() {
        let log_extractor = TfIdfExtractor::new().with_log_tf(true);
        let raw_extractor = TfIdfExtractor::new().with_log_tf(false);

        let text = "data data data science science";
        let log_kw = log_extractor.extract(text, 10);
        let raw_kw = raw_extractor.extract(text, 10);

        let raw_data_score = raw_kw.iter().find(|(k, _)| k == "data").unwrap().1;
        let log_data_score = log_kw.iter().find(|(k, _)| k == "data").unwrap().1;
        assert!(
            (raw_data_score - 3.0).abs() < f64::EPSILON,
            "raw TF for 'data' should be 3.0, got {raw_data_score}"
        );
        assert!(
            (log_data_score - (3.0_f64).ln_1p()).abs() < f64::EPSILON,
            "log TF for 'data' should be ln(4), got {log_data_score}"
        );
    }

    #[test]
    fn test_unicode_accented_text() {
        let extractor = TfIdfExtractor::new().with_log_tf(false);
        let text = "cafe resume naive cafe resume cafe";
        let keywords = extractor.extract(text, 5);
        assert_eq!(keywords[0].0, "cafe");
        assert_eq!(keywords[1].0, "resume");
        assert_eq!(keywords[2].0, "naive");
    }

    #[test]
    fn golden_tfidf_extracts_key_terms() {
        let extractor = TfIdfExtractor::new();
        let keywords = extractor.extract(GOLDEN_TEXT, 10);
        let kw = keyword_strings(&keywords);

        // "neural" and "learning" are the most frequent non-stopwords.
        assert!(
            kw[..3].contains(&"neural".to_string()),
            "TF-IDF top-3 should include 'neural', got: {kw:?}"
        );

        let has_learning = kw.iter().any(|k| k == "learning");
        let has_networks = kw.iter().any(|k| k == "networks");

        assert!(
            has_learning,
            "TF-IDF should include 'learning', got: {kw:?}"
        );
        assert!(
            has_networks,
            "TF-IDF should include 'networks', got: {kw:?}"
        );
    }

    #[test]
    fn golden_tfidf_all_scores_positive() {
        let extractor = TfIdfExtractor::new();
        let keywords = extractor.extract(GOLDEN_TEXT, 20);
        for (kw, score) in &keywords {
            assert!(
                *score > 0.0,
                "TF-IDF score for '{kw}' should be positive, got {score}"
            );
        }
    }

    #[test]
    fn property_tfidf_scores() {
        let texts = [
            GOLDEN_TEXT,
            "Rust is a systems programming language focused on safety and performance.",
            "data data science",
            "algorithm",
        ];
        let extractor = TfIdfExtractor::new();
        for text in &texts {
            let kw = extractor.extract(text, 20);
            assert_score_properties("TF-IDF", &kw);
        }
    }
}
