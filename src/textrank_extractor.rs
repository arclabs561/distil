//! Word-level TextRank keyword extractor ([`TextRankExtractor`]).

use std::collections::{HashMap, HashSet};

use crate::{default_stopwords, KeywordExtractor};

/// TextRank keyword extractor.
///
/// Builds a word co-occurrence graph (words within a sliding window are
/// connected) and runs PageRank to find important terms.
///
/// Differs from Mihalcea & Tarau: edges are weighted by co-occurrence count,
/// a stopword filter stands in for the part-of-speech filter, the default
/// window is 4, and it returns single words without merging adjacent
/// keywords into phrases. [`TextRank`](crate::TextRank) follows the paper.
///
/// # Reference
///
/// Mihalcea, R., & Tarau, P. (2004).
/// "TextRank: Bringing Order into Text"
#[derive(Debug, Clone)]
pub struct TextRankExtractor {
    stopwords: HashSet<String>,
    window_size: usize,
    damping: f64,
    iterations: usize,
}

impl Default for TextRankExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextRankExtractor {
    /// Create a new TextRank extractor with default parameters.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            window_size: 4,
            damping: 0.85,
            iterations: 30,
        }
    }

    /// Set window size for co-occurrence.
    pub fn with_window(mut self, size: usize) -> Self {
        // A window below 2 links no words, and `windows(0)` panics.
        self.window_size = size.max(2);
        self
    }

    /// Set damping factor (clamped to [0, 1]).
    pub fn with_damping(mut self, d: f64) -> Self {
        self.damping = d.clamp(0.0, 1.0);
        self
    }

    /// Filter and tokenize text, removing stopwords and short words.
    fn tokenize_filtered(&self, text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty() && w.len() > 2)
            .map(|w| w.to_lowercase())
            .filter(|w| !self.stopwords.contains(w))
            .collect()
    }

    /// Build co-occurrence graph and run PageRank via graphops.
    fn compute_pagerank(&self, words: &[String]) -> HashMap<String, f64> {
        if words.is_empty() {
            return HashMap::new();
        }

        // Sorted so node order, and with it the floating-point summation
        // order inside PageRank, is the same on every call.
        let mut vocab: Vec<_> = words
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        vocab.sort();
        let word_to_idx: HashMap<_, _> = vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();
        let n = vocab.len();

        if n == 0 {
            return HashMap::new();
        }

        // Build adjacency matrix
        let mut adj = vec![vec![0.0; n]; n];

        for window in words.windows(self.window_size) {
            for i in 0..window.len() {
                for j in (i + 1)..window.len() {
                    if let (Some(&idx_i), Some(&idx_j)) =
                        (word_to_idx.get(&window[i]), word_to_idx.get(&window[j]))
                    {
                        adj[idx_i][idx_j] += 1.0;
                        adj[idx_j][idx_i] += 1.0;
                    }
                }
            }
        }

        // Delegate to graphops for PageRank
        let g = graphops::AdjacencyMatrix(&adj);
        let config = graphops::pagerank::PageRankConfig {
            damping: self.damping,
            max_iterations: self.iterations,
            tolerance: 1e-6,
        };
        let scores = graphops::pagerank::pagerank_weighted(&g, config);

        vocab
            .into_iter()
            .enumerate()
            .map(|(i, word)| (word, scores[i]))
            .collect()
    }
}

impl KeywordExtractor for TextRankExtractor {
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        let words = self.tokenize_filtered(text);
        let word_scores = self.compute_pagerank(&words);

        let mut sorted: Vec<_> = word_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted.truncate(max_keywords);

        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_score_properties, keyword_strings, GOLDEN_TEXT, TEST_TEXT};

    #[test]
    fn test_textrank_extraction() {
        let extractor = TextRankExtractor::new();
        let keywords = extractor.extract(TEST_TEXT, 5);

        assert!(!keywords.is_empty());
        let words: Vec<_> = keywords.iter().map(|(k, _)| k.as_str()).collect();
        assert!(
            words
                .iter()
                .any(|w| w.contains("learn") || w.contains("neural")),
            "TextRank should find key terms"
        );
    }

    #[test]
    fn textrank_window_below_two_does_not_panic() {
        let ex = TextRankExtractor::new().with_window(0);
        let kw = ex.extract("graph ranking of words in a short text about graphs", 3);
        assert!(!kw.is_empty());
    }

    #[test]
    fn golden_textrank_extracts_key_terms() {
        let extractor = TextRankExtractor::new();
        let keywords = extractor.extract(GOLDEN_TEXT, 10);
        let kw = keyword_strings(&keywords);

        let has_neural = kw.iter().any(|k| k.contains("neural"));
        let has_learning = kw.iter().any(|k| k.contains("learning"));
        let has_network = kw.iter().any(|k| k.contains("network"));

        assert!(
            has_neural && has_learning,
            "TextRank should extract 'neural' and 'learning', got: {kw:?}"
        );
        assert!(
            has_network,
            "TextRank should extract 'network' variant, got: {kw:?}"
        );
    }

    #[test]
    fn golden_textrank_scores_descending() {
        let extractor = TextRankExtractor::new();
        let keywords = extractor.extract(GOLDEN_TEXT, 10);
        for pair in keywords.windows(2) {
            assert!(
                pair[0].1 >= pair[1].1,
                "TextRank scores not descending: {} ({}) vs {} ({})",
                pair[0].0,
                pair[0].1,
                pair[1].0,
                pair[1].1,
            );
        }
    }

    #[test]
    fn edge_single_content_word_textrank() {
        // TextRank filters words with len <= 2, so "optimization" (12 chars) should pass.
        let kw = TextRankExtractor::new().extract("optimization", 5);
        assert_eq!(kw.len(), 1, "TextRank single word: {kw:?}");
        assert_eq!(kw[0].0, "optimization");
    }

    #[test]
    fn property_textrank_scores() {
        let texts = [
            GOLDEN_TEXT,
            "Rust is a systems programming language focused on safety and performance.",
            "optimization",
        ];
        let extractor = TextRankExtractor::new();
        for text in &texts {
            let kw = extractor.extract(text, 20);
            assert_score_properties("TextRank", &kw);
        }
    }
}
