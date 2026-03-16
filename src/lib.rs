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

use std::collections::{HashMap, HashSet};

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

// =============================================================================
// Stopwords
// =============================================================================

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
/// For other languages, use [`stopwords::for_language`] or construct your own set.
pub fn default_stopwords() -> HashSet<String> {
    STOPWORDS.iter().map(|s| s.to_string()).collect()
}

/// Stopword lists for various languages.
///
/// These are minimal lists covering common function words. For production use
/// with non-English text, consider more comprehensive sources.
pub mod stopwords {
    use std::collections::HashSet;

    /// German stopwords (common function words).
    pub fn german() -> HashSet<String> {
        [
            "der", "die", "das", "und", "in", "zu", "den", "ist", "nicht", "von", "sie", "mit",
            "auf", "es", "ein", "eine", "dem", "für", "sich", "an", "als", "auch", "er", "hat",
            "aus", "bei", "war", "so", "werden", "ich", "ihr", "wir", "aber", "wie", "nur", "oder",
            "nach", "noch", "kann", "über",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// French stopwords.
    pub fn french() -> HashSet<String> {
        [
            "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "à", "au", "aux",
            "que", "qui", "ne", "pas", "pour", "sur", "ce", "cette", "il", "elle", "nous", "vous",
            "ils", "elles", "son", "sa", "ses", "leur", "leurs", "mais", "ou", "donc", "car",
            "avec", "dans", "par",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Spanish stopwords.
    pub fn spanish() -> HashSet<String> {
        [
            "el", "la", "los", "las", "de", "del", "en", "y", "a", "que", "es", "un", "una",
            "por", "con", "no", "para", "se", "su", "al", "lo", "como", "más", "pero", "sus",
            "le", "ya", "o", "este", "si", "porque", "esta", "entre", "cuando", "muy", "sin",
            "sobre", "también", "me",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Portuguese stopwords.
    pub fn portuguese() -> HashSet<String> {
        [
            "o", "a", "os", "as", "de", "da", "do", "das", "dos", "em", "um", "uma", "e", "é",
            "que", "no", "na", "nos", "nas", "por", "para", "com", "não", "se", "mais", "como",
            "mas", "ao", "ele", "ela", "seu", "sua", "ou", "ser", "quando", "muito", "há", "foi",
            "são",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Italian stopwords.
    pub fn italian() -> HashSet<String> {
        [
            "il", "lo", "la", "i", "gli", "le", "di", "a", "da", "in", "con", "su", "per", "tra",
            "fra", "un", "uno", "una", "e", "che", "non", "è", "si", "come", "più", "ma", "o",
            "anche", "questo", "quello", "essere", "sono", "sono", "suo", "sua", "loro", "chi",
            "cui", "dove",
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
            "uit", "naar", "nog", "wel", "kan", "meer", "was", "worden", "tot", "er", "al",
            "worden",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Russian stopwords (Cyrillic).
    pub fn russian() -> HashSet<String> {
        [
            "и", "в", "не", "на", "я", "с", "он", "что", "это", "по", "но", "они", "к", "у",
            "же", "вы", "за", "бы", "так", "от", "все", "как", "она", "его", "только", "или",
            "мы", "ещё", "из", "для", "если", "уже", "при", "их", "во", "когда", "до", "ни",
            "чтобы", "да", "был",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Arabic stopwords.
    pub fn arabic() -> HashSet<String> {
        [
            "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "التي", "الذي", "أن", "كان",
            "قد", "ما", "لم", "لا", "و", "أو", "ثم", "بين", "كل", "بعد", "قبل", "حتى", "إذا",
            "هو", "هي", "هم", "أنت", "نحن",
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
            "en" | "eng" | "english" => Some(super::default_stopwords()),
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
}

// =============================================================================
// RAKE (Rapid Automatic Keyword Extraction)
// =============================================================================

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
    stopwords: HashSet<String>,
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
    /// Create a new RAKE extractor with default English stopwords.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            min_word_length: 1,
            min_phrase_length: 1,
            max_phrase_length: 5,
        }
    }

    /// Set custom stopwords.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
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

    /// Extract candidate phrases by splitting on stopwords and punctuation.
    fn extract_candidates(&self, text: &str) -> Vec<Vec<String>> {
        let mut candidates = Vec::new();
        let mut current_phrase = Vec::new();

        for word in text.split(|c: char| !c.is_alphanumeric() && c != '\'') {
            let word_lower = word.to_lowercase();

            if word.is_empty() || word.len() < self.min_word_length {
                continue;
            }

            if self.stopwords.contains(&word_lower) {
                if current_phrase.len() >= self.min_phrase_length
                    && current_phrase.len() <= self.max_phrase_length
                {
                    candidates.push(current_phrase.clone());
                }
                current_phrase.clear();
            } else {
                current_phrase.push(word_lower);
            }
        }

        if current_phrase.len() >= self.min_phrase_length
            && current_phrase.len() <= self.max_phrase_length
        {
            candidates.push(current_phrase);
        }

        candidates
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
        let candidates = self.extract_candidates(text);
        let word_scores = self.compute_word_scores(&candidates);

        let mut phrase_scores: HashMap<String, f64> = HashMap::new();

        for phrase in &candidates {
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
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(max_keywords);

        sorted
    }
}

// =============================================================================
// YAKE (Yet Another Keyword Extractor)
// =============================================================================

/// YAKE keyword extractor.
///
/// Statistical, unsupervised keyword extractor that uses multiple features
/// (casing, position, frequency, relatedness, sentence frequency) without
/// requiring external corpora.
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
                if window.iter().any(|w| self.stopwords.contains(w)) {
                    continue;
                }

                let ngram = window.join(" ");
                let score: f64 = window
                    .iter()
                    .filter_map(|w| word_scores.get(w))
                    .product::<f64>()
                    .powf(1.0 / n as f64); // Geometric mean

                ngram_scores
                    .entry(ngram)
                    .and_modify(|s| *s = s.max(score))
                    .or_insert(score);
            }
        }

        let mut sorted: Vec<_> = ngram_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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

// =============================================================================
// TextRank for Keywords
// =============================================================================

/// TextRank keyword extractor.
///
/// Builds a word co-occurrence graph (words within a sliding window are
/// connected) and runs PageRank to find important terms.
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
        self.window_size = size;
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

        let vocab: Vec<_> = words
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
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
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(max_keywords);

        sorted
    }
}

// =============================================================================
// TF-IDF (Simple Baseline)
// =============================================================================

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

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_keywords);

        scored
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TEXT: &str = "Machine learning is a subset of artificial intelligence. \
        Deep learning uses neural networks for machine learning tasks. \
        Neural networks are inspired by biological neural networks.";

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
    fn test_yake_extraction() {
        let extractor = YakeExtractor::new();
        let keywords = extractor.extract(TEST_TEXT, 5);

        assert!(!keywords.is_empty());
        for (_, score) in &keywords {
            assert!(*score > 0.0);
        }
    }

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

    #[test]
    fn test_stopwords_only() {
        let text = "the and or but if then";
        let extractor = RakeExtractor::new();
        let keywords = extractor.extract(text, 5);
        assert!(keywords.is_empty());
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
    fn test_unicode_accented_text() {
        let extractor = TfIdfExtractor::new().with_log_tf(false);
        let text = "cafe resume naive cafe resume cafe";
        let keywords = extractor.extract(text, 5);
        assert_eq!(keywords[0].0, "cafe");
        assert_eq!(keywords[1].0, "resume");
        assert_eq!(keywords[2].0, "naive");
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

    #[test]
    fn test_stopwords_for_language() {
        assert!(stopwords::for_language("en").is_some());
        assert!(stopwords::for_language("de").is_some());
        assert!(stopwords::for_language("fr").is_some());
        assert!(stopwords::for_language("es").is_some());
        assert!(stopwords::for_language("pt").is_some());
        assert!(stopwords::for_language("it").is_some());
        assert!(stopwords::for_language("nl").is_some());
        assert!(stopwords::for_language("ru").is_some());
        assert!(stopwords::for_language("ar").is_some());
        assert!(stopwords::for_language("xx").is_none());
    }
}
