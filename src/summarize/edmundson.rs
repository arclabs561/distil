//! Edmundson (1969): a weighted sum of cue, key, title and location
//! scores.
//!
//! - Cue: bonus words in the sentence minus stigma words (compared by stem).
//! - Key: words of the sentence among the frequent bonus words, those whose
//!   document count exceeds `key_threshold` times the most frequent bonus
//!   word's count.
//! - Title: words of the sentence that are content words of the title.
//! - Location: title words in the sentence (as for Title), plus 1 for a
//!   sentence in the first paragraph, 1 in the last paragraph (when there
//!   are several), 1 for a paragraph's first sentence and 1 for its last
//!   (when it has several).
//!
//! The weights default to sumy's: cue 1, key 0, title 1, location 1. sumy
//! takes bonus and stigma words from the caller, and its command line uses
//! heading words as bonus words and stopwords as null words; here bonus and
//! stigma words default to empty, the title is optional and the stopwords
//! are the null words.
//!
//! H. P. Edmundson, "New methods in automatic extracting", Journal of the
//! ACM 16(2), 1969. Parity: sumy `EdmundsonSummarizer`.

use super::{by_score, SentenceExtractor};
use crate::text::{is_content_in, Document, Roles};
use crate::Language;
use std::collections::{HashMap, HashSet};

/// Edmundson's cue, key, title and location method.
#[derive(Debug, Clone)]
pub struct Edmundson {
    stopwords: Roles,
    bonus: Vec<String>,
    stigma: Vec<String>,
    title: HashSet<String>,
    paragraphs: Vec<usize>,
    weights: [f64; 4],
    key_threshold: f64,
}

impl Default for Edmundson {
    fn default() -> Self {
        Self::new()
    }
}

impl Edmundson {
    /// sumy's weights, English stopwords, no bonus, stigma or title words,
    /// one paragraph.
    pub fn new() -> Self {
        Edmundson {
            stopwords: Roles::default(),
            bonus: Vec::new(),
            stigma: Vec::new(),
            title: HashSet::new(),
            paragraphs: Vec::new(),
            weights: [1.0, 0.0, 1.0, 1.0],
            key_threshold: 0.5,
        }
    }

    /// Replace the stopword (null word) list.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords.set_stopwords(stopwords);
        self
    }

    /// Stopword language: its list replaces the current one. Sentences of a
    /// document in another language use that language's stopwords
    /// ([`Stoplist`](crate::text::Stoplist)).
    pub fn with_language(mut self, language: crate::Language) -> Self {
        self.stopwords.set_language(language);
        self
    }

    /// Bonus words, compared by stem: each sentence stems them in its
    /// language ([`Language::stem`]; the method's language for sentences
    /// without one).
    pub fn with_bonus_words<S: AsRef<str>>(mut self, words: &[S]) -> Self {
        self.bonus = words.iter().map(|w| w.as_ref().to_string()).collect();
        self
    }

    /// Stigma words, compared by stem as bonus words are.
    pub fn with_stigma_words<S: AsRef<str>>(mut self, words: &[S]) -> Self {
        self.stigma = words.iter().map(|w| w.as_ref().to_string()).collect();
        self
    }

    /// Title or heading text; its content words' stems are the title words.
    pub fn with_title(mut self, title: &Document) -> Self {
        self.title = title
            .tokens_in()
            .filter(|(t, l)| is_content_in(t, self.stopwords.language(), *l))
            .map(|(t, _)| t)
            .map(|t| t.stem.clone())
            .collect();
        self
    }

    /// Paragraph sizes in sentences, in order; they should sum to the
    /// document's sentence count. Without them the document is one
    /// paragraph.
    pub fn with_paragraphs(mut self, sizes: Vec<usize>) -> Self {
        self.paragraphs = sizes;
        self
    }

    /// Weights of the cue, key, title and location scores (negative values
    /// count as 0).
    pub fn with_weights(mut self, cue: f64, key: f64, title: f64, location: f64) -> Self {
        self.weights = [cue, key, title, location].map(|w| w.max(0.0));
        self
    }

    /// Key words are bonus words whose count exceeds this fraction of the
    /// most frequent bonus word's count (default 0.5).
    pub fn with_key_threshold(mut self, threshold: f64) -> Self {
        self.key_threshold = threshold;
        self
    }
}

impl SentenceExtractor for Edmundson {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        // Bonus and stigma stems per language code.
        let language = |l: Option<&'_ Language>| l.unwrap_or(self.stopwords.language()).clone();
        let mut cues: HashMap<&str, (HashSet<String>, HashSet<String>)> = HashMap::new();
        for (_, l) in doc.sentences_in() {
            let l = language(l);
            let stems = |words: &[String]| words.iter().map(|w| l.stem(w)).collect();
            cues.entry(l.code())
                .or_insert_with(|| (stems(&self.bonus), stems(&self.stigma)));
        }
        let sentence_cues: Vec<&(HashSet<String>, HashSet<String>)> = doc
            .sentences_in()
            .map(|(_, l)| &cues[language(l).code()])
            .collect();
        let mut bonus_counts: HashMap<&str, usize> = HashMap::new();
        for ((s, _), (bonus, _)) in doc.sentences_in().zip(&sentence_cues) {
            for t in s.iter().filter(|t| bonus.contains(&t.stem)) {
                *bonus_counts.entry(t.stem.as_str()).or_insert(0) += 1;
            }
        }
        let max = bonus_counts.values().copied().max().unwrap_or(0) as f64;
        let key: HashSet<&str> = bonus_counts
            .iter()
            .filter(|&(_, &c)| c as f64 / max > self.key_threshold)
            .map(|(&w, _)| w)
            .collect();

        let n = doc.sentences().len();
        let mut sizes = self.paragraphs.clone();
        if sizes.iter().sum::<usize>() != n {
            sizes = vec![n];
        }
        let mut location = Vec::with_capacity(n);
        for (p, &size) in sizes.iter().enumerate() {
            for s in 0..size {
                let first_or_last = |k: usize, len: usize| k == 0 || k + 1 == len;
                let paragraph = f64::from(u8::from(first_or_last(p, sizes.len())));
                let sentence = f64::from(u8::from(first_or_last(s, size)));
                location.push(paragraph + sentence);
            }
        }

        let [w_cue, w_key, w_title, w_location] = self.weights;
        let scores = doc
            .sentences()
            .iter()
            .zip(location)
            .zip(&sentence_cues)
            .map(|((s, loc), (bonus, stigma))| {
                let words = || s.iter().filter(|t| !t.is_punct());
                let count = |set: &dyn Fn(&str) -> bool| words().filter(|t| set(&t.stem)).count();
                let cue =
                    count(&|w| bonus.contains(w)) as f64 - count(&|w| stigma.contains(w)) as f64;
                let key = count(&|w| key.contains(w)) as f64;
                let title = count(&|w| self.title.contains(w)) as f64;
                w_cue * cue + w_key * key + w_title * title + w_location * (title + loc)
            })
            .collect();
        by_score(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two paragraphs of 3 and 2 sentences, title "solar power", bonus word
    /// "significant", stigma word "hardly". Location bonuses: 2 1 2 | 2 2.
    #[test]
    fn combines_the_four_scores() {
        let doc = Document::from_text(
            "Solar power grows. It is significant. Prices fall. \
             Hardly anyone objects. Solar power wins.",
        );
        let e = Edmundson::new()
            .with_title(&Document::from_text("Solar power"))
            .with_bonus_words(&["significant"])
            .with_stigma_words(&["hardly"])
            .with_paragraphs(vec![3, 2])
            .with_weights(1.0, 1.0, 1.0, 1.0);
        let mut got = e.rank(&doc);
        got.sort_by_key(|x| x.0);
        // cue + key + title + (title + location)
        let want = [
            0.0 + 0.0 + 2.0 + 4.0,
            1.0 + 1.0 + 0.0 + 1.0,
            0.0 + 0.0 + 0.0 + 2.0,
            -1.0 + 0.0 + 0.0 + 2.0,
            0.0 + 0.0 + 2.0 + 4.0,
        ];
        let got: Vec<f64> = got.iter().map(|x| x.1).collect();
        assert_eq!(got, want);
    }
}
