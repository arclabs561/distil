//! Extractive summarization: rank the sentences of a [`Document`] and keep
//! the best few.
//!
//! Every method implements [`SentenceExtractor`]: [`rank`] scores every
//! sentence, best first, and [`summarize`] picks sentences under a
//! [`Budget`] (a sentence count or a word count) and returns them in
//! document order. Methods that select greedily (SumBasic, KL-Sum, MMR)
//! return sentences in selection order, scored by their criterion at the
//! moment of selection.
//!
//! Sentences are the document's sentences; words are its non-punctuation
//! tokens, and content words also exclude stopwords (compared lowercased).
//! Methods compare words by [`Token::stem`], so callers control
//! normalization through [`Document::from_tokens`].
//!
//! [`rank`]: SentenceExtractor::rank
//! [`summarize`]: SentenceExtractor::summarize
//! [`Token::stem`]: crate::text::Token::stem

use crate::text::{is_content, Document, Token};
use std::collections::{BTreeMap, HashMap, HashSet};

pub mod centroid;
pub mod edmundson;
pub mod graph;
pub mod kl;
pub mod lead;
pub mod lexrank;
pub mod lsa;
pub mod luhn;
pub mod mmr;
pub mod reduction;
pub mod select;
pub mod sumbasic;
pub mod textrank;

pub use centroid::Centroid;
pub use edmundson::Edmundson;
pub use kl::KlSum;
pub use lead::{Lead, Random};
pub use lexrank::{BiasedLexRank, LexRank};
pub use lsa::{Lsa, LsaVariant};
pub use luhn::Luhn;
pub use mmr::Mmr;
pub use reduction::Reduction;
pub use sumbasic::SumBasic;
pub use textrank::TextRank;

/// A sentence ranking method.
pub trait SentenceExtractor {
    /// Every sentence index with its score, best first.
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)>;

    /// Indices of the summary sentences in document order.
    fn summarize(&self, doc: &Document, budget: Budget) -> Vec<usize> {
        budget.select(doc, &self.rank(doc))
    }
}

/// How much of the document a summary keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Budget {
    /// At most this many sentences.
    Sentences(usize),
    /// At most this many words (non-punctuation tokens) in total. Sentences
    /// are taken in rank order; one that does not fit in what is left is
    /// skipped and later, shorter ones may still be taken.
    Words(usize),
}

impl Budget {
    /// Take sentences from `ranked` (best first) until the budget is spent,
    /// then sort them into document order.
    pub fn select(self, doc: &Document, ranked: &[(usize, f64)]) -> Vec<usize> {
        let mut out: Vec<usize> = match self {
            Budget::Sentences(n) => ranked.iter().take(n).map(|&(i, _)| i).collect(),
            Budget::Words(mut left) => {
                let mut out = Vec::new();
                for &(i, _) in ranked {
                    let n = word_count(&doc.sentences()[i]);
                    if n <= left {
                        left -= n;
                        out.push(i);
                    }
                }
                out
            }
        };
        out.sort_unstable();
        out
    }

    /// True when `n` more words (one more sentence) fit after `used_words`
    /// words in `used_sentences` sentences.
    pub(crate) fn fits(self, used_sentences: usize, used_words: usize, n: usize) -> bool {
        match self {
            Budget::Sentences(k) => used_sentences < k,
            Budget::Words(k) => used_words + n <= k,
        }
    }
}

/// Number of words (non-punctuation tokens) in a sentence.
pub fn word_count(sentence: &[Token]) -> usize {
    sentence.iter().filter(|t| !t.is_punct()).count()
}

/// Stems of the content words of each sentence, in order.
pub(crate) fn content_stems<'a>(doc: &'a Document, stop: &HashSet<String>) -> Vec<Vec<&'a str>> {
    doc.sentences()
        .iter()
        .map(|s| {
            s.iter()
                .filter(|t| is_content(t, stop))
                .map(|t| t.stem.as_str())
                .collect()
        })
        .collect()
}

/// Indices with scores, sorted by descending score; ties keep the earlier
/// sentence first.
pub(crate) fn by_score(scores: Vec<f64>) -> Vec<(usize, f64)> {
    let mut v: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

// -----------------------------------------------------------------------------
// Term statistics (Luhn, SumBasic, KL-Sum, Centroid, and the vectors below)
// -----------------------------------------------------------------------------

/// Occurrences of each stem over all sentences.
pub(crate) fn term_counts<'a>(terms: &[Vec<&'a str>]) -> HashMap<&'a str, usize> {
    let mut c = HashMap::new();
    for &w in terms.iter().flatten() {
        *c.entry(w).or_insert(0) += 1;
    }
    c
}

/// Share of all stem occurrences taken by each stem.
pub(crate) fn distribution<'a>(terms: &[Vec<&'a str>]) -> HashMap<&'a str, f64> {
    let total = terms.iter().map(Vec::len).sum::<usize>() as f64;
    term_counts(terms)
        .into_iter()
        .map(|(w, c)| (w, c as f64 / total))
        .collect()
}

/// A sparse term vector, ordered so that sums over it are reproducible.
pub(crate) type Vector<'a> = BTreeMap<&'a str, f64>;

/// Term counts of each sentence.
pub(crate) fn counts<'a>(terms: &[Vec<&'a str>]) -> Vec<Vector<'a>> {
    terms
        .iter()
        .map(|s| {
            let mut v = Vector::new();
            for &w in s {
                *v.entry(w).or_insert(0.0) += 1.0;
            }
            v
        })
        .collect()
}

/// `ln(N / n_w)`, treating each sentence as a document.
pub(crate) fn sentence_idf<'a>(terms: &[Vec<&'a str>]) -> HashMap<&'a str, f64> {
    let mut df: HashMap<&str, f64> = HashMap::new();
    for s in terms {
        for w in s.iter().copied().collect::<HashSet<_>>() {
            *df.entry(w).or_insert(0.0) += 1.0;
        }
    }
    let n = terms.len() as f64;
    df.into_iter().map(|(w, d)| (w, (n / d).ln())).collect()
}

/// Multiply each term count by its weight from `idf` (0 when missing).
pub(crate) fn weighted<'a>(tf: &[Vector<'a>], idf: &dyn Fn(&str) -> f64) -> Vec<Vector<'a>> {
    tf.iter()
        .map(|v| v.iter().map(|(&w, &c)| (w, c * idf(w))).collect())
        .collect()
}

pub(crate) fn dot(a: &Vector, b: &Vector) -> f64 {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    small
        .iter()
        .filter_map(|(w, x)| large.get(w).map(|y| x * y))
        .sum()
}

/// Cosine similarity, 0 when either vector is zero.
pub(crate) fn cosine(a: &Vector, b: &Vector) -> f64 {
    let d = dot(a, a).sqrt() * dot(b, b).sqrt();
    if d > 0.0 {
        dot(a, b) / d
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_budget_skips_sentences_that_do_not_fit() {
        let doc = Document::from_text("One two three four. Five six. Seven eight nine.");
        let ranked = [(0, 3.0), (2, 2.0), (1, 1.0)];
        assert_eq!(Budget::Words(6).select(&doc, &ranked), vec![0, 1]);
        assert_eq!(Budget::Words(3).select(&doc, &ranked), vec![2]);
        assert_eq!(Budget::Sentences(2).select(&doc, &ranked), vec![0, 2]);
    }

    #[test]
    fn sentence_spans_cover_each_sentence() {
        let text = "  Héllo there.  A second one!\nThird ";
        let doc = Document::from_text(text);
        let got: Vec<&str> = doc
            .sentence_spans()
            .iter()
            .map(|r| &text[r.clone()])
            .collect();
        assert_eq!(got, vec!["Héllo there.", "A second one!", "Third"]);
    }
}
