//! Maximal marginal relevance (Carbonell and Goldstein 1998) over a
//! document's sentences.
//!
//! Sentences are tf-idf vectors of content-word stems (idf `ln(N / n_w)`
//! over the document's sentences). Relevance is the cosine to the query,
//! or to the whole document when there is no query; similarity is the
//! cosine between sentences. Selection is the shared
//! [`select::mmr`], which also takes candidates and
//! scores from elsewhere. `lambda` defaults to 0.7.
//!
//! J. Carbonell and J. Goldstein, "The use of MMR, diversity-based
//! reranking for reordering documents and producing summaries", SIGIR
//! 1998. No maintained reference implementation; tested on hand-computed
//! inputs.

use super::graph::similarity_matrix;
use super::{
    content_stems, cosine, counts, select, sentence_idf, weighted, word_count, Budget,
    SentenceExtractor,
};
use crate::text::{is_content, Document};
use std::collections::HashSet;

/// MMR sentence selection.
#[derive(Debug, Clone)]
pub struct Mmr {
    stopwords: HashSet<String>,
    lambda: f64,
    query: Option<Vec<String>>,
}

impl Default for Mmr {
    fn default() -> Self {
        Self::new()
    }
}

impl Mmr {
    /// Lambda 0.7, relevance to the whole document, English stopwords.
    pub fn new() -> Self {
        Mmr {
            stopwords: crate::default_stopwords(),
            lambda: 0.7,
            query: None,
        }
    }

    /// Replace the stopword list.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Weight of relevance against redundancy, in [0, 1].
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// Measure relevance against this query instead of the document.
    pub fn with_query(mut self, query: &Document) -> Self {
        self.query = Some(
            query
                .tokens()
                .filter(|t| is_content(t, &self.stopwords))
                .map(|t| t.stem.clone())
                .collect(),
        );
        self
    }

    fn select(&self, doc: &Document, budget: Budget) -> Vec<(usize, f64)> {
        let terms = content_stems(doc, &self.stopwords);
        let idf = sentence_idf(&terms);
        let idf = |w: &str| idf.get(w).copied().unwrap_or(0.0);
        let vectors = weighted(&counts(&terms), &idf);
        let target: Vec<&str> = match &self.query {
            Some(q) => q.iter().map(String::as_str).collect(),
            None => terms.iter().flatten().copied().collect(),
        };
        let target = &weighted(&counts(&[target]), &idf)[0];
        let relevance: Vec<f64> = vectors.iter().map(|v| cosine(v, target)).collect();
        let sim = similarity_matrix(terms.len(), true, |i, j| cosine(&vectors[i], &vectors[j]));
        let lengths: Vec<usize> = doc.sentences().iter().map(|s| word_count(s)).collect();
        select::mmr(&relevance, &sim, self.lambda, &lengths, budget)
    }
}

impl SentenceExtractor for Mmr {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        self.select(doc, Budget::Sentences(doc.sentences().len()))
    }

    fn summarize(&self, doc: &Document, budget: Budget) -> Vec<usize> {
        let mut out: Vec<usize> = self.select(doc, budget).iter().map(|x| x.0).collect();
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sentences 0 and 1 are copies: once MMR takes one, the other has
    /// redundancy 1 and comes last.
    #[test]
    fn avoids_repeating_a_sentence() {
        let doc = Document::from_text(
            "Solar power is cheap. Solar power is cheap. Wind power is cheap too. \
             Storage costs fall.",
        );
        let r = Mmr::new().with_lambda(0.5).rank(&doc);
        assert_eq!(r[3].0, 1, "{r:?}");
        assert_eq!(Mmr::new().summarize(&doc, Budget::Sentences(2)).len(), 2);
    }
}
