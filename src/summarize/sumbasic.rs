//! SumBasic (Nenkova and Vanderwende 2005): greedy selection by average
//! word probability, discounting words already covered.
//!
//! `p(w)` is the share of content-word occurrences with stem `w`. Each step
//! picks the sentence with the highest average `p` over its content words
//! (0 for a sentence without any), using the shared term distribution of
//! [`crate::summarize`] and the greedy loop of [`select`](super::select).
//! By default only sentences containing the currently most probable word
//! compete (step 3 of Vanderwende et al. 2007). Then every word of the chosen sentence is
//! updated once, `p(w) <- p(w)^2`. Scores are the winning averages, in
//! selection order.
//!
//! A. Nenkova and L. Vanderwende, "The impact of frequency on
//! summarization", Microsoft Research MSR-TR-2005-101, 2005. L. Vanderwende,
//! H. Suzuki, C. Brockett and A. Nenkova, "Beyond SumBasic: Task-focused
//! summarization with sentence simplification and lexical expansion",
//! Information Processing and Management 43(6), 2007. Parity: sumy
//! `SumBasicSummarizer`, which drops the most-probable-word rule
//! ([`SumBasic::with_top_word_rule`]`(false)`) and squares a word once per
//! occurrence rather than once (the fixture generator patches the latter).

use super::select::greedy;
use super::{content_stems, distribution, Budget, SentenceExtractor};
use crate::text::Document;
use std::collections::{HashMap, HashSet};

/// SumBasic.
#[derive(Debug, Clone)]
pub struct SumBasic {
    stopwords: HashSet<String>,
    top_word_rule: bool,
}

impl Default for SumBasic {
    fn default() -> Self {
        Self::new()
    }
}

impl SumBasic {
    /// English stopwords, with the paper's most-probable-word rule.
    pub fn new() -> Self {
        SumBasic {
            stopwords: crate::default_stopwords(),
            top_word_rule: true,
        }
    }

    /// Replace the stopword list.
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Whether only sentences containing the most probable remaining word
    /// compete (the paper's step 3); sumy omits the rule.
    pub fn with_top_word_rule(mut self, on: bool) -> Self {
        self.top_word_rule = on;
        self
    }
}

impl SentenceExtractor for SumBasic {
    fn rank(&self, doc: &Document) -> Vec<(usize, f64)> {
        let terms = content_stems(doc, &self.stopwords);
        let p0 = distribution(&terms);
        // After m chosen sentences contain w, p(w) = p0(w)^(2^m).
        greedy(
            terms.len(),
            &[],
            Budget::Sentences(terms.len()),
            |i, picked| {
                let mut m: HashMap<&str, i32> = HashMap::new();
                for w in picked
                    .iter()
                    .flat_map(|&j| terms[j].iter().collect::<HashSet<_>>())
                {
                    *m.entry(w).or_insert(0) += 1;
                }
                let p = |w: &str| p0[w].powf(2f64.powi(m.get(w).copied().unwrap_or(0)));
                if self.top_word_rule {
                    let top = (0..terms.len())
                        .filter(|j| !picked.contains(j))
                        .flat_map(|j| terms[j].iter())
                        .map(|&w| (w, p(w)))
                        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(a.0)));
                    if let Some((w, _)) = top {
                        if !terms[i].contains(&w) {
                            return f64::NEG_INFINITY;
                        }
                    }
                }
                let n = terms[i].len();
                if n == 0 {
                    return 0.0;
                }
                terms[i].iter().map(|&w| p(w)).sum::<f64>() / n as f64
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Content words: a b | a c | d, so p = a 2/5, b c d 1/5. Sentences 0
    /// and 1 tie at 3/10 and the earlier wins; then a -> 4/25, b -> 1/25,
    /// sentence 1 averages 9/50 and sentence 2 keeps 1/5.
    #[test]
    fn follows_the_update_rule() {
        let doc = Document::from_text("Alpha beta. Alpha gamma. Delta.");
        let r = SumBasic::new().with_top_word_rule(false).rank(&doc);
        let order: Vec<usize> = r.iter().map(|x| x.0).collect();
        assert_eq!(order, vec![0, 2, 1]);
        assert!((r[0].1 - 0.3).abs() < 1e-12 && (r[1].1 - 0.2).abs() < 1e-12);
        assert!((r[2].1 - 9.0 / 50.0).abs() < 1e-12);
    }

    /// p = a b 2/6, c d 1/6. The top word is a (ties go to the smaller
    /// stem), so sentence 0 (avg 1/3) beats sentence 1 (5/18). After the
    /// update a, b -> 1/9 and the top word is delta, held only by sentence 2.
    #[test]
    fn top_word_rule_restricts_candidates() {
        let doc = Document::from_text("Alpha beta. Alpha beta gamma. Delta.");
        let r = SumBasic::new().rank(&doc);
        assert_eq!(r[0].0, 0);
        assert_eq!(r[1].0, 2);
    }
}
