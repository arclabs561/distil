//! YAKE (Campos et al., 2020), following the published algorithm.
//!
//! Campos, R., Mangaravite, V., Pasquali, A., Jorge, A., Nunes, C. and
//! Jatowt, A. (2020). YAKE! Keyword extraction from single documents using
//! multiple local features. Information Sciences 509, 257-289.
//! doi:10.1016/j.ins.2019.09.013
//!
//! [`YakeExtractor`](crate::YakeExtractor) is an older, simplified scorer
//! kept for compatibility; this type is the published method and matches
//! the `yake` Python package (`tests/pke_parity.rs`).
//!
//! # Method
//!
//! Tokens made only of punctuation split a sentence into blocks. Each other
//! token gets a tag: `d` (number), `u` (unparsable: letters and digits mixed,
//! neither, or more than one punctuation character), `a` (all caps), `n`
//! (capitalized, not sentence-initial) or `p`. A term is the lowercased
//! token with one final `s` removed when longer than three characters; it is
//! a stopword when listed or shorter than three characters.
//!
//! For each term `t`, with `TF` its frequency, `MaxTF` the highest term
//! frequency, and mean and population standard deviation taken over
//! non-stopword terms:
//!
//! - casing `T_Case = max(TF(U), TF(A)) / (1 + ln TF)`;
//! - position `T_Pos = ln(ln(3 + Median(Sen_t)))`, the median over the
//!   indices of the sentences containing `t`;
//! - frequency `T_FNorm = TF / (MeanTF + sigma)`;
//! - relatedness `T_Rel = 1 + (DL/WL + DR/WR) * TF / MaxTF`, where `DL` and
//!   `DR` count the distinct terms seen within `window = 1` tokens before
//!   and after `t` in a block, and `WL` and `WR` the co-occurrences;
//! - spread `T_Sentence = SF(t) / #Sentences`;
//!
//! $$S(t) = \frac{T_{Pos} \cdot T_{Rel}}{T_{Case} + \frac{T_{FNorm}}{T_{Rel}} + \frac{T_{Sentence}}{T_{Rel}}}$$
//!
//! Candidates are n-grams of up to three tokens within a block that contain
//! no `u` or `d` token in some occurrence and neither start nor end with a
//! stopword. With `prob = P(t | t_prev) * P(t_next | t)` for a stopword `t`
//! inside the candidate,
//!
//! $$S(kw) = \frac{\prod_{t \in kw} S'(t)}{TF(kw) \left(1 + \sum_{t \in kw} S''(t)\right)}$$
//!
//! where `S'(t) = S(t)` and `S''(t) = S(t)` for content terms and
//! `S'(t) = 2 - prob`, `S''(t) = -(1 - prob)` for stopwords. Lower is
//! better. Candidates are taken in increasing `S(kw)`, skipping any whose
//! similarity `1 - lev(a, b) / max(|a|, |b|)` to a taken one exceeds 0.9.
//!
//! # Deviations
//!
//! - Tokens and sentences come from a [`Document`] instead of segtok; the
//!   parity test feeds the package's own segmentation.
//! - English stopwords are [`default_stopwords`],
//!   not the package's 574-word list; pass that list with
//!   [`Yake::with_stopwords`] for package scores.
//! - Deduplication uses the Levenshtein ratio (the package's `levs`). The
//!   package's default `seqm` in version 0.7 is a heuristic approximation of
//!   `difflib` similarity and is not reproduced.
//! - Phrases are returned lowercased.

use crate::text::Document;
use crate::{default_stopwords, KeywordExtractor};
use std::collections::{BTreeMap, HashMap, HashSet};

/// YAKE keyphrase extractor (the published algorithm).
#[derive(Debug, Clone)]
pub struct Yake {
    stopwords: HashSet<String>,
    max_ngram: usize,
    window: usize,
    dedup: f64,
}

impl Default for Yake {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
struct Term {
    stopword: bool,
    tf: f64,
    tf_a: f64,
    tf_n: f64,
    sentences: BTreeMap<usize, usize>,
    h: f64,
}

struct Kw {
    key: String,
    terms: Vec<usize>,
    tf: f64,
    valid_tag: bool,
}

impl Yake {
    /// Trigrams, window 1, deduplication threshold 0.9, English stopwords.
    pub fn new() -> Self {
        Self {
            stopwords: default_stopwords(),
            max_ngram: 3,
            window: 1,
            dedup: 0.9,
        }
    }

    /// Stopword list (lowercase).
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Longest candidate in tokens (at least 1).
    pub fn with_max_ngram(mut self, n: usize) -> Self {
        self.max_ngram = n.max(1);
        self
    }

    /// Co-occurrence window in tokens.
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window;
        self
    }

    /// Similarity above which a candidate is dropped as a near-duplicate of
    /// a better one; 1.0 or more keeps every candidate.
    pub fn with_dedup(mut self, threshold: f64) -> Self {
        self.dedup = threshold;
        self
    }

    /// Up to `max` keyphrases with their YAKE scores, best (lowest) first.
    pub fn rank(&self, doc: &Document, max: usize) -> Vec<(String, f64)> {
        let sentences: Vec<Vec<&str>> = doc
            .sentences()
            .iter()
            .map(|s| s.iter().map(|t| t.word.as_str()).collect())
            .collect();
        self.rank_tokens(&sentences, max)
    }

    /// [`Yake::rank`] on sentences of tokens from the caller's tokenizer.
    pub fn rank_tokens<S: AsRef<str>>(
        &self,
        sentences: &[Vec<S>],
        max: usize,
    ) -> Vec<(String, f64)> {
        let mut terms: Vec<Term> = Vec::new();
        let mut term_ids: HashMap<String, usize> = HashMap::new();
        let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
        let mut kws: Vec<Kw> = Vec::new();
        let mut kw_ids: HashMap<String, usize> = HashMap::new();
        for (sid, sentence) in sentences.iter().enumerate() {
            // (tag, lowercased word, term id) of the current block.
            let mut block: Vec<(char, String, usize)> = Vec::new();
            for (pos, word) in sentence.iter().map(AsRef::as_ref).enumerate() {
                if word.chars().all(|c| c.is_ascii_punctuation()) {
                    block.clear();
                    continue;
                }
                let tag = tag(word, pos);
                let id = self.term(word, &mut terms, &mut term_ids);
                let t = &mut terms[id];
                *t.sentences.entry(sid).or_insert(0) += 1;
                t.tf += 1.0;
                t.tf_a += f64::from(tag == 'a');
                t.tf_n += f64::from(tag == 'n');
                if !matches!(tag, 'u' | 'd') {
                    for prev in &block[block.len().saturating_sub(self.window)..] {
                        if !matches!(prev.0, 'u' | 'd') {
                            *edges.entry((prev.2, id)).or_insert(0.0) += 1.0;
                        }
                    }
                }
                block.push((tag, word.to_lowercase(), id));
                let first = block.len().saturating_sub(self.max_ngram);
                for start in (first..block.len()).rev() {
                    let span = &block[start..];
                    let key = span
                        .iter()
                        .map(|b| b.1.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let clean = span.iter().all(|b| !matches!(b.0, 'u' | 'd'));
                    let i = *kw_ids.entry(key.clone()).or_insert_with(|| {
                        kws.push(Kw {
                            key,
                            terms: span.iter().map(|b| b.2).collect(),
                            tf: 0.0,
                            valid_tag: false,
                        });
                        kws.len() - 1
                    });
                    kws[i].tf += 1.0;
                    kws[i].valid_tag |= clean;
                }
            }
        }
        self.score_terms(&mut terms, &edges, sentences.len());
        let mut scored: Vec<(&str, f64)> = kws
            .iter()
            .filter(|k| {
                k.valid_tag
                    && !terms[k.terms[0]].stopword
                    && !terms[k.terms[k.terms.len() - 1]].stopword
            })
            .map(|k| (k.key.as_str(), kw_score(k, &terms, &edges)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut out: Vec<(String, f64)> = Vec::new();
        for (key, h) in scored {
            if out.len() >= max {
                break;
            }
            if self.dedup < 1.0 && out.iter().any(|(o, _)| similarity(key, o) > self.dedup) {
                continue;
            }
            out.push((key.to_string(), h));
        }
        out
    }

    fn term(&self, word: &str, terms: &mut Vec<Term>, ids: &mut HashMap<String, usize>) -> usize {
        let lower = word.to_lowercase();
        let listed = self.stopwords.contains(&lower);
        let mut unique = lower;
        if unique.ends_with('s') && unique.chars().count() > 3 {
            unique.pop();
        }
        if let Some(&id) = ids.get(&unique) {
            return id;
        }
        let letters = unique.chars().filter(|c| !c.is_ascii_punctuation()).count();
        terms.push(Term {
            stopword: listed || self.stopwords.contains(&unique) || letters < 3,
            ..Term::default()
        });
        ids.insert(unique, terms.len() - 1);
        terms.len() - 1
    }

    fn score_terms(
        &self,
        terms: &mut [Term],
        edges: &HashMap<(usize, usize), f64>,
        n_sentences: usize,
    ) {
        let tfs: Vec<f64> = terms.iter().filter(|t| !t.stopword).map(|t| t.tf).collect();
        if tfs.is_empty() {
            return;
        }
        let mean = tfs.iter().sum::<f64>() / tfs.len() as f64;
        let std = (tfs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / tfs.len() as f64).sqrt();
        let max_tf = terms.iter().map(|t| t.tf).fold(0.0, f64::max);
        // (distinct, total) co-occurrences on each side of each term.
        let mut left = vec![(0.0, 0.0); terms.len()];
        let mut right = vec![(0.0, 0.0); terms.len()];
        for (&(a, b), &w) in edges {
            right[a].0 += 1.0;
            right[a].1 += w;
            left[b].0 += 1.0;
            left[b].1 += w;
        }
        let ratio = |(d, w): (f64, f64)| if w == 0.0 { 0.0 } else { d / w };
        for (i, t) in terms.iter_mut().enumerate() {
            let rel = 1.0 + (ratio(left[i]) + ratio(right[i])) * t.tf / max_tf;
            let freq = t.tf / (mean + std);
            let spread = t.sentences.len() as f64 / n_sentences as f64;
            let case = t.tf_a.max(t.tf_n) / (1.0 + t.tf.ln());
            let pos = (3.0 + median(t.sentences.keys())).ln().ln();
            t.h = pos * rel / (case + freq / rel + spread / rel);
        }
    }
}

fn kw_score(k: &Kw, terms: &[Term], edges: &HashMap<(usize, usize), f64>) -> f64 {
    let (mut sum, mut prod) = (0.0, 1.0);
    let n = k.terms.len();
    for (j, &id) in k.terms.iter().enumerate() {
        let t = &terms[id];
        if !t.stopword {
            sum += t.h;
            prod *= t.h;
            continue;
        }
        let p = |a: usize, b: usize, denom: usize| {
            edges.get(&(a, b)).map_or(0.0, |w| w / terms[denom].tf)
        };
        let before = if j > 0 {
            p(k.terms[j - 1], id, k.terms[j - 1])
        } else {
            0.0
        };
        let after = if j + 1 < n {
            p(id, k.terms[j + 1], k.terms[j + 1])
        } else {
            0.0
        };
        let prob = before * after;
        prod *= 2.0 - prob;
        sum -= 1.0 - prob;
    }
    prod / ((sum + 1.0) * k.tf)
}

fn median<'a>(xs: impl Iterator<Item = &'a usize>) -> f64 {
    let v: Vec<f64> = xs.map(|&x| x as f64).collect();
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

/// The `yake` package's token tag.
fn tag(word: &str, pos: usize) -> char {
    let no_commas: String = word.chars().filter(|&c| c != ',').collect();
    let is_digits = |s: &str| !s.is_empty() && s.chars().all(char::is_numeric);
    if is_digits(&no_commas) || is_digits(&no_commas.replacen('.', "", 1)) {
        return 'd';
    }
    let digits = word.chars().filter(|c| c.is_numeric()).count();
    let alpha = word.chars().filter(|c| c.is_alphabetic()).count();
    let punct = word.chars().filter(|c| c.is_ascii_punctuation()).count();
    if (digits > 0 && alpha > 0) || (digits == 0 && alpha == 0) || punct > 1 {
        return 'u';
    }
    let cased: Vec<char> = word
        .chars()
        .filter(|c| c.is_lowercase() || c.is_uppercase())
        .collect();
    if !cased.is_empty() && cased.iter().all(|c| c.is_uppercase()) {
        return 'a';
    }
    let first_upper = word.chars().next().is_some_and(char::is_uppercase);
    let uppers = word.chars().filter(|c| c.is_uppercase()).count();
    if word.chars().count() > 1 && first_upper && pos > 0 && uppers == 1 {
        return 'n';
    }
    'p'
}

/// `1 - levenshtein(a, b) / max(|a|, |b|)`, in characters.
fn similarity(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let longest = a.len().max(b.len());
    if longest == 0 {
        return 1.0;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    1.0 - prev[b.len()] as f64 / longest as f64
}

impl KeywordExtractor for Yake {
    /// Scores are negated so that higher is better.
    fn extract(&self, text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        self.rank(&Document::from_text(text), max_keywords)
            .into_iter()
            .map(|(k, h)| (k, -h))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_follow_the_package() {
        assert_eq!(tag("1,000", 3), 'd');
        assert_eq!(tag("3.14", 3), 'd');
        assert_eq!(tag("v3", 3), 'u');
        assert_eq!(tag("NASA", 0), 'a');
        assert_eq!(tag("Paris", 2), 'n');
        assert_eq!(tag("Paris", 0), 'p');
        assert_eq!(tag("McDonald", 2), 'p');
    }

    #[test]
    fn levenshtein_ratio() {
        assert!((similarity("kitten", "sitting") - (1.0 - 3.0 / 7.0)).abs() < 1e-12);
        assert_eq!(similarity("same", "same"), 1.0);
    }

    /// One sentence "alpha beta": both terms have TF = 1, MaxTF = 1,
    /// mean 1 and sigma 0, so T_FNorm = 1, T_Case = 0, T_Sentence = 1 and
    /// T_Pos = ln(ln 3). alpha has one right neighbour (DR/WR = 1), beta one
    /// left (DL/WL = 1), so T_Rel = 2 for both and
    /// S(t) = 2 ln(ln 3) / (1/2 + 1/2) = 2 ln(ln 3). A unigram candidate
    /// scores S(t) / (1 + S(t)) and the bigram S(t)^2 / (1 + 2 S(t)).
    /// (The yake package gives 0.158317 and 0.025709 for this input.)
    #[test]
    fn scores_match_the_equations_on_a_bigram() {
        let got: HashMap<String, f64> = Yake::new()
            .with_dedup(1.0)
            .rank_tokens(&[vec!["alpha", "beta"]], 10)
            .into_iter()
            .collect();
        let s = 2.0 * 3f64.ln().ln();
        assert!((got["alpha"] - s / (1.0 + s)).abs() < 1e-12, "{got:?}");
        assert!((got["beta"] - s / (1.0 + s)).abs() < 1e-12, "{got:?}");
        assert!(
            (got["alpha beta"] - s * s / (1.0 + 2.0 * s)).abs() < 1e-12,
            "{got:?}"
        );
    }

    #[test]
    fn candidates_may_not_start_or_end_with_a_stopword() {
        let got = Yake::new().rank_tokens(&[vec!["the", "graph", "of", "words"]], 10);
        let keys: Vec<&str> = got.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"graph of words"), "{keys:?}");
        assert!(
            !keys
                .iter()
                .any(|k| k.starts_with("the") || k.ends_with(" of")),
            "{keys:?}"
        );
    }
}
