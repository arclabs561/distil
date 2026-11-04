//! Text preprocessing shared by the keyphrase methods.
//!
//! A [`Document`] is a list of sentences of [`Token`]s. Each token carries its
//! surface word, a lowercased stem, and an optional part-of-speech tag.
//! [`Document::from_text`] tokenizes, splits sentences and stems without
//! tagging; [`Document::from_tagged`] takes tokens and universal POS tags from
//! the caller's tagger, which lets the methods select noun-phrase candidates
//! the way their papers (and pke) do.
//!
//! Candidate selection returns [`Candidate`]s grouped by lexical form (the
//! stems of their words), in order of first occurrence:
//!
//! - [`delimited_phrases`]: maximal runs of tokens that are neither stopwords
//!   nor punctuation. This is the default when tokens have no tags.
//! - [`ngrams`]: every n-gram inside those runs.
//! - [`all_ngrams`]: every n-gram of a sentence, for pke's select-then-filter
//!   order ([`filter_candidates`]).
//! - [`longest_sequences`]: maximal runs of tokens accepted by a predicate,
//!   for example tags in {NOUN, PROPN, ADJ}.
//! - [`noun_phrases`]: matches of `(ADJ)*(NOUN|PROPN)+`.
//!
//! Stems come from the English Snowball stemmer (Porter2) in `rust-stemmers`.
//! pke uses NLTK's Porter stemmer, which differs from Porter2 on about 2% of
//! word types (157 of the 7,335 alphabetic types in the Inspec test split).
//! Callers who need specific stems pass them through [`Document::from_tokens`].

use std::collections::{HashMap, HashSet};

/// Universal part-of-speech tags (Universal Dependencies v2 UPOS).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum Pos {
    Adj,
    Adp,
    Adv,
    Aux,
    Cconj,
    Det,
    Intj,
    Noun,
    Num,
    Part,
    Pron,
    Propn,
    Punct,
    Sconj,
    Sym,
    Verb,
    X,
}

impl Pos {
    /// Parse a UPOS tag such as `"NOUN"`. Returns `None` for other tags
    /// (spaCy's `SPACE`, Penn Treebank tags).
    pub fn from_upos(tag: &str) -> Option<Pos> {
        Some(match tag {
            "ADJ" => Pos::Adj,
            "ADP" => Pos::Adp,
            "ADV" => Pos::Adv,
            "AUX" => Pos::Aux,
            "CCONJ" => Pos::Cconj,
            "DET" => Pos::Det,
            "INTJ" => Pos::Intj,
            "NOUN" => Pos::Noun,
            "NUM" => Pos::Num,
            "PART" => Pos::Part,
            "PRON" => Pos::Pron,
            "PROPN" => Pos::Propn,
            "PUNCT" => Pos::Punct,
            "SCONJ" => Pos::Sconj,
            "SYM" => Pos::Sym,
            "VERB" => Pos::Verb,
            "X" => Pos::X,
            _ => return None,
        })
    }

    /// Noun, proper noun or adjective: the word classes the graph methods
    /// put in their word graphs.
    pub fn is_noun_or_adjective(self) -> bool {
        matches!(self, Pos::Noun | Pos::Propn | Pos::Adj)
    }
}

/// One token of a [`Document`].
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// Surface form as it appears in the text.
    pub word: String,
    /// Lowercased stem; graph nodes and candidate identity use it.
    pub stem: String,
    /// Part-of-speech tag, if the document was tagged.
    pub pos: Option<Pos>,
}

impl Token {
    /// A token with its stem computed by [`stem`].
    pub fn new(word: &str, pos: Option<Pos>) -> Token {
        Token {
            word: word.to_string(),
            stem: stem(word),
            pos,
        }
    }

    /// True when the word has no alphanumeric character.
    pub fn is_punct(&self) -> bool {
        !self.word.chars().any(char::is_alphanumeric)
    }
}

/// Lowercase and stem an English word with the Snowball English stemmer.
pub fn stem(word: &str) -> String {
    thread_local! {
        static STEMMER: rust_stemmers::Stemmer =
            rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English);
    }
    let lower = word.to_lowercase();
    STEMMER.with(|s| s.stem(&lower).into_owned())
}

/// A tokenized document: sentences of tokens.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Document {
    sentences: Vec<Vec<Token>>,
}

impl Document {
    /// Tokenize, split into sentences and stem untagged text.
    ///
    /// Words are runs of alphanumeric characters, joined across a single
    /// inner `-`, `'` or `.` (so "state-of-the-art", "don't" and "3.5" stay
    /// whole); every other non-space character is its own token. A sentence
    /// ends at `.`, `!` or `?` followed by whitespace and an uppercase letter
    /// or digit, or by the end of the text.
    pub fn from_text(text: &str) -> Document {
        let chars: Vec<char> = text.chars().collect();
        let mut sentences = Vec::new();
        let mut current: Vec<Token> = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            if c.is_alphanumeric() {
                let start = i;
                while i < chars.len() {
                    let joiner = matches!(chars[i], '-' | '\'' | '.' | '\u{2019}')
                        && i + 1 < chars.len()
                        && chars[i + 1].is_alphanumeric()
                        && (chars[i] != '.' || chars[i - 1].is_ascii_digit());
                    if !(chars[i].is_alphanumeric() || joiner) {
                        break;
                    }
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                current.push(Token::new(&word, None));
                continue;
            }
            current.push(Token::new(&c.to_string(), None));
            i += 1;
            if matches!(c, '.' | '!' | '?') {
                let mut j = i;
                while j < chars.len() && chars[j].is_whitespace() {
                    j += 1;
                }
                let at_end = j == chars.len();
                let next_starts = j > i && (chars[j].is_uppercase() || chars[j].is_numeric());
                if at_end || next_starts {
                    sentences.push(std::mem::take(&mut current));
                }
            }
        }
        if !current.is_empty() {
            sentences.push(current);
        }
        Document { sentences }
    }

    /// Build from the caller's tokenizer and tagger: sentences of
    /// `(word, tag)` pairs. Stems are computed with [`stem`].
    pub fn from_tagged<S: AsRef<str>>(sentences: &[Vec<(S, Pos)>]) -> Document {
        Document {
            sentences: sentences
                .iter()
                .map(|s| {
                    s.iter()
                        .map(|(w, p)| Token::new(w.as_ref(), Some(*p)))
                        .collect()
                })
                .collect(),
        }
    }

    /// Build from fully specified tokens, including stems.
    pub fn from_tokens(sentences: Vec<Vec<Token>>) -> Document {
        Document { sentences }
    }

    /// The sentences.
    pub fn sentences(&self) -> &[Vec<Token>] {
        &self.sentences
    }

    /// All tokens in order.
    pub fn tokens(&self) -> impl Iterator<Item = &Token> {
        self.sentences.iter().flatten()
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.sentences.iter().map(Vec::len).sum()
    }

    /// True when the document has no tokens.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True when every token has a part-of-speech tag.
    pub fn is_tagged(&self) -> bool {
        !self.is_empty() && self.tokens().all(|t| t.pos.is_some())
    }
}

/// A keyphrase candidate: all occurrences of one lexical form.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Stems of the candidate's words; two spans with the same stems are
    /// the same candidate.
    pub lexical_form: Vec<String>,
    /// Lowercased words of the first occurrence, joined by single spaces.
    pub surface: String,
    /// Token offset (counted over the whole document) of each occurrence.
    pub offsets: Vec<usize>,
}

impl Candidate {
    /// Number of words.
    pub fn len(&self) -> usize {
        self.lexical_form.len()
    }

    /// True for a candidate with no words (never produced by selection).
    pub fn is_empty(&self) -> bool {
        self.lexical_form.is_empty()
    }

    /// Offset of the first occurrence.
    pub fn first_offset(&self) -> usize {
        self.offsets[0]
    }
}

/// Collects spans into candidates keyed by lexical form.
#[derive(Default)]
struct Collector {
    index: HashMap<Vec<String>, usize>,
    out: Vec<Candidate>,
}

impl Collector {
    fn add(&mut self, span: &[Token], offset: usize) {
        if span.is_empty() {
            return;
        }
        let form: Vec<String> = span.iter().map(|t| t.stem.clone()).collect();
        match self.index.get(&form) {
            Some(&i) => self.out[i].offsets.push(offset),
            None => {
                let surface = span
                    .iter()
                    .map(|t| t.word.to_lowercase())
                    .collect::<Vec<_>>()
                    .join(" ");
                self.index.insert(form.clone(), self.out.len());
                self.out.push(Candidate {
                    lexical_form: form,
                    surface,
                    offsets: vec![offset],
                });
            }
        }
    }
}

/// Maximal runs of consecutive tokens (within a sentence) accepted by `keep`.
pub fn longest_sequences(doc: &Document, keep: impl Fn(&Token) -> bool) -> Vec<Candidate> {
    let mut c = Collector::default();
    let mut shift = 0;
    for sentence in doc.sentences() {
        let mut start = None;
        for (j, token) in sentence.iter().enumerate() {
            match (keep(token), start) {
                (true, None) => start = Some(j),
                (false, Some(s)) => {
                    c.add(&sentence[s..j], shift + s);
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            c.add(&sentence[s..], shift + s);
        }
        shift += sentence.len();
    }
    c.out
}

/// True when the token is neither punctuation nor (lowercased) a stopword.
pub fn is_content(token: &Token, stopwords: &HashSet<String>) -> bool {
    !token.is_punct() && !stopwords.contains(&token.word.to_lowercase())
}

/// Maximal stopword- and punctuation-delimited phrases of at most
/// `max_words` words; longer runs are dropped.
pub fn delimited_phrases(
    doc: &Document,
    stopwords: &HashSet<String>,
    max_words: usize,
) -> Vec<Candidate> {
    let mut out = longest_sequences(doc, |t| is_content(t, stopwords));
    out.retain(|c| c.len() <= max_words);
    out
}

/// Every n-gram of 1 to `max_n` words inside stopword- and
/// punctuation-delimited runs.
pub fn ngrams(doc: &Document, stopwords: &HashSet<String>, max_n: usize) -> Vec<Candidate> {
    ngrams_where(doc, |t| is_content(t, stopwords), max_n)
}

/// Every n-gram of 1 to `max_n` tokens, all accepted by `keep`, within a
/// sentence.
pub fn ngrams_where(doc: &Document, keep: impl Fn(&Token) -> bool, max_n: usize) -> Vec<Candidate> {
    let mut c = Collector::default();
    let mut shift = 0;
    for sentence in doc.sentences() {
        for j in 0..sentence.len() {
            for k in j + 1..=(j + max_n).min(sentence.len()) {
                if !keep(&sentence[k - 1]) {
                    break;
                }
                c.add(&sentence[j..k], shift + j);
            }
        }
        shift += sentence.len();
    }
    c.out
}

/// Every n-gram of 1 to `max_n` tokens within each sentence, stopwords and
/// punctuation included, as pke's `ngram_selection` builds them. Pass the
/// result through [`filter_candidates`] to get pke's n-gram candidates: the
/// filter looks only at each lexical form's first occurrence, so the
/// occurrence counts include spans that [`ngrams`] would never produce.
pub fn all_ngrams(doc: &Document, max_n: usize) -> Vec<Candidate> {
    let mut c = Collector::default();
    let mut shift = 0;
    for sentence in doc.sentences() {
        for j in 0..sentence.len() {
            for k in j + 1..=(j + max_n).min(sentence.len()) {
                c.add(&sentence[j..k], shift + j);
            }
        }
        shift += sentence.len();
    }
    c.out
}

/// Noun phrases matching `(ADJ)*(NOUN|PROPN)+`, scanned left to right with
/// longest matches, as pke's grammar `NP:{<ADJ>*<NOUN|PROPN>+}` does. Phrases
/// longer than `max_words` are dropped. Needs a tagged document; untagged
/// tokens match nothing.
pub fn noun_phrases(doc: &Document, max_words: usize) -> Vec<Candidate> {
    let mut c = Collector::default();
    let mut shift = 0;
    let is = |t: &Token, f: fn(Pos) -> bool| t.pos.is_some_and(f);
    let adj = |p: Pos| p == Pos::Adj;
    let noun = |p: Pos| matches!(p, Pos::Noun | Pos::Propn);
    for sentence in doc.sentences() {
        let mut j = 0;
        while j < sentence.len() {
            let mut k = j;
            while k < sentence.len() && is(&sentence[k], adj) {
                k += 1;
            }
            let nouns = k;
            while k < sentence.len() && is(&sentence[k], noun) {
                k += 1;
            }
            if k > nouns {
                if k - j <= max_words {
                    c.add(&sentence[j..k], shift + j);
                }
                j = k;
            } else {
                j += 1;
            }
        }
        shift += sentence.len();
    }
    c.out
}

/// pke's default candidate filter (`candidate_filtering`), applied to the
/// lowercased words of each candidate's first occurrence. A candidate is
/// dropped when a word is a stopword, a word is only punctuation, the words
/// total fewer than 3 characters, a word has fewer than 2 characters, it has
/// more than `max_words` words, or a word has characters other than
/// alphanumerics and `-`.
pub fn filter_candidates(
    candidates: Vec<Candidate>,
    stopwords: &HashSet<String>,
    max_words: usize,
) -> Vec<Candidate> {
    candidates
        .into_iter()
        .filter(|c| {
            let words: Vec<&str> = c.surface.split(' ').collect();
            !words.iter().any(|w| stopwords.contains(*w))
                && !words.iter().any(|w| !w.chars().any(char::is_alphanumeric))
                && words.iter().map(|w| w.chars().count()).sum::<usize>() >= 3
                && words.iter().all(|w| w.chars().count() >= 2)
                && c.len() <= max_words
                && words
                    .iter()
                    .all(|w| w.chars().all(|ch| ch.is_alphanumeric() || ch == '-'))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(doc: &Document) -> Vec<Vec<&str>> {
        doc.sentences()
            .iter()
            .map(|s| s.iter().map(|t| t.word.as_str()).collect())
            .collect()
    }

    #[test]
    fn tokenizer_keeps_inner_hyphens_and_decimals() {
        let doc = Document::from_text("A state-of-the-art model, v3.5, works. Next one!");
        assert_eq!(
            words(&doc),
            vec![
                vec![
                    "A",
                    "state-of-the-art",
                    "model",
                    ",",
                    "v3.5",
                    ",",
                    "works",
                    "."
                ],
                vec!["Next", "one", "!"],
            ]
        );
        let doc = Document::from_text("Pi is 3.14 here.");
        assert_eq!(words(&doc), vec![vec!["Pi", "is", "3.14", "here", "."]]);
    }

    #[test]
    fn sentence_needs_uppercase_after_period() {
        let doc = Document::from_text("see fig. two. Then more.");
        assert_eq!(doc.sentences().len(), 2);
    }

    #[test]
    fn stems_are_lowercased_porter2() {
        assert_eq!(stem("Networks"), "network");
        assert_eq!(stem("generalization"), "general");
    }

    fn tagged(spec: &[(&str, &str)]) -> Document {
        let s: Vec<(&str, Pos)> = spec
            .iter()
            .map(|(w, t)| (*w, Pos::from_upos(t).unwrap()))
            .collect();
        Document::from_tagged(&[s])
    }

    #[test]
    fn noun_phrase_pattern() {
        // A trailing adjective with no noun after it starts no phrase.
        let doc = tagged(&[
            ("Mad", "ADJ"),
            ("cow", "NOUN"),
            ("disease", "NOUN"),
            ("killed", "VERB"),
            ("old", "ADJ"),
            ("big", "ADJ"),
            ("cattle", "NOUN"),
            ("quickly", "ADV"),
            ("red", "ADJ"),
        ]);
        let got: Vec<_> = noun_phrases(&doc, 5)
            .into_iter()
            .map(|c| c.surface)
            .collect();
        assert_eq!(got, vec!["mad cow disease", "old big cattle"]);
        assert_eq!(noun_phrases(&doc, 2).len(), 0);
    }

    #[test]
    fn candidates_group_by_stems_and_keep_offsets() {
        let doc = Document::from_text("Neural networks learn. A neural network learns.");
        let c = delimited_phrases(&doc, &crate::default_stopwords(), 5);
        let neural = c
            .iter()
            .find(|c| c.surface == "neural networks learn")
            .unwrap();
        assert_eq!(neural.offsets, vec![0, 5]);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn ngrams_stay_inside_runs() {
        let doc = Document::from_text("deep learning of graph models");
        let got: Vec<_> = ngrams(&doc, &crate::default_stopwords(), 2)
            .into_iter()
            .map(|c| c.surface)
            .collect();
        assert_eq!(
            got,
            vec![
                "deep",
                "deep learning",
                "learning",
                "graph",
                "graph models",
                "models"
            ]
        );
    }

    #[test]
    fn filter_matches_pke_rules() {
        let doc = Document::from_text("x ab. c++ code. the model. big-data sets. a b c d e f");
        let stop = crate::default_stopwords();
        let all = longest_sequences(&doc, |t| !t.is_punct());
        let kept: Vec<_> = filter_candidates(all, &stop, 5)
            .into_iter()
            .map(|c| c.surface)
            .collect();
        assert_eq!(kept, vec!["code", "big-data sets"]);
    }
}
