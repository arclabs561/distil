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
//! # Languages
//!
//! Each sentence of a document has a [`Language`], chosen by a
//! [`LanguagePolicy`] in [`Document::from_text_with`]: one fixed language
//! (English by default), one detected for the document, or one detected per
//! sentence. A sentence is stemmed with its language's Snowball stemmer
//! (Porter2 for English; languages without a Snowball stemmer keep the
//! lowercased word) and filtered with its language's stopwords. A method's
//! own language ([`Stoplist`] for `Language`) applies to the sentences in
//! that language, so a method's custom stopword list still wins there.
//!
//! pke uses NLTK's Porter stemmer, which differs from Porter2 on about 2% of
//! word types (157 of the 7,335 alphabetic types in the Inspec test split).
//! Callers who need specific stems pass them through [`Document::from_tokens`].

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use crate::{Language, LanguagePolicy};
use textprep::language::Detector;

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
    /// A token with its English stem, from [`stem`].
    pub fn new(word: &str, pos: Option<Pos>) -> Token {
        Token::new_in(word, pos, english())
    }

    /// A token stemmed in `language` ([`Language::stem`]).
    pub fn new_in(word: &str, pos: Option<Pos>, language: &Language) -> Token {
        Token {
            word: word.to_string(),
            stem: language.stem(word),
            pos,
        }
    }

    /// True when the word has no alphanumeric character.
    pub fn is_punct(&self) -> bool {
        !self.word.chars().any(char::is_alphanumeric)
    }
}

/// English, built once.
pub(crate) fn english() -> &'static Language {
    static EN: OnceLock<Language> = OnceLock::new();
    EN.get_or_init(Language::english)
}

/// Lowercase and stem an English word with the Snowball English stemmer
/// (Porter2). U+2018 and U+2019 count as `'`, so `company’s` stems like
/// `company's`. Same as `Language::english().stem(word)`.
pub fn stem(word: &str) -> String {
    english().stem(word)
}

/// Byte ranges of the sentences of `text`, without surrounding whitespace.
///
/// A re-export of [`textprep::segment::sentence_byte_ranges`]: UAX #29
/// sentence boundaries, with single line breaks joined and periods after
/// known English abbreviations (`Mr.`, `e.g.`, `U.S.`) kept inside the
/// sentence.
pub use textprep::segment::sentence_byte_ranges as sentence_ranges;

/// Byte ranges of the tokens of `text`, skipping whitespace.
///
/// A re-export of [`textprep::segment::word_byte_ranges`]: UAX #29 word
/// boundaries ("don't", "3.5", "U.S" are single words, every punctuation
/// character is its own token), with hyphenated compounds such as
/// "state-of-the-art" kept whole.
pub use textprep::segment::word_byte_ranges as word_ranges;

/// A tokenized document: sentences of tokens.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Document {
    sentences: Vec<Vec<Token>>,
    /// Byte range of each sentence in the source text (from_text only).
    spans: Vec<std::ops::Range<usize>>,
    /// Language of each sentence; empty for documents built from tokens.
    languages: Vec<Language>,
}

/// A detector that never answers, for detection policies without the
/// `detect` feature.
#[cfg(not(feature = "detect"))]
fn no_detection(_: &str) -> Option<Language> {
    None
}

impl Document {
    /// Tokenize, split into sentences and stem untagged English text.
    ///
    /// Sentences follow [`sentence_ranges`]; words follow [`word_ranges`].
    /// [`Document::sentence_spans`] gives each sentence's byte range in
    /// `text`, from its first token to its last non-space character. The
    /// same as [`Document::from_text_with`] with the default policy,
    /// `LanguagePolicy::Fixed(Language::english())`.
    pub fn from_text(text: &str) -> Document {
        Document::from_text_with(text, &LanguagePolicy::default())
    }

    /// Like [`Document::from_text`], with each sentence's language chosen
    /// by `policy` and each sentence stemmed in its language.
    ///
    /// `Detect` and `DetectPerSentence` use whatlang through textprep's
    /// detector with the `detect` feature. Without it they detect nothing:
    /// `Detect` gives English and `DetectPerSentence` its fallback. Use
    /// [`Document::from_text_with_detector`] for another detector.
    pub fn from_text_with(text: &str, policy: &LanguagePolicy) -> Document {
        #[cfg(feature = "detect")]
        let detector = textprep::language::WhatlangDetector::new();
        #[cfg(not(feature = "detect"))]
        let detector = no_detection;
        Document::from_text_with_detector(text, policy, &detector)
    }

    /// Like [`Document::from_text_with`], detecting languages with
    /// `detector` (any [`Detector`], including a closure
    /// `Fn(&str) -> Option<Language>`). Texts with fewer than 10 letters
    /// are never passed to it.
    pub fn from_text_with_detector<D: Detector + ?Sized>(
        text: &str,
        policy: &LanguagePolicy,
        detector: &D,
    ) -> Document {
        let mut doc = Document::default();
        for s in textprep::language::sentence_languages_with(text, policy, detector) {
            let range = s.range;
            let words = word_ranges(&text[range.clone()]);
            let Some(first) = words.first() else {
                continue;
            };
            doc.spans.push(range.start + first.start..range.end);
            doc.sentences.push(
                words
                    .into_iter()
                    .map(|w| {
                        let word = &text[range.start + w.start..range.start + w.end];
                        Token::new_in(word, None, &s.language)
                    })
                    .collect(),
            );
            doc.languages.push(s.language);
        }
        doc
    }

    /// Build from the caller's tokenizer and tagger: sentences of
    /// `(word, tag)` pairs, stemmed in English with [`stem`].
    pub fn from_tagged<S: AsRef<str>>(sentences: &[Vec<(S, Pos)>]) -> Document {
        Document::from_tagged_in(sentences, english())
    }

    /// Like [`Document::from_tagged`], with every sentence in `language`
    /// and stemmed with its stemmer.
    pub fn from_tagged_in<S: AsRef<str>>(
        sentences: &[Vec<(S, Pos)>],
        language: &Language,
    ) -> Document {
        Document {
            spans: Vec::new(),
            languages: vec![language.clone(); sentences.len()],
            sentences: sentences
                .iter()
                .map(|s| {
                    s.iter()
                        .map(|(w, p)| Token::new_in(w.as_ref(), Some(*p), language))
                        .collect()
                })
                .collect(),
        }
    }

    /// Build from fully specified tokens, including stems. The sentences
    /// have no language, so methods filter them with their own
    /// ([`Stoplist`]).
    pub fn from_tokens(sentences: Vec<Vec<Token>>) -> Document {
        Document {
            sentences,
            spans: Vec::new(),
            languages: Vec::new(),
        }
    }

    /// Byte range of each sentence in the text given to
    /// [`Document::from_text`]; empty for documents built from tokens.
    pub fn sentence_spans(&self) -> &[std::ops::Range<usize>] {
        &self.spans
    }

    /// Language of each sentence; empty for documents built with
    /// [`Document::from_tokens`].
    pub fn sentence_languages(&self) -> &[Language] {
        &self.languages
    }

    /// The sentences.
    pub fn sentences(&self) -> &[Vec<Token>] {
        &self.sentences
    }

    /// Each sentence with its language, if the document has one.
    pub(crate) fn sentences_in(&self) -> impl Iterator<Item = (&[Token], Option<&Language>)> {
        self.sentences
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_slice(), self.languages.get(i)))
    }

    /// All tokens in order, each with its sentence's language.
    pub(crate) fn tokens_in(&self) -> impl Iterator<Item = (&Token, Option<&Language>)> {
        self.sentences_in()
            .flat_map(|(s, l)| s.iter().map(move |t| (t, l)))
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

/// A stopword lookup that can depend on the sentence's language.
///
/// - A `HashSet<String>` is one list for every sentence. Words are compared
///   lowercased, also with U+2018 and U+2019 replaced by `'`.
/// - A [`Language`] is a method's language: its own stopwords for sentences
///   in that language (same code) and for sentences without a language
///   ([`Document::from_tokens`]), and the sentence language's stopwords
///   for sentences in another language.
pub trait Stoplist {
    /// Whether `word` is a stopword in a sentence of language `sentence`.
    fn is_stopword_in(&self, word: &str, sentence: Option<&Language>) -> bool;
}

impl Stoplist for HashSet<String> {
    fn is_stopword_in(&self, word: &str, _: Option<&Language>) -> bool {
        let lower = word.to_lowercase();
        self.contains(&lower)
            || self.contains(textprep::stopwords::fold_apostrophes(&lower).as_ref())
    }
}

impl Stoplist for Language {
    fn is_stopword_in(&self, word: &str, sentence: Option<&Language>) -> bool {
        match sentence {
            Some(s) if s.code() != self.code() => s.is_stopword(word),
            _ => self.is_stopword(word),
        }
    }
}

/// Stopwords split by role (see the [`crate::stopwords`] module): the
/// language's list for term weighting, and the same list plus an extension
/// for candidate boundaries in untagged text.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Roles {
    language: Language,
    /// Extra candidate delimiters, lowercase with `'` for U+2018 and U+2019;
    /// `None` is the default, SMART for English sentences.
    delimiters: Option<HashSet<String>>,
}

impl Roles {
    pub(crate) fn set_language(&mut self, language: Language) {
        self.language = language;
    }

    pub(crate) fn set_stopwords(&mut self, stopwords: HashSet<String>) {
        self.language = self.language.clone().with_stopwords(stopwords);
    }

    pub(crate) fn set_delimiters<I: IntoIterator<Item = S>, S: AsRef<str>>(&mut self, words: I) {
        self.delimiters = Some(words.into_iter().map(|w| key(w.as_ref())).collect());
    }

    /// The language, also the stoplist for term weighting.
    pub(crate) fn language(&self) -> &Language {
        &self.language
    }

    /// The stoplist for candidate boundaries in `doc`: with the extension
    /// when `doc` is untagged, the language's stopwords alone when it is
    /// tagged (part-of-speech tags select its candidates).
    pub(crate) fn delimiting(&self, doc: &Document) -> Delimiting<'_> {
        self.delimiting_untagged(!doc.is_tagged())
    }

    /// [`Roles::delimiting`] for text that is (or is not) untagged.
    pub(crate) fn delimiting_untagged(&self, untagged: bool) -> Delimiting<'_> {
        Delimiting {
            roles: self,
            untagged,
        }
    }

    /// `text` as a document in the language, for `extract`.
    pub(crate) fn document(&self, text: &str) -> Document {
        Document::from_text_with(text, &LanguagePolicy::Fixed(self.language.clone()))
    }
}

/// Lowercase, with `'` for U+2018 and U+2019: how stoplists store words.
fn key(word: &str) -> String {
    textprep::stopwords::fold_apostrophes(word).to_lowercase()
}

/// The SMART English list, built once.
fn smart() -> &'static HashSet<String> {
    static SMART: OnceLock<HashSet<String>> = OnceLock::new();
    SMART.get_or_init(textprep::stopwords::smart_english)
}

/// A [`Roles`]'s candidate-boundary stoplist: the language's stopwords,
/// plus, in untagged text, the extension. The default extension is SMART
/// in English sentences; a caller's extension applies in sentences of the
/// method's language.
pub(crate) struct Delimiting<'a> {
    roles: &'a Roles,
    untagged: bool,
}

impl Stoplist for Delimiting<'_> {
    fn is_stopword_in(&self, word: &str, sentence: Option<&Language>) -> bool {
        let own = &self.roles.language;
        if own.is_stopword_in(word, sentence) {
            return true;
        }
        if !self.untagged {
            return false;
        }
        let language = match sentence {
            Some(s) if s.code() != own.code() => s,
            _ => own,
        };
        match &self.roles.delimiters {
            None => language.code() == "en" && smart().contains(&key(word)),
            Some(extra) => language.code() == own.code() && extra.contains(&key(word)),
        }
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
    longest_sequences_in(doc, |t, _| keep(t))
}

/// [`longest_sequences`] with each token's sentence language.
pub(crate) fn longest_sequences_in(
    doc: &Document,
    keep: impl Fn(&Token, Option<&Language>) -> bool,
) -> Vec<Candidate> {
    let mut c = Collector::default();
    let mut shift = 0;
    for (sentence, language) in doc.sentences_in() {
        let mut start = None;
        for (j, token) in sentence.iter().enumerate() {
            match (keep(token, language), start) {
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
/// U+2018 and U+2019 in the word also match `'` in the list.
pub fn is_content(token: &Token, stopwords: &HashSet<String>) -> bool {
    is_content_in(token, stopwords, None)
}

/// True when the token is neither punctuation nor a stopword of a sentence
/// in `language` (see [`Stoplist`]).
pub fn is_content_in<S: Stoplist + ?Sized>(
    token: &Token,
    stopwords: &S,
    language: Option<&Language>,
) -> bool {
    !token.is_punct() && !stopwords.is_stopword_in(&token.word, language)
}

/// Maximal stopword- and punctuation-delimited phrases of at most
/// `max_words` words; longer runs are dropped. Each sentence is delimited
/// with the stopwords of its language ([`Stoplist`]).
pub fn delimited_phrases<S: Stoplist + ?Sized>(
    doc: &Document,
    stopwords: &S,
    max_words: usize,
) -> Vec<Candidate> {
    let mut out = longest_sequences_in(doc, |t, l| is_content_in(t, stopwords, l));
    out.retain(|c| c.len() <= max_words);
    out
}

/// Every n-gram of 1 to `max_n` words inside stopword- and
/// punctuation-delimited runs.
pub fn ngrams<S: Stoplist + ?Sized>(doc: &Document, stopwords: &S, max_n: usize) -> Vec<Candidate> {
    ngrams_where_in(doc, |t, l| is_content_in(t, stopwords, l), max_n)
}

/// Every n-gram of 1 to `max_n` tokens, all accepted by `keep`, within a
/// sentence.
pub fn ngrams_where(doc: &Document, keep: impl Fn(&Token) -> bool, max_n: usize) -> Vec<Candidate> {
    ngrams_where_in(doc, |t, _| keep(t), max_n)
}

/// [`ngrams_where`] with each token's sentence language.
pub(crate) fn ngrams_where_in(
    doc: &Document,
    keep: impl Fn(&Token, Option<&Language>) -> bool,
    max_n: usize,
) -> Vec<Candidate> {
    let mut c = Collector::default();
    let mut shift = 0;
    for (sentence, language) in doc.sentences_in() {
        for j in 0..sentence.len() {
            for k in j + 1..=(j + max_n).min(sentence.len()) {
                if !keep(&sentence[k - 1], language) {
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
/// alphanumerics and `-`. Stopwords are looked up with no sentence
/// language; see [`filter_candidates_in`] for documents with languages.
pub fn filter_candidates<S: Stoplist + ?Sized>(
    candidates: Vec<Candidate>,
    stopwords: &S,
    max_words: usize,
) -> Vec<Candidate> {
    candidates
        .into_iter()
        .filter(|c| keeps(c, stopwords, None, max_words))
        .collect()
}

/// [`filter_candidates`] for candidates of `doc`, looking up stopwords in
/// the language of the sentence of each candidate's first occurrence.
pub fn filter_candidates_in<S: Stoplist + ?Sized>(
    doc: &Document,
    candidates: Vec<Candidate>,
    stopwords: &S,
    max_words: usize,
) -> Vec<Candidate> {
    let languages = doc.sentence_languages();
    if languages.is_empty() {
        return filter_candidates(candidates, stopwords, max_words);
    }
    // First token offset of each sentence, to find a candidate's sentence.
    let starts: Vec<usize> = doc
        .sentences()
        .iter()
        .scan(0, |n, s| {
            let start = *n;
            *n += s.len();
            Some(start)
        })
        .collect();
    candidates
        .into_iter()
        .filter(|c| {
            let i = starts.partition_point(|&s| s <= c.first_offset()) - 1;
            keeps(c, stopwords, languages.get(i), max_words)
        })
        .collect()
}

fn keeps<S: Stoplist + ?Sized>(
    c: &Candidate,
    stopwords: &S,
    language: Option<&Language>,
    max_words: usize,
) -> bool {
    let words: Vec<&str> = c.surface.split(' ').collect();
    !words.iter().any(|w| stopwords.is_stopword_in(w, language))
        && !words.iter().any(|w| !w.chars().any(char::is_alphanumeric))
        && words.iter().map(|w| w.chars().count()).sum::<usize>() >= 3
        && words.iter().all(|w| w.chars().count() >= 2)
        && c.len() <= max_words
        && words
            .iter()
            .all(|w| w.chars().all(|ch| ch.is_alphanumeric() || ch == '-'))
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
    fn words_follow_uax29_but_keep_hyphenated_compounds() {
        let text = "U.S. e.g. don\u{2019}t x86_64 non-linear -- a - b C++";
        let got: Vec<&str> = word_ranges(text).into_iter().map(|r| &text[r]).collect();
        assert_eq!(
            got,
            vec![
                "U.S",
                ".",
                "e.g",
                ".",
                "don\u{2019}t",
                "x86_64",
                "non-linear",
                "-",
                "-",
                "a",
                "-",
                "b",
                "C",
                "+",
                "+"
            ]
        );
    }

    #[test]
    fn sentence_needs_uppercase_after_period() {
        let doc = Document::from_text("see fig. two. Then more.");
        assert_eq!(doc.sentences().len(), 2);
    }

    #[test]
    fn sentence_ends_after_closing_quote() {
        let text = "He left in 2014.\" Later he (aged 40.) returned! Why? Yes.";
        let got: Vec<&str> = sentence_ranges(text)
            .into_iter()
            .map(|r| &text[r])
            .collect();
        assert_eq!(
            got,
            vec![
                "He left in 2014.\"",
                "Later he (aged 40.) returned!",
                "Why?",
                "Yes."
            ]
        );
    }

    #[test]
    fn abbreviations_do_not_end_sentences() {
        let text = "Mr. Smith met Dr. J. Doe in the U.S. on Jan. 5. He hasn't. U.S. Senator \
                    Lee told me. We left the U.S. The end.";
        let got: Vec<&str> = sentence_ranges(text)
            .into_iter()
            .map(|r| &text[r])
            .collect();
        assert_eq!(
            got,
            vec![
                "Mr. Smith met Dr. J. Doe in the U.S. on Jan. 5.",
                "He hasn't.",
                "U.S. Senator Lee told me.",
                "We left the U.S.",
                "The end."
            ]
        );
    }

    #[test]
    fn single_line_breaks_join_and_blank_lines_split() {
        let text = "A hard\nwrapped sentence.\nNext one\r\n\r\nHeading\n\nBody text.";
        let got: Vec<&str> = sentence_ranges(text)
            .into_iter()
            .map(|r| &text[r])
            .collect();
        assert_eq!(
            got,
            vec![
                "A hard\nwrapped sentence.",
                "Next one",
                "Heading",
                "Body text."
            ]
        );
    }

    #[test]
    fn spans_index_the_source_text() {
        let text = "  \u{201C}Caf\u{e9} au lait.\u{201D} Next.  ";
        let doc = Document::from_text(text);
        let got: Vec<&str> = doc
            .sentence_spans()
            .iter()
            .map(|r| &text[r.clone()])
            .collect();
        assert_eq!(got, vec!["\u{201C}Caf\u{e9} au lait.\u{201D}", "Next."]);
        assert_eq!(doc.sentences().len(), 2);
    }

    #[test]
    fn trailing_whitespace_after_final_period() {
        let doc = Document::from_text("One sentence.\n");
        assert_eq!(words(&doc), vec![vec!["One", "sentence", "."]]);
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

    /// German if the text has "die" or "und", else English if it has
    /// "the", else nothing.
    fn german_detector(text: &str) -> Option<Language> {
        let has = |w: &str| text.split_whitespace().any(|t| t.eq_ignore_ascii_case(w));
        if has("die") || has("und") {
            Language::from_code("de")
        } else if has("the") {
            Some(Language::english())
        } else {
            None
        }
    }

    const MIXED: &str = "Die Hunde und die Katzen schlafen im Garten. \
                         The dogs and the cats sleep in the garden.";

    #[test]
    fn each_sentence_is_stemmed_and_filtered_in_its_language() {
        let policy = LanguagePolicy::DetectPerSentence {
            fallback: Language::english(),
        };
        let doc = Document::from_text_with_detector(MIXED, &policy, &german_detector);
        let codes: Vec<&str> = doc.sentence_languages().iter().map(|l| l.code()).collect();
        assert_eq!(codes, ["de", "en"]);
        // German Snowball: "Hunde" -> "hund"; English Porter2: "dogs" -> "dog".
        assert_eq!(doc.sentences()[0][1].stem, "hund");
        assert_eq!(doc.sentences()[1][1].stem, "dog");
        // An English method filters the German sentence with German stopwords.
        let got: Vec<String> = delimited_phrases(&doc, &Language::english(), 5)
            .into_iter()
            .map(|c| c.surface)
            .collect();
        assert_eq!(
            got,
            [
                "hunde",
                "katzen schlafen",
                "garten",
                "dogs",
                "cats sleep",
                "garden"
            ]
        );
        // One set for every sentence: English "die" is not a stopword.
        let en = Language::english().stopwords().clone();
        assert_eq!(
            delimited_phrases(&doc, &en, 10)[0].surface,
            "die hunde und die katzen schlafen im garten"
        );
    }

    #[test]
    fn fixed_language_matches_from_text_for_english() {
        let text = "Graph-based ranking of keyphrases. It works.";
        let en = Document::from_text_with(text, &LanguagePolicy::Fixed(Language::english()));
        assert_eq!(en, Document::from_text(text));
        let de = Document::from_text_with(
            "Die Häuser stehen.",
            &LanguagePolicy::Fixed(Language::from_code("de").unwrap()),
        );
        assert_eq!(de.sentences()[0][1].stem, "haus");
        assert_eq!(de.sentence_spans().len(), 1);
        assert_eq!(de.sentence_spans()[0], 0..19);
    }

    #[test]
    fn method_list_applies_to_its_own_language_only() {
        let policy = LanguagePolicy::DetectPerSentence {
            fallback: Language::english(),
        };
        let doc = Document::from_text_with_detector(MIXED, &policy, &german_detector);
        // A custom English list ("cats") on an English method; German
        // sentences keep NLTK's German list.
        let method = Language::english().with_stopwords(["the", "and", "in", "cats"]);
        let got: Vec<String> = delimited_phrases(&doc, &method, 5)
            .into_iter()
            .map(|c| c.surface)
            .collect();
        assert_eq!(
            got,
            [
                "hunde",
                "katzen schlafen",
                "garten",
                "dogs",
                "sleep",
                "garden"
            ]
        );
        // Documents built from tokens have no language: the method's list
        // applies everywhere.
        let tokens = Document::from_tokens(doc.sentences().to_vec());
        assert!(tokens.sentence_languages().is_empty());
        assert_eq!(
            delimited_phrases(&tokens, &method, 10)[0].surface,
            "die hunde und die katzen schlafen im garten"
        );
    }

    #[test]
    fn delimiters_end_candidates_but_not_content_words() {
        let doc = Document::from_text("Smith et al proposed graph ranking.");
        let mut roles = Roles::default();
        roles.set_delimiters(["ET", "al", "proposed"]);
        let got: Vec<String> = delimited_phrases(&doc, &roles.delimiting(&doc), 5)
            .into_iter()
            .map(|c| c.surface)
            .collect();
        assert_eq!(got, ["smith", "graph ranking"]);
        let content: Vec<&str> = doc
            .tokens()
            .filter(|t| is_content_in(t, roles.language(), None))
            .map(|t| t.word.as_str())
            .collect();
        assert_eq!(
            content,
            ["Smith", "et", "al", "proposed", "graph", "ranking"]
        );
        // The extension belongs to the method's language.
        let de = Document::from_text_with(
            "Smith et al",
            &LanguagePolicy::Fixed(Language::from_code("de").unwrap()),
        );
        assert_eq!(delimited_phrases(&de, &roles.delimiting(&de), 5).len(), 1);
    }

    #[test]
    fn curly_apostrophes_match_the_list() {
        let doc = Document::from_text("They don\u{2019}t rank. Company\u{2019}s graph.");
        let stop = crate::default_stopwords();
        let words: Vec<&str> = doc
            .tokens()
            .filter(|t| is_content(t, &stop))
            .map(|t| t.word.as_str())
            .collect();
        assert_eq!(words, ["rank", "Company\u{2019}s", "graph"]);
        assert_eq!(stem("Company\u{2019}s"), stem("company's"));
    }

    #[test]
    fn smart_extends_english_boundaries_in_untagged_text_only() {
        let roles = Roles::default();
        let text = "The following graph ranking provides various keyphrases.";
        let doc = Document::from_text(text);
        let got: Vec<String> = delimited_phrases(&doc, &roles.delimiting(&doc), 5)
            .into_iter()
            .map(|c| c.surface)
            .collect();
        assert_eq!(got, ["graph ranking", "keyphrases"]);
        // Weighting keeps NLTK: "following" is still a content word.
        assert!(is_content_in(
            &doc.sentences()[0][1],
            roles.language(),
            None
        ));
        // Tagged text: part-of-speech tags select candidates, NLTK only.
        let tagged = Document::from_tagged(&[vec![("following", Pos::Adj), ("graph", Pos::Noun)]]);
        let delimiting = roles.delimiting(&tagged);
        assert!(!delimiting.is_stopword_in("following", None));
        assert!(delimiting.is_stopword_in("the", None));
        // Other languages keep NLTK; an empty extension restores NLTK for English.
        let de = Language::from_code("de").unwrap();
        assert!(!roles
            .delimiting(&doc)
            .is_stopword_in("following", Some(&de)));
        let mut nltk = Roles::default();
        nltk.set_delimiters([""; 0]);
        assert!(!nltk.delimiting(&doc).is_stopword_in("following", None));
    }
}
