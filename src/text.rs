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
use std::ops::Range;

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

/// Byte ranges of the sentences of `text`, without surrounding whitespace.
///
/// Boundaries are the Unicode sentence boundaries of UAX #29 (from
/// `unicode-segmentation`), so a sentence ends after `.`, `!`, `?` and the
/// other sentence terminals, including any closing quotes or brackets that
/// follow them (`2014." Later` splits after the quote), but not before a
/// lowercase word (`see fig. two`). Two changes adapt UAX #29 to running
/// text:
///
/// - A single line break does not end a sentence; text is often hard
///   wrapped. A break that UAX #29 places only because of a line break is
///   kept when the line break is part of a blank line. Otherwise it is kept
///   only where UAX #29 would also break if the line break were a space.
/// - A period after a known English abbreviation (`Mr.`, `e.g.`, `U.S.`,
///   single-letter initials) does not end a sentence, unless the next word
///   is a common sentence opener such as "The", "We" or "However" (`the
///   U.S. The` splits, `U.S. Senator` does not). Both word lists and the
///   rule come from the sentencex segmenter (Wikimedia; MIT License);
///   abbreviations are matched lowercased, openers case-sensitively.
pub fn sentence_ranges(text: &str) -> Vec<Range<usize>> {
    use unicode_segmentation::UnicodeSegmentation;
    let mut out = Vec::new();
    let mut start = 0;
    for (i, chunk) in text.split_sentence_bound_indices() {
        let end = i + chunk.len();
        if end < text.len() && !keep_break(&text[start..end], &text[end..]) {
            continue;
        }
        let body = text[start..end].trim();
        if !body.is_empty() {
            let lead = text[start..end].len() - text[start..end].trim_start().len();
            out.push(start + lead..start + lead + body.len());
        }
        start = end;
    }
    out
}

/// Whether a UAX #29 sentence break between `before` and `after` is kept.
fn keep_break(before: &str, after: &str) -> bool {
    use unicode_segmentation::UnicodeSegmentation;
    let body = before.trim_end();
    let gap = &before[body.len()..];
    let line_breaks = gap
        .replace("\r\n", "\n")
        .chars()
        .filter(|&c| is_line_break(c))
        .count();
    if line_breaks >= 2 {
        return true;
    }
    if line_breaks > 0 {
        // Would UAX #29 break here if every line break were a space?
        let flow = |c: char| if is_line_break(c) { ' ' } else { c };
        let mut tail: Vec<char> = body.chars().rev().take(32).map(flow).collect();
        tail.reverse();
        let tail: String = tail.into_iter().collect();
        let next: String = after.trim_start().chars().take(32).map(flow).collect();
        let flowed = format!("{tail} {next}");
        let cut = tail.len() + 1;
        if !flowed.split_sentence_bound_indices().any(|(i, _)| i == cut) {
            return false;
        }
    }
    !ends_with_abbreviation(body) || starts_sentence(after)
}

/// Reads one of the vendored word lists: a word per line, `#` comments.
fn word_list(list: &'static str) -> impl Iterator<Item = &'static str> {
    list.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
}

/// True when the first word of `after` is a sentence starter such as "The",
/// "We" or "However" (case-sensitive).
fn starts_sentence(after: &str) -> bool {
    thread_local! {
        static STARTERS: HashSet<&'static str> =
            word_list(include_str!("sentence_starters_en.txt")).collect();
    }
    let after = after.trim_start();
    let end = after
        .find(|c: char| c.is_whitespace() || matches!(c, ',' | '.' | '!' | '?'))
        .unwrap_or(after.len());
    end > 0 && STARTERS.with(|s| s.contains(&after[..end]))
}

/// UAX #29's paragraph separators (classes CR, LF and Sep).
fn is_line_break(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// True when `body` ends with a period directly after a known abbreviation.
fn ends_with_abbreviation(body: &str) -> bool {
    thread_local! {
        static ABBREVIATIONS: HashSet<String> = word_list(include_str!("abbreviations_en.txt"))
            .map(str::to_lowercase)
            .collect();
    }
    let Some(head) = body.strip_suffix('.') else {
        return false;
    };
    // The last whitespace-separated word, without opening quotes or brackets.
    let word = head
        .rsplit(char::is_whitespace)
        .next()
        .unwrap_or("")
        .trim_start_matches(|c: char| !c.is_alphanumeric());
    !word.is_empty()
        && word.chars().all(|c| c.is_alphanumeric() || c == '.')
        && ABBREVIATIONS.with(|a| a.contains(&word.to_lowercase()))
}

/// Byte ranges of the tokens of `text`, skipping whitespace.
///
/// Tokens are the Unicode word boundaries of UAX #29 (from
/// `unicode-segmentation`): "don't", "3.5", "v3.5", "e.g" and "U.S" are
/// single words, and every punctuation character is its own token. One
/// change: words joined by a single `-` with no space around it stay one
/// token ("state-of-the-art"), where UAX #29 would split at each hyphen.
pub fn word_ranges(text: &str) -> Vec<Range<usize>> {
    use unicode_segmentation::UnicodeSegmentation;
    let is_word = |r: &Range<usize>| text[r.clone()].chars().any(char::is_alphanumeric);
    let mut out: Vec<Range<usize>> = Vec::new();
    for (i, segment) in text.split_word_bound_indices() {
        if segment.chars().all(char::is_whitespace) {
            continue;
        }
        let range = i..i + segment.len();
        if let [.., left, hyphen] = out.as_slice() {
            let joined = is_word(&range)
                && &text[hyphen.clone()] == "-"
                && hyphen.end == range.start
                && left.end == hyphen.start
                && is_word(left);
            if joined {
                let start = left.start;
                out.truncate(out.len() - 2);
                out.push(start..range.end);
                continue;
            }
        }
        out.push(range);
    }
    out
}

/// A tokenized document: sentences of tokens.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Document {
    sentences: Vec<Vec<Token>>,
    /// Byte range of each sentence in the source text (from_text only).
    spans: Vec<std::ops::Range<usize>>,
}

impl Document {
    /// Tokenize, split into sentences and stem untagged text.
    ///
    /// Sentences follow [`sentence_ranges`]; words follow [`word_ranges`].
    /// [`Document::sentence_spans`] gives each sentence's byte range in
    /// `text`, from its first token to its last non-space character.
    pub fn from_text(text: &str) -> Document {
        let mut sentences = Vec::new();
        let mut spans = Vec::new();
        for range in sentence_ranges(text) {
            let words = word_ranges(&text[range.clone()]);
            let Some(first) = words.first() else {
                continue;
            };
            spans.push(range.start + first.start..range.end);
            sentences.push(
                words
                    .into_iter()
                    .map(|w| Token::new(&text[range.start + w.start..range.start + w.end], None))
                    .collect(),
            );
        }
        Document { sentences, spans }
    }

    /// Build from the caller's tokenizer and tagger: sentences of
    /// `(word, tag)` pairs. Stems are computed with [`stem`].
    pub fn from_tagged<S: AsRef<str>>(sentences: &[Vec<(S, Pos)>]) -> Document {
        Document {
            spans: Vec::new(),
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
        Document {
            sentences,
            spans: Vec::new(),
        }
    }

    /// Byte range of each sentence in the text given to
    /// [`Document::from_text`]; empty for documents built from tokens.
    pub fn sentence_spans(&self) -> &[std::ops::Range<usize>] {
        &self.spans
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
}
