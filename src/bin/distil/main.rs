//! `distil`: keyphrases and extractive summaries from files, URLs and stdin.

mod input;

use clap::error::ErrorKind;
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use distil::summarize::{
    Centroid, Edmundson, KlSum, Lead, LexRank, Lsa, LsaVariant, Luhn, Mmr, Random, Reduction,
    SumBasic,
};
use distil::text::{is_content_in, Document, Token};
use distil::{
    stopwords, CollabRank, ExpandRank, FirstPhrases, KeywordExtractor, KpMiner, Language,
    LanguagePolicy, MultipartiteRank, PositionRank, RakeExtractor, SgRank, SingleRank, TextRank,
    TfIdf, TopicRank, TopicalPageRank, Yake,
};
use distil::{Budget, SentenceExtractor};
use std::collections::HashMap;
use std::io::{self, Write};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "distil",
    version,
    about = "Keyphrase extraction and extractive summarization from files, URLs and stdin"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Rank the keyphrases of each input.
    Keywords(KeywordsArgs),
    /// Pick the most important sentences of each input, in document order.
    Summarize(SummarizeArgs),
    /// Print the text extracted from each input.
    Text(InputArgs),
    /// List the methods: name, kind (keyphrase or sentence), description.
    Methods,
}

#[derive(Args)]
struct InputArgs {
    /// Files, http(s) URLs, or `-` for stdin (the default). HTML, PDF, DOCX,
    /// EPUB, RTF, XLSX and PPTX are converted to text.
    inputs: Vec<String>,
    /// Strip HTML tags instead of extracting the main content with
    /// readability. Navigation, footer and aside elements are still dropped.
    #[arg(long)]
    raw: bool,
}

#[derive(Args)]
struct KeywordsArgs {
    #[command(flatten)]
    input: InputArgs,
    /// Keyphrase method (see `distil methods`).
    #[arg(short, long, value_enum, default_value_t = Method::PositionRank)]
    method: Method,
    /// Number of keyphrases per input.
    #[arg(short = 'k', long = "top", default_value_t = 10)]
    top: usize,
    /// Language as an ISO 639-1 code (en, de, fr, es, ...; an unknown code
    /// lists them): its stopwords and Snowball stemmer. `auto` detects one
    /// language per input; `auto-sentence` detects each sentence's language
    /// (an input's language for sentences too short to call). Detection is
    /// whatlang's best guess; English when there is none.
    #[arg(long, default_value = "en")]
    lang: String,
    /// Word-topic distributions for topical-page-rank: one line per word,
    /// the word, a tab, then P(w|z) for each topic separated by spaces.
    /// Lines starting with `#` are skipped.
    #[arg(long, value_name = "FILE")]
    topics: Option<String>,
    /// One JSON object per input per line:
    /// {"source", "method", "keyphrases": [{"phrase", "score"}]}.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct SummarizeArgs {
    #[command(flatten)]
    input: InputArgs,
    /// Sentence method (see `distil methods`).
    #[arg(short, long, value_enum, default_value_t = SentenceMethod::LexRank)]
    method: SentenceMethod,
    /// Number of sentences per input.
    #[arg(
        short = 'n',
        long = "sentences",
        default_value_t = 3,
        conflicts_with = "words"
    )]
    sentences: usize,
    /// Word budget per input instead of a sentence count.
    #[arg(long)]
    words: Option<usize>,
    /// Language as an ISO 639-1 code (en, de, fr, es, ...; an unknown code
    /// lists them): its stopwords and Snowball stemmer. `auto` detects one
    /// language per input; `auto-sentence` detects each sentence's language
    /// (an input's language for sentences too short to call). Detection is
    /// whatlang's best guess; English when there is none.
    #[arg(long, default_value = "en")]
    lang: String,
    /// One JSON object per input per line:
    /// {"source", "method", "sentences": [{"index", "score", "text"}]}.
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SentenceMethod {
    #[value(alias = "lexrank")]
    LexRank,
    #[value(name = "lex-rank-continuous", alias = "lexrank-continuous")]
    LexRankContinuous,
    #[value(name = "text-rank", alias = "textrank")]
    TextRank,
    Lead,
    Random,
    Luhn,
    #[value(alias = "sumbasic")]
    SumBasic,
    #[value(name = "kl-sum", alias = "klsum")]
    KlSum,
    Edmundson,
    Lsa,
    #[value(name = "lsa-gong-liu")]
    LsaGongLiu,
    Reduction,
    Mmr,
    Centroid,
}

impl SentenceMethod {
    fn name(self) -> String {
        self.to_possible_value()
            .map(|v| v.get_name().to_string())
            .unwrap_or_default()
    }

    fn description(self) -> &'static str {
        match self {
            SentenceMethod::LexRank => "random walk on a thresholded tf-idf cosine graph (default)",
            SentenceMethod::LexRankContinuous => "random walk on the tf-idf cosine graph",
            SentenceMethod::TextRank => "PageRank on a shared-word sentence graph",
            SentenceMethod::Lead => "the first sentences",
            SentenceMethod::Random => "random sentences, seed 0",
            SentenceMethod::Luhn => "dense clusters of frequent words",
            SentenceMethod::SumBasic => {
                "greedy by average word probability, covered words discounted"
            }
            SentenceMethod::KlSum => "greedy, word distribution closest to the document's",
            SentenceMethod::Edmundson => "position in the text (no title, cue or key words)",
            SentenceMethod::Lsa => "length in the leading singular vectors",
            SentenceMethod::LsaGongLiu => "one sentence per leading singular vector",
            SentenceMethod::Reduction => "sum of shared-word similarities",
            SentenceMethod::Mmr => "relevance to the document minus redundancy",
            SentenceMethod::Centroid => "centroid words, position and first-sentence overlap",
        }
    }

    fn extractor(self, lang: Language) -> Box<dyn SentenceExtractor> {
        match self {
            SentenceMethod::LexRank => Box::new(LexRank::new().with_language(lang)),
            SentenceMethod::LexRankContinuous => {
                Box::new(LexRank::new().continuous().with_language(lang))
            }
            SentenceMethod::TextRank => {
                Box::new(distil::summarize::TextRank::new().with_language(lang))
            }
            SentenceMethod::Lead => Box::new(Lead::new()),
            SentenceMethod::Random => Box::new(Random::new()),
            SentenceMethod::Luhn => Box::new(Luhn::new().with_language(lang)),
            SentenceMethod::SumBasic => Box::new(SumBasic::new().with_language(lang)),
            SentenceMethod::KlSum => Box::new(KlSum::new().with_language(lang)),
            SentenceMethod::Edmundson => Box::new(Edmundson::new().with_language(lang)),
            SentenceMethod::Lsa => Box::new(Lsa::new().with_language(lang)),
            SentenceMethod::LsaGongLiu => Box::new(
                Lsa::new()
                    .with_language(lang)
                    .with_variant(LsaVariant::GongLiu),
            ),
            SentenceMethod::Reduction => Box::new(Reduction::new().with_language(lang)),
            SentenceMethod::Mmr => Box::new(Mmr::new().with_language(lang)),
            SentenceMethod::Centroid => Box::new(Centroid::new().with_language(lang)),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Method {
    #[value(alias = "positionrank")]
    PositionRank,
    #[value(alias = "textrank")]
    TextRank,
    #[value(alias = "singlerank")]
    SingleRank,
    #[value(alias = "expandrank")]
    ExpandRank,
    #[value(alias = "collabrank")]
    CollabRank,
    #[value(alias = "topicrank")]
    TopicRank,
    #[value(alias = "multipartiterank")]
    MultipartiteRank,
    #[value(alias = "topicalpagerank")]
    TopicalPageRank,
    #[value(alias = "sgrank")]
    SgRank,
    #[value(alias = "firstphrases")]
    FirstPhrases,
    Rake,
    Yake,
    #[value(alias = "tfidf")]
    TfIdf,
    #[value(alias = "kpminer")]
    KpMiner,
}

impl Method {
    fn name(self) -> String {
        self.to_possible_value()
            .map(|v| v.get_name().to_string())
            .unwrap_or_default()
    }

    fn description(self) -> &'static str {
        match self {
            Method::PositionRank => "word graph biased toward early words (default)",
            Method::TextRank => "word co-occurrence graph, window 2, merged top words",
            Method::SingleRank => "weighted word co-occurrence graph, window 10",
            Method::ExpandRank => "SingleRank graph extended with the other inputs",
            Method::CollabRank => "one SingleRank graph over all inputs as a cluster",
            Method::TopicRank => "graph of topics, clusters of similar candidates",
            Method::MultipartiteRank => "multipartite graph of candidates by topic",
            Method::TopicalPageRank => "word graph biased by a topic model (needs --topics)",
            Method::SgRank => "statistical scores refined by a candidate graph",
            Method::FirstPhrases => "earliest noun phrases (baseline)",
            Method::Rake => "phrases between stopwords, frequency times degree",
            Method::Yake => "YAKE: casing, position, frequency and spread of terms",
            Method::TfIdf => "TF-IDF over the inputs (needs two or more inputs)",
            Method::KpMiner => {
                "TF-IDF of phrases seen 3+ times, compounds boosted (two or more inputs)"
            }
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Methods => print_methods(),
        Command::Text(args) => text(&args),
        Command::Keywords(args) => keywords(&args),
        Command::Summarize(args) => summarize(&args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Io(e)) if e.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(Failure::Io(e)) => {
            eprintln!("distil: {e}");
            ExitCode::FAILURE
        }
        Err(Failure::Input { source, message }) => {
            eprintln!("distil: {source}: {message}");
            ExitCode::FAILURE
        }
    }
}

/// A runtime error; usage errors exit through clap.
enum Failure {
    Io(io::Error),
    Input { source: String, message: String },
}

impl From<io::Error> for Failure {
    fn from(e: io::Error) -> Self {
        Failure::Io(e)
    }
}

/// Exit with status 2 and clap's usage formatting.
fn usage_error(kind: ErrorKind, message: impl std::fmt::Display) -> ! {
    Cli::command().error(kind, message).exit()
}

fn print_methods() -> Result<(), Failure> {
    let mut out = io::stdout().lock();
    for m in Method::value_variants() {
        writeln!(out, "{}\tkeyphrase\t{}", m.name(), m.description())?;
    }
    for m in SentenceMethod::value_variants() {
        writeln!(out, "{}\tsentence\t{}", m.name(), m.description())?;
    }
    Ok(())
}

/// The language policy for `--lang`, or a usage error.
fn policy(lang: &str) -> LanguagePolicy {
    match lang {
        "auto" => LanguagePolicy::Detect,
        "auto-sentence" => LanguagePolicy::DetectPerSentence {
            fallback: Language::english(),
        },
        code => LanguagePolicy::Fixed(Language::from_code(code).unwrap_or_else(|| {
            usage_error(
                ErrorKind::InvalidValue,
                format!(
                    "unsupported --lang {code:?}; use auto, auto-sentence or one of {}",
                    stopwords::languages().join(", ")
                ),
            )
        })),
    }
}

/// The methods' own language: the fixed one, or English when documents
/// carry detected languages (their sentences then use their own lists).
fn method_language(policy: &LanguagePolicy) -> Language {
    match policy {
        LanguagePolicy::Fixed(lang) => lang.clone(),
        _ => Language::english(),
    }
}

/// Each text as a document under `policy`. Says on stderr when detection
/// found no language for a whole input.
fn documents(texts: &[String], sources: &[String], policy: &LanguagePolicy) -> Vec<Document> {
    if !matches!(policy, LanguagePolicy::Fixed(_)) {
        for (source, text) in sources.iter().zip(texts) {
            if textprep::language::detect(text).is_none() {
                eprintln!("distil: {source}: no supported language detected; using en");
            }
        }
    }
    texts
        .iter()
        .map(|t| Document::from_text_with(t, policy))
        .collect()
}

/// Tokens that are neither punctuation nor stopwords of their sentence.
fn content_tokens<'a>(doc: &'a Document, lang: &'a Language) -> impl Iterator<Item = &'a Token> {
    let langs = doc.sentence_languages();
    doc.sentences()
        .iter()
        .enumerate()
        .flat_map(move |(i, s)| s.iter().map(move |t| (t, langs.get(i))))
        .filter(move |(t, l)| is_content_in(t, lang, *l))
        .map(|(t, _)| t)
}

fn summarize(args: &SummarizeArgs) -> Result<(), Failure> {
    let policy = policy(&args.lang);
    let budget = match args.words {
        Some(w) => Budget::Words(w),
        None => Budget::Sentences(args.sentences),
    };
    let sources = sources(&args.input);
    let texts = read_all(&sources, args.input.raw)?;
    let method = args.method.extractor(method_language(&policy));
    let docs = documents(&texts, &sources, &policy);
    let mut out = io::stdout().lock();
    for (i, ((source, text), doc)) in sources.iter().zip(&texts).zip(&docs).enumerate() {
        let scores: HashMap<usize, f64> = method.rank(doc).into_iter().collect();
        let picked: Vec<(usize, f64, String)> = method
            .summarize(doc, budget)
            .into_iter()
            .map(|k| {
                let span = doc.sentence_spans()[k].clone();
                let text = text[span].split_whitespace().collect::<Vec<_>>().join(" ");
                (k, scores[&k], text)
            })
            .collect();
        if args.json {
            let items: Vec<String> = picked
                .iter()
                .map(|(k, s, t)| {
                    format!(
                        "{{\"index\":{k},\"score\":{},\"text\":{}}}",
                        json_number(*s),
                        json_string(t)
                    )
                })
                .collect();
            writeln!(
                out,
                "{{\"source\":{},\"method\":{},\"sentences\":[{}]}}",
                json_string(source),
                json_string(&args.method.name()),
                items.join(",")
            )?;
            continue;
        }
        if sources.len() > 1 {
            if i > 0 {
                writeln!(out)?;
            }
            writeln!(out, "== {source}")?;
        }
        for (_, _, t) in &picked {
            writeln!(out, "{t}")?;
        }
    }
    Ok(())
}

fn json_number(x: f64) -> String {
    if x.is_finite() {
        x.to_string()
    } else {
        "null".to_string()
    }
}

/// Sources to read: the arguments, or stdin when there are none.
fn sources(args: &InputArgs) -> Vec<String> {
    if args.inputs.is_empty() {
        return vec!["-".to_string()];
    }
    if args.inputs.iter().filter(|s| *s == "-").count() > 1 {
        usage_error(
            ErrorKind::ArgumentConflict,
            "stdin (`-`) can be read only once",
        );
    }
    args.inputs.clone()
}

fn read_all(sources: &[String], raw: bool) -> Result<Vec<String>, Failure> {
    sources
        .iter()
        .map(|s| {
            input::read_text(s, raw).map_err(|message| Failure::Input {
                source: s.clone(),
                message,
            })
        })
        .collect()
}

fn text(args: &InputArgs) -> Result<(), Failure> {
    let sources = sources(args);
    let texts = read_all(&sources, args.raw)?;
    let mut out = io::stdout().lock();
    for (i, (source, text)) in sources.iter().zip(&texts).enumerate() {
        if sources.len() > 1 {
            if i > 0 {
                writeln!(out)?;
            }
            writeln!(out, "== {source}")?;
        }
        writeln!(out, "{}", text.trim_end())?;
    }
    Ok(())
}

fn keywords(args: &KeywordsArgs) -> Result<(), Failure> {
    let policy = policy(&args.lang);
    let lang = method_language(&policy);
    let topics = match (&args.topics, args.method) {
        (Some(path), Method::TopicalPageRank) => Some(read_topics(path, &lang)?),
        (None, Method::TopicalPageRank) => usage_error(
            ErrorKind::MissingRequiredArgument,
            "topical-page-rank needs --topics <FILE>: one line per word, the word, a tab, \
             then P(w|z) for each topic separated by spaces",
        ),
        (Some(_), _) => usage_error(
            ErrorKind::ArgumentConflict,
            "--topics is used only by topical-page-rank",
        ),
        (None, _) => None,
    };

    let sources = sources(&args.input);
    if matches!(args.method, Method::TfIdf | Method::KpMiner) && sources.len() < 2 {
        usage_error(
            ErrorKind::TooFewValues,
            format!(
                "{} needs two or more inputs to estimate document frequencies",
                args.method.name()
            ),
        );
    }
    let texts = read_all(&sources, args.input.raw)?;
    let docs = documents(&texts, &sources, &policy);
    let fixed = matches!(policy, LanguagePolicy::Fixed(_));
    let ranked = rank(
        args.method,
        &texts,
        &docs,
        fixed,
        args.top,
        lang,
        topics.as_ref(),
    );

    let mut out = io::stdout().lock();
    for (i, (source, phrases)) in sources.iter().zip(&ranked).enumerate() {
        if args.json {
            writeln!(out, "{}", json_line(source, args.method, phrases))?;
            continue;
        }
        if sources.len() > 1 {
            if i > 0 {
                writeln!(out)?;
            }
            writeln!(out, "== {source}")?;
        }
        for (phrase, score) in phrases {
            writeln!(out, "{score:.4}\t{phrase}")?;
        }
    }
    Ok(())
}

/// Keyphrases of each text. ExpandRank uses the other texts as neighbours
/// and CollabRank all texts as one cluster. With a fixed language,
/// single-text methods read the text directly (`extract`); otherwise they
/// rank the documents, whose sentences carry detected languages.
fn rank(
    method: Method,
    texts: &[String],
    docs: &[Document],
    fixed: bool,
    k: usize,
    lang: Language,
    topics: Option<&HashMap<String, Vec<f64>>>,
) -> Vec<Vec<(String, f64)>> {
    let each = |f: &dyn Fn(&Document) -> Vec<(String, f64)>| docs.iter().map(f).collect();
    let single: Box<dyn KeywordExtractor> = match method {
        Method::PositionRank => {
            let m = PositionRank::new().with_language(lang);
            return each(&|d| m.rank(d, k));
        }
        Method::TextRank => {
            let m = TextRank::new().with_language(lang);
            return each(&|d| m.rank(d, k));
        }
        Method::SingleRank => {
            let m = SingleRank::new().with_language(lang);
            return each(&|d| m.rank(d, k));
        }
        Method::TopicRank => {
            let m = TopicRank::new().with_language(lang);
            return each(&|d| m.rank(d, k));
        }
        Method::MultipartiteRank => {
            let m = MultipartiteRank::new().with_language(lang);
            return each(&|d| m.rank(d, k));
        }
        Method::SgRank => {
            let m = SgRank::new().with_language(lang);
            return each(&|d| m.rank(d, k));
        }
        Method::FirstPhrases => {
            let m = FirstPhrases::new().with_language(lang);
            return each(&|d| m.rank(d, k));
        }
        Method::Rake if !fixed => {
            let m = RakeExtractor::new().with_language(lang);
            return each(&|d| m.rank(d, k));
        }
        Method::Rake => Box::new(RakeExtractor::new().with_language(lang)),
        Method::Yake => Box::new(Yake::new().with_language(lang)),
        Method::TfIdf => {
            let tfidf = TfIdf::new().with_language(lang);
            let df = tfidf.document_frequency(docs);
            return each(&|d| tfidf.rank(d, &df, k));
        }
        Method::KpMiner => {
            let kpminer = KpMiner::new().with_language(lang);
            let df = kpminer.document_frequency(docs);
            return each(&|d| kpminer.rank(d, &df, k));
        }
        Method::ExpandRank => {
            let sims = similarities(docs, &lang);
            let er = ExpandRank::new().with_language(lang);
            return docs
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let near: Vec<(&Document, f64)> = (0..docs.len())
                        .filter(|&j| j != i)
                        .map(|j| (&docs[j], sims[i][j]))
                        .collect();
                    er.rank(d, &near, k)
                })
                .collect();
        }
        Method::CollabRank => {
            return CollabRank::new().with_language(lang).rank_cluster(docs, k);
        }
        Method::TopicalPageRank => {
            let word_topics = topics.expect("checked before reading inputs");
            let tpr = TopicalPageRank::new().with_language(lang.clone());
            return each(&|d| tpr.rank(d, word_topics, &fold_in(d, word_topics, &lang), k));
        }
    };
    if fixed {
        return texts.iter().map(|t| single.extract(t, k)).collect();
    }
    // YAKE: its document ranking returns raw scores (lower is better);
    // `extract` reports 1 / (1 + S).
    let yake = Yake::new();
    each(&|d| {
        yake.rank(d, k)
            .into_iter()
            .map(|(p, h)| (p, 1.0 / (1.0 + h)))
            .collect()
    })
}

/// Pairwise TF-IDF cosine similarity of the documents' content stems, with
/// smoothed IDF `ln((1 + n) / (1 + df)) + 1` so that two or three inputs
/// sharing a word still count as similar.
fn similarities(docs: &[Document], lang: &Language) -> Vec<Vec<f64>> {
    let tfs: Vec<HashMap<&str, f64>> = docs
        .iter()
        .map(|d| {
            let mut tf = HashMap::new();
            for t in content_tokens(d, lang) {
                *tf.entry(t.stem.as_str()).or_insert(0.0) += 1.0;
            }
            tf
        })
        .collect();
    let mut df: HashMap<&str, f64> = HashMap::new();
    for tf in &tfs {
        for &w in tf.keys() {
            *df.entry(w).or_insert(0.0) += 1.0;
        }
    }
    let n = docs.len() as f64;
    let vecs: Vec<HashMap<&str, f64>> = tfs
        .iter()
        .map(|tf| {
            let v: HashMap<&str, f64> = tf
                .iter()
                .map(|(&w, c)| (w, c * (((1.0 + n) / (1.0 + df[w])).ln() + 1.0)))
                .collect();
            let norm = v.values().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            v.into_iter().map(|(w, x)| (w, x / norm)).collect()
        })
        .collect();
    vecs.iter()
        .map(|a| {
            vecs.iter()
                .map(|b| a.iter().filter_map(|(w, x)| b.get(w).map(|y| x * y)).sum())
                .collect()
        })
        .collect()
}

/// Word-topic vectors keyed by the word as written and by its stem in
/// `lang`.
fn read_topics(path: &str, lang: &Language) -> Result<HashMap<String, Vec<f64>>, Failure> {
    let fail = |message: String| Failure::Input {
        source: path.to_string(),
        message,
    };
    let raw = std::fs::read_to_string(path).map_err(|e| fail(e.to_string()))?;
    let mut out = HashMap::new();
    let mut width = None;
    for (n, line) in raw.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let bad = |what: &str| fail(format!("line {}: {what}", n + 1));
        let (word, values) = line
            .split_once('\t')
            .ok_or_else(|| bad("expected a word, a tab, then numbers"))?;
        let values: Vec<f64> = values
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()
            .map_err(|_| bad("not a number"))?;
        if values.is_empty() || values.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(bad("expected non-negative numbers"));
        }
        if *width.get_or_insert(values.len()) != values.len() {
            return Err(bad("topic count differs from the first line"));
        }
        let word = word.trim().to_lowercase();
        out.entry(lang.stem(&word))
            .or_insert_with(|| values.clone());
        out.insert(word, values);
    }
    if out.is_empty() {
        return Err(fail("no topic lines".to_string()));
    }
    Ok(out)
}

/// `P(Z|d)` estimated from the word-topic model: the sum of `P(w|Z)` over
/// the document's content words. TopicalPageRank only uses its direction.
fn fold_in(doc: &Document, word_topics: &HashMap<String, Vec<f64>>, lang: &Language) -> Vec<f64> {
    let width = word_topics.values().next().map_or(0, Vec::len);
    let mut sum = vec![0.0; width];
    for t in content_tokens(doc, lang) {
        if let Some(v) = word_topics.get(&t.stem) {
            sum.iter_mut().zip(v).for_each(|(s, x)| *s += x);
        }
    }
    sum
}

fn json_line(source: &str, method: Method, phrases: &[(String, f64)]) -> String {
    let items: Vec<String> = phrases
        .iter()
        .map(|(p, s)| {
            let score = if s.is_finite() {
                s.to_string()
            } else {
                "null".to_string()
            };
            format!("{{\"phrase\":{},\"score\":{score}}}", json_string(p))
        })
        .collect();
    format!(
        "{{\"source\":{},\"method\":{},\"keyphrases\":[{}]}}",
        json_string(source),
        json_string(&method.name()),
        items.join(",")
    )
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escapes_quotes_and_controls() {
        assert_eq!(json_string("a\"b\\c\n\u{1}é"), "\"a\\\"b\\\\c\\n\\u0001é\"");
    }

    #[test]
    fn similar_documents_score_higher() {
        let docs: Vec<Document> = [
            "graph ranking of keyphrases",
            "keyphrases ranked on a word graph",
            "baking bread at home",
        ]
        .iter()
        .map(|t| Document::from_text(t))
        .collect();
        let sims = similarities(&docs, &Language::english());
        assert!(sims[0][1] > 0.0);
        assert_eq!(sims[0][2], 0.0);
        assert!((sims[0][0] - 1.0).abs() < 1e-9);
    }
}
