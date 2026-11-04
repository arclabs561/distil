//! `distil`: keyphrases from files, URLs and stdin.

mod input;

use clap::error::ErrorKind;
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use distil::text::{is_content, stem, Document};
use distil::{
    stopwords, CollabRank, DocumentFrequency, ExpandRank, FirstPhrases, KeywordExtractor, KpMiner,
    MultipartiteRank, PositionRank, RakeExtractor, SgRank, SingleRank, TextRank, TfIdf, TopicRank,
    TopicalPageRank, Yake,
};
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "distil",
    version,
    about = "Keyphrase extraction from files, URLs and stdin"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Rank the keyphrases of each input.
    Keywords(KeywordsArgs),
    /// Print the text extracted from each input.
    Text(InputArgs),
    /// List the keyphrase methods.
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
    /// Stopword language: en, de, fr, es, pt, it, nl, ru or ar. Stemming
    /// stays English.
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
        writeln!(out, "{}\t{}", m.name(), m.description())?;
    }
    Ok(())
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
    let Some(stop) = stopwords::for_language(&args.lang) else {
        usage_error(
            ErrorKind::InvalidValue,
            format!(
                "unsupported --lang {:?}; use en, de, fr, es, pt, it, nl, ru or ar",
                args.lang
            ),
        );
    };
    let topics = match (&args.topics, args.method) {
        (Some(path), Method::TopicalPageRank) => Some(read_topics(path)?),
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
    let ranked = rank(args.method, &texts, args.top, stop, topics.as_ref());

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
/// and CollabRank all texts as one cluster.
fn rank(
    method: Method,
    texts: &[String],
    k: usize,
    stop: HashSet<String>,
    topics: Option<&HashMap<String, Vec<f64>>>,
) -> Vec<Vec<(String, f64)>> {
    let docs = || -> Vec<Document> { texts.iter().map(|t| Document::from_text(t)).collect() };
    let single: Box<dyn KeywordExtractor> = match method {
        Method::PositionRank => Box::new(PositionRank::new().with_stopwords(stop)),
        Method::TextRank => Box::new(TextRank::new().with_stopwords(stop)),
        Method::SingleRank => Box::new(SingleRank::new().with_stopwords(stop)),
        Method::TopicRank => Box::new(TopicRank::new().with_stopwords(stop)),
        Method::MultipartiteRank => Box::new(MultipartiteRank::new().with_stopwords(stop)),
        Method::Rake => Box::new(RakeExtractor::new().with_stopwords(stop)),
        Method::Yake => Box::new(Yake::new().with_stopwords(stop)),
        Method::TfIdf => {
            let docs = docs();
            let df = DocumentFrequency::from_documents(&docs, &stop, 3);
            let tfidf = TfIdf::new().with_stopwords(stop);
            return docs.iter().map(|d| tfidf.rank(d, &df, k)).collect();
        }
        Method::KpMiner => {
            let docs = docs();
            let df = DocumentFrequency::from_documents(&docs, &stop, 5);
            let kpminer = KpMiner::new().with_stopwords(stop);
            return docs.iter().map(|d| kpminer.rank(d, &df, k)).collect();
        }
        Method::SgRank => Box::new(SgRank::new().with_stopwords(stop)),
        Method::FirstPhrases => Box::new(FirstPhrases::new().with_stopwords(stop)),
        Method::ExpandRank => {
            let docs = docs();
            let sims = similarities(&docs, &stop);
            let er = ExpandRank::new().with_stopwords(stop);
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
            return CollabRank::new()
                .with_stopwords(stop)
                .rank_cluster(&docs(), k);
        }
        Method::TopicalPageRank => {
            let word_topics = topics.expect("checked before reading inputs");
            let tpr = TopicalPageRank::new().with_stopwords(stop.clone());
            return docs()
                .iter()
                .map(|d| tpr.rank(d, word_topics, &fold_in(d, word_topics, &stop), k))
                .collect();
        }
    };
    texts.iter().map(|t| single.extract(t, k)).collect()
}

/// Pairwise TF-IDF cosine similarity of the documents' content stems, with
/// smoothed IDF `ln((1 + n) / (1 + df)) + 1` so that two or three inputs
/// sharing a word still count as similar.
fn similarities(docs: &[Document], stop: &HashSet<String>) -> Vec<Vec<f64>> {
    let tfs: Vec<HashMap<&str, f64>> = docs
        .iter()
        .map(|d| {
            let mut tf = HashMap::new();
            for t in d.tokens().filter(|t| is_content(t, stop)) {
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

/// Word-topic vectors keyed by the word as written and by its stem.
fn read_topics(path: &str) -> Result<HashMap<String, Vec<f64>>, Failure> {
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
        out.entry(stem(&word)).or_insert_with(|| values.clone());
        out.insert(word, values);
    }
    if out.is_empty() {
        return Err(fail("no topic lines".to_string()));
    }
    Ok(out)
}

/// `P(Z|d)` estimated from the word-topic model: the sum of `P(w|Z)` over
/// the document's content words. TopicalPageRank only uses its direction.
fn fold_in(
    doc: &Document,
    word_topics: &HashMap<String, Vec<f64>>,
    stop: &HashSet<String>,
) -> Vec<f64> {
    let width = word_topics.values().next().map_or(0, Vec::len);
    let mut sum = vec![0.0; width];
    for t in doc.tokens().filter(|t| is_content(t, stop)) {
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
        let sims = similarities(&docs, &distil::default_stopwords());
        assert!(sims[0][1] > 0.0);
        assert_eq!(sims[0][2], 0.0);
        assert!((sims[0][0] - 1.0).abs() < 1e-9);
    }
}
