//! Keyphrase extraction quality on the Inspec test split (Hulth 2003).
//!
//! Fetch the data first with `scripts/fetch_inspec.sh`, then run
//! `cargo run --release --example eval_inspec`. Without the data the example
//! prints how to get it and exits successfully. TopicalPageRank also needs
//! topic distributions from `scripts/export_inspec_lda.py`; without them its
//! row is skipped.
//!
//! Scoring: for each abstract, the top k extracted phrases are compared with
//! the indexer-assigned keyphrases that occur in the text. Both sides are
//! lowercased and every run of non-alphanumeric characters becomes one space.
//! The "exact" columns compare those strings; the "stemmed" columns also stem
//! every word (English Snowball stemmer), the stemmed exact match pke and
//! most papers report. Precision, recall and F1 are averaged over documents.
//!
//! All methods run on untagged text, so the graph methods use their
//! stopword-delimited candidates rather than noun phrases. ExpandRank gets
//! each abstract's 5 nearest abstracts by TF-IDF cosine (as in its paper);
//! CollabRank gets the abstract and the same 5 neighbours as its cluster.
//! TF-IDF (corpus), KP-Miner and SGRank take document frequencies counted
//! over the 500 test abstracts; Tomokiyo-Hurst takes the same abstracts'
//! word counts as its background. EmbedRank and WordAttractionRank need
//! phrase or word vectors from an embedding model and are not run here.

use distil::text::{stem, Document};
use distil::tomokiyohurst::Background;
use distil::{
    default_stopwords, CollabRank, CommunityCluster, DocumentFrequency, ExpandRank, FirstPhrases,
    KeyCluster, KeywordExtractor, KpMiner, MultipartiteRank, PositionRank, RakeExtractor, SgRank,
    SingleRank, TextRank, TfIdf, TomokiyoHurst, TopicRank, TopicalPageRank, Yake,
};
use std::collections::{HashMap, HashSet};

const DATA: &str = "data/inspec/test.tsv";
const LDA_DOCS: &str = "data/inspec/lda_docs.tsv";
const LDA_WORDS: &str = "data/inspec/lda_words.tsv";
const NEIGHBOURS: usize = 5;

/// Each document's ranked phrases with scores.
type Ranked = Vec<Vec<(String, f64)>>;

fn normalize(phrase: &str) -> String {
    phrase
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn stemmed(phrase: &str) -> String {
    normalize(phrase)
        .split(' ')
        .map(stem)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Mean precision, recall and F1 at `k`, given each document's ranked
/// phrases and gold set, both already passed through the same `key`.
/// Phrases that become equal under the key count once.
fn score(predicted: &[Vec<String>], gold: &[HashSet<String>], k: usize) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for (pred, gold) in predicted.iter().zip(gold) {
        let top: HashSet<&String> = pred.iter().take(k).collect();
        let hits = top.iter().filter(|p| gold.contains(**p)).count() as f64;
        let p = if top.is_empty() {
            0.0
        } else {
            hits / top.len() as f64
        };
        let r = if gold.is_empty() {
            0.0
        } else {
            hits / gold.len() as f64
        };
        let f = if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        };
        sum[0] += p;
        sum[1] += r;
        sum[2] += f;
    }
    sum.map(|s| s / predicted.len() as f64)
}

/// The `k` most similar other documents by TF-IDF cosine, with similarities.
fn neighbours(docs: &[Document], k: usize) -> Vec<Vec<(usize, f64)>> {
    let stop = default_stopwords();
    let tfs: Vec<HashMap<String, f64>> = docs
        .iter()
        .map(|d| {
            let mut tf = HashMap::new();
            for t in d.tokens() {
                if distil::text::is_content(t, &stop) {
                    *tf.entry(t.stem.clone()).or_insert(0.0) += 1.0;
                }
            }
            tf
        })
        .collect();
    let mut df: HashMap<&str, f64> = HashMap::new();
    for tf in &tfs {
        for w in tf.keys() {
            *df.entry(w).or_insert(0.0) += 1.0;
        }
    }
    let n = docs.len() as f64;
    let vecs: Vec<HashMap<&str, f64>> = tfs
        .iter()
        .map(|tf| {
            let v: HashMap<&str, f64> = tf
                .iter()
                .map(|(w, c)| (w.as_str(), c * (n / df[w.as_str()]).ln()))
                .collect();
            let norm = v.values().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            v.into_iter().map(|(w, x)| (w, x / norm)).collect()
        })
        .collect();
    (0..docs.len())
        .map(|i| {
            let mut sims: Vec<(usize, f64)> = (0..docs.len())
                .filter(|&j| j != i)
                .map(|j| {
                    let s = vecs[i]
                        .iter()
                        .filter_map(|(w, x)| vecs[j].get(w).map(|y| x * y))
                        .sum();
                    (j, s)
                })
                .collect();
            sims.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            sims.truncate(k);
            sims
        })
        .collect()
}

fn parse_floats(s: &str) -> Vec<f64> {
    s.split(' ').filter_map(|x| x.parse().ok()).collect()
}

/// Per-document P(Z|d) and the word-topic map, if exported.
#[allow(clippy::type_complexity)]
fn topic_model() -> Option<(Vec<Vec<f64>>, HashMap<String, Vec<f64>>)> {
    let docs = std::fs::read_to_string(LDA_DOCS).ok()?;
    let words = std::fs::read_to_string(LDA_WORDS).ok()?;
    let docs = docs.lines().map(parse_floats).collect();
    let words = words
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(w, v)| (w.to_string(), parse_floats(v)))
        .collect();
    Some((docs, words))
}

fn main() {
    let Ok(raw) = std::fs::read_to_string(DATA) else {
        println!("{DATA} not found; run scripts/fetch_inspec.sh first");
        return;
    };
    let rows: Vec<(String, Vec<String>)> = raw
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(text, keys)| {
            let gold = keys.split('|').map(str::to_string).collect();
            (text.to_string(), gold)
        })
        .collect();
    let texts: Vec<&str> = rows.iter().map(|(t, _)| t.as_str()).collect();
    let docs: Vec<Document> = texts.iter().map(|t| Document::from_text(t)).collect();
    let near = neighbours(&docs, NEIGHBOURS);
    let topics = topic_model();
    if topics.is_none() {
        println!("{LDA_DOCS} not found; TopicalPageRank skipped (scripts/export_inspec_lda.py)");
    }

    let k = 10;
    let by_text =
        |e: &dyn KeywordExtractor| -> Ranked { texts.iter().map(|t| e.extract(t, k)).collect() };
    let mut runs: Vec<(&str, Ranked)> = vec![
        ("RAKE", by_text(&RakeExtractor::new())),
        ("TextRank", by_text(&TextRank::new())),
        ("SingleRank", by_text(&SingleRank::new())),
        ("PositionRank", by_text(&PositionRank::new())),
        ("TopicRank", by_text(&TopicRank::new())),
        ("MultipartiteRank", by_text(&MultipartiteRank::new())),
    ];
    let er = ExpandRank::new();
    let expand = docs
        .iter()
        .zip(&near)
        .map(|(d, nn)| {
            let nn: Vec<(&Document, f64)> = nn.iter().map(|&(j, s)| (&docs[j], s)).collect();
            er.rank(d, &nn, k)
        })
        .collect();
    runs.push(("ExpandRank", expand));
    let cr = CollabRank::new();
    let collab = docs
        .iter()
        .zip(&near)
        .map(|(d, nn)| {
            let mut cluster = vec![d.clone()];
            cluster.extend(nn.iter().map(|&(j, _)| docs[j].clone()));
            cr.rank_cluster(&cluster, k).swap_remove(0)
        })
        .collect();
    runs.push(("CollabRank", collab));
    let df = DocumentFrequency::from_documents(&docs, &default_stopwords(), 5);
    let bg = Background::from_documents(&docs);
    let each =
        |f: &dyn Fn(&Document) -> Vec<(String, f64)>| -> Ranked { docs.iter().map(f).collect() };
    runs.push(("TF-IDF", each(&|d| TfIdf::new().rank(d, &df, k))));
    runs.push(("KP-Miner", each(&|d| KpMiner::new().rank(d, &df, k))));
    runs.push((
        "KP-Miner lasf=1",
        each(&|d| KpMiner::new().with_lasf(1).rank(d, &df, k)),
    ));
    runs.push(("YAKE", each(&|d| Yake::new().rank(d, k))));
    runs.push(("FirstPhrases", by_text(&FirstPhrases::new())));
    runs.push((
        "SGRank",
        each(&|d| SgRank::new().rank_with(d, Some(&df), k)),
    ));
    runs.push(("KeyCluster", each(&|d| KeyCluster::new().rank(d, k))));
    runs.push((
        "CommunityCluster",
        each(&|d| CommunityCluster::new().rank(d, k)),
    ));
    runs.push((
        "Tomokiyo-Hurst",
        each(&|d| TomokiyoHurst::new().rank(d, &bg, k)),
    ));
    if let Some((doc_topics, word_topics)) = &topics {
        let tpr = TopicalPageRank::new();
        let ranked = docs
            .iter()
            .zip(doc_topics)
            .map(|(d, dt)| tpr.rank(d, word_topics, dt, k))
            .collect();
        runs.push(("TopicalPageRank", ranked));
    }

    println!(
        "Inspec test, {} abstracts; exact match after normalization | stemmed exact match",
        docs.len()
    );
    println!(
        "{:<17} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "",
        "P@5",
        "R@5",
        "F@5",
        "P@10",
        "R@10",
        "F@10",
        "P@5",
        "R@5",
        "F@5",
        "P@10",
        "R@10",
        "F@10"
    );
    for (name, ranked) in &runs {
        let mut cells = Vec::new();
        for key in [normalize, stemmed] {
            let gold: Vec<HashSet<String>> = rows
                .iter()
                .map(|(_, g)| g.iter().map(|p| key(p)).filter(|p| !p.is_empty()).collect())
                .collect();
            let pred: Vec<Vec<String>> = ranked
                .iter()
                .map(|r| r.iter().map(|(p, _)| key(p)).collect())
                .collect();
            for k in [5, 10] {
                cells.extend(score(&pred, &gold, k));
            }
        }
        let c: Vec<String> = cells.iter().map(|x| format!("{x:>5.3}")).collect();
        println!("{name:<17} {} | {}", c[..6].join(" "), c[6..].join(" "));
    }
}
