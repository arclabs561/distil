//! Keyphrase extraction quality on the Inspec test split (Hulth 2003).
//!
//! Fetch the data first with `scripts/fetch_inspec.sh`, then run
//! `cargo run --release --example eval_inspec`. Without the data the example
//! prints how to get it and exits successfully.
//!
//! Scoring: for each abstract, the top k extracted phrases are compared with
//! the indexer-assigned keyphrases that occur in the text. Both sides are
//! lowercased and every run of non-alphanumeric characters becomes one space;
//! there is no stemming, so this is stricter than the stemmed exact match
//! many papers report. Precision, recall and F1 are averaged over documents.

use distil::{KeywordExtractor, RakeExtractor, TextRankExtractor, TfIdfExtractor, YakeExtractor};
use std::collections::HashSet;

const DATA: &str = "data/inspec/test.tsv";

fn normalize(phrase: &str) -> String {
    phrase
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Mean precision, recall and F1 at `k` over all documents.
fn score(
    extractor: &dyn KeywordExtractor,
    docs: &[(String, HashSet<String>)],
    k: usize,
) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for (text, gold) in docs {
        let predicted: HashSet<String> = extractor
            .extract(text, k)
            .into_iter()
            .map(|(phrase, _)| normalize(&phrase))
            .collect();
        let hits = predicted.intersection(gold).count() as f64;
        let p = if predicted.is_empty() {
            0.0
        } else {
            hits / predicted.len() as f64
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
    sum.map(|s| s / docs.len() as f64)
}

fn main() {
    let Ok(raw) = std::fs::read_to_string(DATA) else {
        println!("{DATA} not found; run scripts/fetch_inspec.sh first");
        return;
    };
    let docs: Vec<(String, HashSet<String>)> = raw
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(text, keys)| {
            let gold = keys
                .split('|')
                .map(normalize)
                .filter(|k| !k.is_empty())
                .collect();
            (text.to_string(), gold)
        })
        .collect();

    let extractors: [(&str, Box<dyn KeywordExtractor>); 4] = [
        ("RAKE", Box::new(RakeExtractor::new())),
        ("YAKE", Box::new(YakeExtractor::new())),
        ("TextRank", Box::new(TextRankExtractor::new())),
        ("TF-IDF", Box::new(TfIdfExtractor::new())),
    ];

    println!(
        "Inspec test, {} abstracts, exact match after normalization",
        docs.len()
    );
    println!(
        "{:<10} {:>6} {:>6} {:>6} {:>7} {:>7} {:>7}",
        "", "P@5", "R@5", "F1@5", "P@10", "R@10", "F1@10"
    );
    for (name, extractor) in &extractors {
        let [p5, r5, f5] = score(extractor.as_ref(), &docs, 5);
        let [p10, r10, f10] = score(extractor.as_ref(), &docs, 10);
        println!("{name:<10} {p5:>6.3} {r5:>6.3} {f5:>6.3} {p10:>7.3} {r10:>7.3} {f10:>7.3}");
    }
}
