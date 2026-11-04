//! Extractive summarization quality on a CNN/DailyMail test sample.
//!
//! Fetch the data first with `scripts/fetch_cnn_dailymail.sh` (the first
//! 500 test articles by default), then run
//! `cargo run --release --example eval_cnn_dailymail`. Without the data the
//! example prints how to get it and exits successfully.
//!
//! Every method picks 3 sentences, returned in document order, and the
//! summary is compared with the reference highlights by ROUGE-1, ROUGE-2
//! and ROUGE-L F1 (Lin 2004), averaged over articles. ROUGE is computed
//! here: text is lowercased, every run of non-alphanumeric characters
//! separates tokens, n-gram matches are clipped by the reference count, and
//! ROUGE-L uses the longest common subsequence of the whole token
//! sequences. The "stemmed" rows also stem every token (English Snowball
//! stemmer; the official scripts use Porter, which differs on a few
//! percent of word types), so numbers are close to, not identical with,
//! published `rouge-score` results.

use distil::summarize::{
    Centroid, Edmundson, KlSum, Lead, LexRank, Lsa, LsaVariant, Luhn, Mmr, Random, Reduction,
    SumBasic, TextRank,
};
use distil::text::{stem, Document};
use distil::{Budget, SentenceExtractor};
use std::collections::HashMap;

const DATA: &str = "data/cnn_dailymail/test.tsv";
const SENTENCES: usize = 3;

fn tokens(text: &str, stemmed: bool) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| if stemmed { stem(w) } else { w.to_string() })
        .collect()
}

fn f1(overlap: f64, candidate: usize, reference: usize) -> f64 {
    if overlap == 0.0 {
        return 0.0;
    }
    let p = overlap / candidate as f64;
    let r = overlap / reference as f64;
    2.0 * p * r / (p + r)
}

fn rouge_n(candidate: &[String], reference: &[String], n: usize) -> f64 {
    fn grams(t: &[String], n: usize) -> HashMap<&[String], usize> {
        let mut m = HashMap::new();
        for g in t.windows(n) {
            *m.entry(g).or_insert(0) += 1;
        }
        m
    }
    let (c, r) = (grams(candidate, n), grams(reference, n));
    let overlap: usize = c
        .iter()
        .map(|(g, &k)| k.min(r.get(g).copied().unwrap_or(0)))
        .sum();
    f1(
        overlap as f64,
        candidate.len().saturating_sub(n - 1),
        reference.len().saturating_sub(n - 1),
    )
}

fn rouge_l(candidate: &[String], reference: &[String]) -> f64 {
    let mut prev = vec![0usize; reference.len() + 1];
    for a in candidate {
        let mut row = vec![0usize; reference.len() + 1];
        for (j, b) in reference.iter().enumerate() {
            row[j + 1] = if a == b {
                prev[j] + 1
            } else {
                row[j].max(prev[j + 1])
            };
        }
        prev = row;
    }
    f1(
        prev[reference.len()] as f64,
        candidate.len(),
        reference.len(),
    )
}

/// ROUGE-1, ROUGE-2 and ROUGE-L F1.
fn rouge(candidate: &str, reference: &str, stemmed: bool) -> [f64; 3] {
    let c = tokens(candidate, stemmed);
    let r = tokens(reference, stemmed);
    [rouge_n(&c, &r, 1), rouge_n(&c, &r, 2), rouge_l(&c, &r)]
}

fn main() {
    let Ok(raw) = std::fs::read_to_string(DATA) else {
        println!("{DATA} not found; run scripts/fetch_cnn_dailymail.sh first.");
        return;
    };
    let articles: Vec<(&str, &str)> = raw.lines().filter_map(|l| l.split_once('\t')).collect();
    let methods: Vec<(&str, Box<dyn SentenceExtractor>)> = vec![
        ("Lead-3", Box::new(Lead::new())),
        ("Random (seed 0)", Box::new(Random::new())),
        ("Luhn", Box::new(Luhn::new())),
        ("SumBasic", Box::new(SumBasic::new())),
        ("KL-Sum", Box::new(KlSum::new())),
        ("Edmundson (location)", Box::new(Edmundson::new())),
        ("LSA (Steinberger-Jezek)", Box::new(Lsa::new())),
        (
            "LSA (Gong-Liu)",
            Box::new(Lsa::new().with_variant(LsaVariant::GongLiu)),
        ),
        ("Reduction", Box::new(Reduction::new())),
        ("TextRank", Box::new(TextRank::new())),
        ("LexRank (threshold 0.1)", Box::new(LexRank::new())),
        (
            "LexRank (continuous)",
            Box::new(LexRank::new().continuous()),
        ),
        ("MMR (lambda 0.7)", Box::new(Mmr::new())),
        ("Centroid (MEAD)", Box::new(Centroid::new())),
    ];
    println!(
        "{} CNN/DailyMail test articles, {SENTENCES} sentences per summary, ROUGE F1 x 100\n",
        articles.len()
    );
    println!(
        "| Method | R-1 | R-2 | R-L | R-1 stem | R-2 stem | R-L stem |\n|---|---|---|---|---|---|---|"
    );
    let docs: Vec<Document> = articles
        .iter()
        .map(|(a, _)| Document::from_text(a))
        .collect();
    for (name, method) in &methods {
        let mut sum = [0.0f64; 6];
        for ((article, reference), doc) in articles.iter().zip(&docs) {
            let picked = method.summarize(doc, Budget::Sentences(SENTENCES));
            let summary: Vec<&str> = picked
                .iter()
                .map(|&i| &article[doc.sentence_spans()[i].clone()])
                .collect();
            let summary = summary.join(" ");
            let plain = rouge(&summary, reference, false);
            let stemmed = rouge(&summary, reference, true);
            for (s, x) in sum.iter_mut().zip(plain.iter().chain(&stemmed)) {
                *s += x;
            }
        }
        let n = articles.len() as f64;
        let cells: Vec<String> = sum
            .iter()
            .map(|s| format!("{:.2}", 100.0 * s / n))
            .collect();
        println!("| {name} | {} |", cells.join(" | "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "the cat sat" against "the cat ran": 2 of 3 unigrams (F 2/3), 1 of 2
    /// bigrams (F 1/2), LCS 2 (F 2/3).
    #[test]
    fn rouge_on_a_hand_computed_pair() {
        let [r1, r2, rl] = rouge("The cat sat.", "the cat ran", false);
        assert!((r1 - 2.0 / 3.0).abs() < 1e-12);
        assert!((r2 - 0.5).abs() < 1e-12);
        assert!((rl - 2.0 / 3.0).abs() < 1e-12);
    }
}
