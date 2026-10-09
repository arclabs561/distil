# distil

Keyword and keyphrase extraction.

## Install

```toml
[dependencies]
distil = { git = "https://github.com/arclabs561/distil" }
```

## Usage

```rust
use distil::{RakeExtractor, KeywordExtractor};

let text = "Rust is a systems programming language focused on safety and performance.";
let extractor = RakeExtractor::new();
let keywords = extractor.extract(text, 5);

for (phrase, score) in &keywords {
    println!("{score:.2}  {phrase}");
}
```

Four extractors share the `KeywordExtractor` trait:

- `TfIdfExtractor`: term frequency within one document (no IDF without a corpus)
- `YakeExtractor`: unsupervised, statistical; a simplified YAKE-style score, not the published algorithm
- `TextRankExtractor`: PageRank over a word co-occurrence graph (uses `graphops`); returns single words
- `RakeExtractor`: rapid automatic keyword extraction

Stopword lists for English, German, French, Spanish, Portuguese, Italian, Dutch, Russian, and Arabic.

## License

MIT OR Apache-2.0
