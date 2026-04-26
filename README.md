# distil

Keyword and keyphrase extraction: TF-IDF, YAKE, TextRank, and RAKE.

[![crates.io](https://img.shields.io/crates/v/distil.svg)](https://crates.io/crates/distil)
[![docs.rs](https://docs.rs/distil/badge.svg)](https://docs.rs/distil)

## Install

```toml
[dependencies]
distil = "0.1"
```

## Usage

```rust
use distil::{RakeExtractor, KeywordExtractor};

let text = "Rust is a systems programming language focused on safety and performance.";
let extractor = RakeExtractor::new();
let keywords = extractor.extract(text, 5);

for (score, phrase) in &keywords {
    println!("{score:.2}  {phrase}");
}
```

Four extractors share the `KeywordExtractor` trait:

- `TfIdfExtractor` -- term frequency, inverse document frequency
- `YakeExtractor` -- unsupervised, statistical
- `TextRankExtractor` -- graph-based (uses `graphops`)
- `RakeExtractor` -- rapid automatic keyword extraction

Stopword lists for English, German, French, Spanish, Portuguese, Italian, Dutch, Russian, and Arabic.

## License

MIT OR Apache-2.0
