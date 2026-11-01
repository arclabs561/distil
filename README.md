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

```text
16.00  systems programming language focused
1.00  performance
1.00  rust
1.00  safety
```

Four extractors share the `KeywordExtractor` trait:

- `TfIdfExtractor`: term frequency within one document (no IDF without a corpus)
- `YakeExtractor`: unsupervised, statistical; a simplified YAKE-style score, not the published algorithm
- `TextRankExtractor`: PageRank over a word co-occurrence graph (uses `graphops`); returns single words
- `RakeExtractor`: rapid automatic keyword extraction

Tokenization splits on whitespace and punctuation, with English stopwords by
default. `stopwords::for_language` returns lists for German, French, Spanish,
Portuguese, Italian, Dutch, Russian, and Arabic; `RakeExtractor` and
`YakeExtractor` take them through `with_stopwords`. Pre-tokenize CJK, Thai, or
Arabic text before extraction.

## License

MIT OR Apache-2.0
