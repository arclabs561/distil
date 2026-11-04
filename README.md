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

## Command line

The `distil` binary is behind the `cli` feature and needs Rust 1.91 or
newer (the library itself builds with 1.80). distil is not on crates.io
yet, so install from git:

```sh
cargo install --git https://github.com/arclabs561/distil --features cli distil
```

It reads files, http(s) URLs, or stdin (`-`, the default). HTML, PDF, DOCX,
EPUB, RTF, XLSX and PPTX are converted to text; HTML pages are reduced to
their main content unless `--raw` is given.

```sh
echo "Rust is a systems programming language focused on safety and performance. \
The Rust compiler checks memory safety at compile time." | distil keywords -k 4
```

```text
0.1862	rust
0.1626	compile time
0.1397	safety
0.0893	performance
```

```sh
distil keywords -k 5 https://en.wikipedia.org/wiki/PageRank
```

```text
0.0889	visited page's pagerank
0.0884	page's pagerank
0.0842	google directory pagerank
0.0833	google pagerank
0.0759	using personalized pagerank
```

- `distil keywords`: `-m/--method` (default `position-rank`), `-k/--top`,
  `--lang` for the stopword list, `--json` for one JSON object per input
  per line. With several inputs, `expand-rank` uses the others as
  neighbours and `collab-rank` ranks them as one cluster;
  `topical-page-rank` needs `--topics <FILE>` (see `distil keywords --help`).
- `distil text`: print the extracted text.
- `distil methods`: list the methods.

URLs are fetched with a 20 s timeout, at most 5 redirects and a 20 MB body
limit; pages that need JavaScript are not rendered. Exit status is 0 on
success, 1 when an input cannot be read or converted, and 2 for usage errors.

## License

MIT OR Apache-2.0
