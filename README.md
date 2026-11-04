# distil

Keyword and keyphrase extraction.

distil implements the main unsupervised keyphrase-extraction methods: graph
ranking, statistical scoring, topic clustering, embedding similarity and
language-model scoring. Each method has its own module whose documentation
gives the paper, the equations used, the defaults and any deviation. Where a
maintained reference implementation exists (pke, the `yake` package), tests
check that distil produces the same candidates and scores from the same
tokens.

## Install

```toml
[dependencies]
distil = "0.1"
```

## Example

```rust
use distil::{KeywordExtractor, PositionRank};

let text = "Keyphrase extraction selects the phrases that best describe a document. \
            Graph-based keyphrase extraction builds a word graph and ranks words \
            with PageRank, then scores candidate phrases by their words.";
for (phrase, score) in PositionRank::new().extract(text, 5) {
    println!("{score:.3}  {phrase}");
}
```

```text
0.297  keyphrase extraction selects
0.161  word graph
0.161  ranks words
0.146  scores candidate phrases
0.123  best describe
```

Methods that need only the text implement `KeywordExtractor`. Methods that
need more (document frequencies, neighbour documents, topic distributions,
vectors, a background corpus) take it in their own `rank` functions.

## Methods

F1@10 is on the Inspec test set (500 abstracts) with plain-text input,
stemmed exact match against the keyphrases that occur in the abstract.

| Method | Family | Extra input | Checked against | F1@10 |
|---|---|---|---|---|
| `PositionRank` | graph | | pke | 0.263 |
| `ExpandRank` | graph | neighbour documents | identities with SingleRank | 0.255 |
| `CollabRank` | graph | a document cluster | identities with ExpandRank | 0.250 |
| `SingleRank` | graph | | pke | 0.248 |
| `TopicalPageRank` | graph | topic distributions | pke | 0.242 |
| `SgRank` | graph over candidates | document frequencies (optional) | paper equations | 0.242 |
| `RakeExtractor` | statistical | | unit tests | 0.239 |
| `MultipartiteRank` | graph over topics | | pke | 0.227 |
| `TopicRank` | graph over topics | | pke | 0.219 |
| `FirstPhrases` | baseline | | pke | 0.218 |
| `KeyCluster` | clustering | relatedness (optional) | paper equations | 0.199 |
| `KpMiner` | statistical | document frequencies | pke | 0.183 |
| `TomokiyoHurst` | language model | background corpus | paper equations | 0.175 |
| `Yake` | statistical | | `yake` package | 0.174 |
| `TfIdf` | statistical | document frequencies | pke | 0.161 |
| `CommunityCluster` | clustering | relatedness (optional) | paper equations | 0.158 |
| `TextRank` | graph | | pke | 0.138 |
| `EmbedRank` | embedding | phrase and document vectors | paper equations | |
| `WordAttractionRank` | graph | word vectors | paper equations | |

ExpandRank and CollabRank use each abstract's five nearest abstracts by
TF-IDF cosine. TF-IDF, KP-Miner and Tomokiyo-Hurst take their corpus
statistics from the 500 abstracts. KP-Miner runs with `with_lasf(1)`,
since the paper's default of 3 suits long documents. EmbedRank and WordAttractionRank need vectors from a
model and are not in the table. Reproduce with `scripts/fetch_inspec.sh`
and `cargo run --release --example eval_inspec`.

Published papers report higher scores for the same methods because they use
noun-phrase candidates from a part-of-speech tagger. distil selects those
candidates when given tagged tokens (`text::Document::from_tagged`); from
plain text it uses phrases between stopwords and punctuation.

## Command line

The `distil` binary is behind the `cli` feature and needs Rust 1.91 or
newer (the library builds with 1.85).

```sh
cargo install distil --features cli
```

It reads files, http(s) URLs or stdin (`-`, the default). HTML, PDF, DOCX,
EPUB, RTF, XLSX and PPTX are converted to text, and HTML pages are reduced to
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
0.0907	visited page's pagerank
0.0902	page's pagerank
0.0858	google directory pagerank
0.0849	google pagerank
0.0773	using personalized pagerank
```

- `distil keywords`: `-m/--method` (default `position-rank`; `distil
  methods` lists the 14 available), `-k/--top`, `--lang` for the stopword
  list (an ISO 639-1 code, or `auto` to detect it with whatlang), `--json`
  for one JSON object per input per line. With several
  inputs, `expand-rank` uses the others as neighbours, `collab-rank` ranks
  them as one cluster, and `tf-idf` and `kp-miner` take document
  frequencies from them; `topical-page-rank` needs `--topics <FILE>`.
- `distil text`: print the extracted text.
- `distil methods`: list the methods.

URLs are fetched with a 20 s timeout, at most 5 redirects and a 20 MB body
limit; pages that need JavaScript are not rendered. Exit status is 0 on
success, 1 when an input cannot be read or converted, and 2 for usage
errors.

## Languages

Sentences and words follow the Unicode text segmentation rules (UAX #29),
with an English abbreviation list for sentence ends, and English stopwords
by default. `stopwords::for_language` has NLTK's stopword lists for German,
French, Spanish, Portuguese, Italian, Dutch, Russian, Arabic and 24 more
(`stopwords::languages`), and every method takes them through
`with_stopwords`.
UAX #29 splits Chinese, Japanese and Thai into single characters, so
pre-tokenize such text and pass space-separated tokens.

## License

MIT OR Apache-2.0
