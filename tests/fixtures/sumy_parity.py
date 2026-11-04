"""Generate tests/fixtures/sumy_parity.rs from sumy and the lexrank package.

Run from the repository root:

    uv run --python 3.12 --with sumy==0.13.0 --with lexrank==0.1.0 \
      python -I tests/fixtures/sumy_parity.py tests/fixtures/sumy_parity.rs

Each case is a few paragraphs of pre-split sentences plus a heading. sumy
gets them as an ObjectDocumentModel whose words come from a regular
expression with sumy's own word filter (a letter, then letters, apostrophes
or hyphens), so no NLTK data is needed; stems come from sumy's English
stemmer. The fixture records each sentence's words and stems, the stopwords
and every method's per-sentence scores, so the Rust test feeds distil
exactly what sumy saw.

sumy deviates from the papers (or from its own documentation) in a few
places. Where distil follows the paper, the reference is patched here and
the Rust test compares against the patched value:

- TextRank: the edge counts each shared word once (the paper's set
  intersection; sumy counts every repeat), sentences get no self-loops,
  the power iteration runs to 1e-12 instead of 1e-4, the zero-division
  guard (1e-7 added to every row sum) is shrunk to 1e-300, and a sentence
  with no edges jumps uniformly (sumy lets its mass leak, which changes the
  effective damping whenever the graph has an isolated sentence).
- LexRank: the power iteration runs to 1e-12 instead of 0.1.
- LSA: zero cells stay 0 (sumy turns them into 0.4, the smoothing constant
  of the augmented term frequency, which the IR book applies only to terms
  present in the document).
- SumBasic: each word of a chosen sentence is squared once, as in the
  paper (sumy squares it once per occurrence).
- KL-Sum: the summary's words are filtered and lowercased like the
  document's (sumy counts raw summary words), and stopwords are removed
  after lowercasing (sumy removes them before, keeping "The").

The truncated LSA run keeps about half the dimensions, moved past any tie
in the singular values (a tie at the cut makes the subspace ambiguous).

Unpatched: Luhn, Edmundson (cue, key, title, location), Reduction.

The lexrank package (LexRank with an idf learnt from a document
collection) runs with its own tokenizer and stopwords on the same
sentences, with the fixture texts as the collection; its power method's
convergence test is tightened from numpy.allclose defaults to 1e-14.
"""

import importlib.metadata
import json
import math
import re
import sys

import numpy
import lexrank
import lexrank.algorithms.power_method as lexrank_power
from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words

CASES = {
    "grid": (
        "Battery storage on the power grid",
        [
            [
                "Grid operators are adding large batteries to store solar power.",
                "Batteries charge at noon when solar output peaks and prices fall.",
                "In the evening the stored power flows back to the grid.",
            ],
            [
                "Lithium batteries dominate new storage projects because their prices keep falling.",
                "Engineers worry that lithium supply could limit storage growth.",
                "Sodium batteries and iron batteries are cheaper alternatives for long storage.",
                "Pumped hydro remains the largest form of grid storage worldwide.",
            ],
            [
                "Regulators now let storage owners sell power and reserve capacity.",
                "Analysts expect grid storage to triple within five years.",
            ],
        ],
    ),
    "trial": (
        "A drug trial for migraine",
        [
            [
                "A new drug cut migraine days by half in a large clinical trial.",
                "The trial enrolled two thousand adults with frequent migraine attacks.",
            ],
            [
                "Patients took the drug or a placebo every day for twelve weeks.",
                "Headache diaries recorded the number of migraine days each month.",
                "Side effects were mild and similar in both groups.",
                "Nausea was the most common complaint among patients on the drug.",
            ],
            [
                "The company will ask regulators to approve the drug next year.",
                "Doctors caution that the trial was short and the drug is expensive.",
                "Longer studies must show whether the benefit lasts.",
            ],
        ],
    ),
    "compiler": (
        "Faster builds with incremental compilation",
        [
            [
                "Incremental compilation reuses work from earlier builds.",
                "The compiler stores a dependency graph of queries between builds.",
                "When a source file changes, only queries that depend on it run again.",
            ],
            [
                "Large projects see the biggest gains from incremental builds.",
                "A clean build of the project takes ten minutes on a laptop.",
                "After a small edit the incremental build finishes in seconds.",
            ],
            [
                "Caching the dependency graph on disk costs space but saves time.",
                "Hardly anyone disables incremental compilation in debug builds.",
            ],
        ],
    ),
}

BONUS = ["important", "largest", "significant", "biggest"]
STIGMA = ["hardly", "impossible"]
WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|['-])*")


class RegexTokenizer:
    """Word tokenizer with sumy's word filter, no NLTK data needed."""

    def to_words(self, text):
        return tuple(WORD.findall(text))


TOKENIZER = RegexTokenizer()
STEMMER = Stemmer("english")


def build(heading, paragraphs, with_heading):
    out = []
    for k, para in enumerate(paragraphs):
        sentences = [Sentence(s, TOKENIZER) for s in para]
        if with_heading and k == 0:
            sentences.insert(0, Sentence(heading, TOKENIZER, is_heading=True))
        out.append(Paragraph(sentences))
    return ObjectDocumentModel(out)


def patch_sumy():
    def edge(self_or_words1, words1, words2=None):
        # Called as a staticmethod: (words1, words2).
        a, b = (self_or_words1, words1) if words2 is None else (words1, words2)
        if a is b:
            return 0.0
        rank = len(set(a) & set(b))
        if rank == 0:
            return 0.0
        norm = math.log(len(a)) + math.log(len(b))
        return float(rank) if numpy.isclose(norm, 0.0) else rank / norm

    TextRankSummarizer._rate_sentences_edge = staticmethod(lambda a, b: edge(a, b))

    create_matrix = TextRankSummarizer._create_matrix

    def uniform_dangling_rows(self, document):
        matrix = create_matrix(self, document)
        n = len(matrix)
        for row in matrix:
            if numpy.allclose(row, (1.0 - self.damping) / n, rtol=0.0, atol=1e-15):
                row[:] = 1.0 / n
        return matrix

    TextRankSummarizer._create_matrix = uniform_dangling_rows
    TextRankSummarizer.epsilon = 1e-12
    TextRankSummarizer._ZERO_DIVISION_PREVENTION = 1e-300
    LexRankSummarizer.epsilon = 1e-12

    def term_frequency(self, matrix, smooth=0.4):
        max_word_frequencies = numpy.max(matrix, axis=0)
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                if max_word_frequencies[col] != 0 and matrix[row, col] != 0:
                    frequency = matrix[row, col] / max_word_frequencies[col]
                    matrix[row, col] = smooth + (1.0 - smooth) * frequency
        return matrix

    LsaSummarizer._compute_term_frequency = term_frequency

    def update_tf(word_freq, words_to_update):
        for w in set(words_to_update):
            word_freq[w] *= word_freq[w]
        return word_freq

    SumBasicSummarizer._update_tf = staticmethod(update_tf)

    def kl_words(self, sentences):
        words = [self.normalize_word(w) for s in sentences for w in s.words]
        return [w for w in words if w not in self.stop_words]

    KLSummarizer._get_all_words_in_doc = kl_words
    KLSummarizer._get_all_content_words_in_doc = kl_words


def patch_lexrank():
    class TightNumpy:
        def __getattr__(self, name):
            return getattr(numpy, name)

        @staticmethod
        def allclose(a, b):
            return numpy.allclose(a, b, rtol=0.0, atol=1e-14)

    lexrank_power.np = TightNumpy()


def lsa_ranks(summarizer, document):
    dictionary = summarizer._create_dictionary(document)
    matrix = summarizer._create_matrix(document, dictionary)
    matrix = summarizer._compute_term_frequency(matrix)
    _u, sigma, v = numpy.linalg.svd(matrix, full_matrices=False)
    dims = max(LsaSummarizer.MIN_DIMENSIONS, int(len(sigma) * LsaSummarizer.REDUCTION_RATIO))
    return summarizer._compute_ranks(sigma, v), min(dims, len(sigma))


def lexrank_sumy(summarizer, document):
    words = [summarizer._to_words_set(s) for s in document.sentences]
    tf = summarizer._compute_tf(words)
    idf = summarizer._compute_idf(words)
    matrix = summarizer._create_matrix(words, summarizer.threshold, tf, idf)
    return list(summarizer.power_method(matrix, summarizer.epsilon)), idf


def per_sentence(document, ratings):
    return [ratings[s] for s in document.sentences]


def order(document, ratings):
    """sumy's greedy ratings are -(selection step); return the selection order."""
    sentences = list(document.sentences)
    return sorted(range(len(sentences)), key=lambda i: -ratings[sentences[i]])


def sumy_case(heading, paragraphs, stop):
    plain = build(heading, paragraphs, False)
    titled = build(heading, paragraphs, True)
    scores, orders = {}, {}

    luhn = LuhnSummarizer(STEMMER)
    luhn.stop_words = stop
    significant = luhn._get_significant_words(plain.words)
    scores["luhn"] = [luhn.rate_sentence(s, significant) for s in plain.sentences]

    sb = SumBasicSummarizer(STEMMER)
    sb.stop_words = stop
    orders["sumbasic"] = order(plain, sb._compute_ratings(plain.sentences))

    kl = KLSummarizer(STEMMER)
    kl.stop_words = frozenset(w.lower() for w in stop)
    orders["kl"] = order(plain, kl._compute_ratings(plain.sentences))

    ed = EdmundsonSummarizer(STEMMER, cue_weight=1.0, key_weight=1.0, title_weight=1.0, location_weight=1.0)
    ed.bonus_words = BONUS
    ed.stigma_words = STIGMA
    ed.null_words = stop
    total = [0.0] * len(titled.sentences)
    for build_method in (ed._build_cue_method_instance, ed._build_key_method_instance,
                         ed._build_title_method_instance, ed._build_location_method_instance):
        r = build_method().rate_sentences(titled)
        total = [t + r[s] for t, s in zip(total, titled.sentences)]
    scores["edmundson"] = total

    lsa = LsaSummarizer(STEMMER)
    lsa.stop_words = stop
    LsaSummarizer.REDUCTION_RATIO = 1.0
    scores["lsa"], _ = lsa_ranks(lsa, plain)
    # About half the dimensions, moved up past any tie in the singular
    # values so that the truncated subspace is unique.
    matrix = lsa._compute_term_frequency(lsa._create_matrix(plain, lsa._create_dictionary(plain)))
    sigma = numpy.linalg.svd(matrix, compute_uv=False)
    k = max(LsaSummarizer.MIN_DIMENSIONS, len(sigma) // 2)
    while k < len(sigma) and sigma[k - 1] - sigma[k] < 1e-6:
        k += 1
    LsaSummarizer.REDUCTION_RATIO = (k + 0.5) / len(sigma)
    scores["lsa_half"], half_dims = lsa_ranks(lsa, plain)
    LsaSummarizer.REDUCTION_RATIO = 1.0

    red = ReductionSummarizer(STEMMER)
    red.stop_words = stop
    scores["reduction"] = per_sentence(plain, red.rate_sentences(plain))

    tr = TextRankSummarizer(STEMMER)
    tr.stop_words = stop
    scores["textrank"] = per_sentence(plain, tr.rate_sentences(plain))

    lr = LexRankSummarizer(STEMMER)
    lr.stop_words = stop
    scores["lexrank"], idf = lexrank_sumy(lr, plain)

    words = [[(w, STEMMER(w.lower())) for w in s.words] for s in plain.sentences]
    heading_words = [(w, STEMMER(w.lower())) for w in TOKENIZER.to_words(heading)]
    sizes = [len(p) for p in paragraphs]
    return {
        "words": words,
        "heading": heading_words,
        "paragraphs": sizes,
        "scores": scores,
        "orders": orders,
        "idf": idf,
        "lsa_half_dims": half_dims,
    }


def lexrank_cases(stop):
    documents = [[s for p in paragraphs for s in p] for _, paragraphs in CASES.values()]
    lxr = lexrank.LexRank(documents, stopwords=stop)
    out = {}
    for name, sentences in zip(CASES, documents):
        tokens = [lxr.tokenize_sentence(s) for s in sentences]
        vocab = sorted({t for ts in tokens for t in ts})
        out[name] = {
            "tokens": tokens,
            "idf": {t: lxr.idf_score[t] for t in vocab},
            "threshold": list(lxr.rank_sentences(sentences, threshold=0.1, fast_power_method=False)),
            "continuous": list(lxr.rank_sentences(sentences, threshold=None, fast_power_method=False)),
        }
    return out, math.log(len(documents) + 1)


def rust_str(s):
    return json.dumps(s, ensure_ascii=False)


def rust_f64(x):
    r = repr(float(x))
    return r if any(ch in r for ch in ".eE") else r + ".0"


def floats(xs):
    return "&[" + ", ".join(rust_f64(x) for x in xs) + "]"


def pairs(ws):
    return "&[" + ", ".join(f"({rust_str(w)}, {rust_str(s)})" for w, s in ws) + "]"


def main(out_path):
    patch_sumy()
    patch_lexrank()
    stop = sorted(get_stop_words("english"))
    lexrank_stop = sorted(lexrank.STOPWORDS["en"])
    lr_cases, lr_unseen = lexrank_cases(set(lexrank_stop))
    versions = ", ".join(f"{p} {importlib.metadata.version(p)}" for p in ("sumy", "lexrank", "nltk", "numpy"))

    lines = [
        f"// @generated by tests/fixtures/sumy_parity.py from {versions}; do not edit.",
        "",
        "pub struct Case {",
        "    pub name: &'static str,",
        "    pub sentences: &'static [&'static [(&'static str, &'static str)]],",
        "    pub heading: &'static [(&'static str, &'static str)],",
        "    pub paragraphs: &'static [usize],",
        "    pub scores: &'static [(&'static str, &'static [f64])],",
        "    pub orders: &'static [(&'static str, &'static [usize])],",
        "    pub sumy_idf: &'static [(&'static str, f64)],",
        "    pub lsa_half_dimensions: usize,",
        "}",
        "",
        "pub struct LexRankCase {",
        "    pub name: &'static str,",
        "    pub tokens: &'static [&'static [&'static str]],",
        "    pub idf: &'static [(&'static str, f64)],",
        "    pub threshold: &'static [f64],",
        "    pub continuous: &'static [f64],",
        "}",
        "",
        "pub const STOPWORDS: &[&str] = &[" + ", ".join(rust_str(w) for w in stop) + "];",
        "pub const BONUS: &[&str] = &[" + ", ".join(rust_str(w) for w in BONUS) + "];",
        "pub const STIGMA: &[&str] = &[" + ", ".join(rust_str(w) for w in STIGMA) + "];",
        "pub const LEXRANK_STOPWORDS: &[&str] = &[" + ", ".join(rust_str(w) for w in lexrank_stop) + "];",
        f"pub const LEXRANK_UNSEEN_IDF: f64 = {rust_f64(lr_unseen)};",
        "",
        "pub const CASES: &[Case] = &[",
    ]
    for name, (heading, paragraphs) in CASES.items():
        c = sumy_case(heading, paragraphs, stop)
        lines.append("    Case {")
        lines.append(f"        name: {rust_str(name)},")
        lines.append("        sentences: &[")
        for ws in c["words"]:
            lines.append(f"            {pairs(ws)},")
        lines.append("        ],")
        lines.append(f"        heading: {pairs(c['heading'])},")
        lines.append("        paragraphs: &[" + ", ".join(str(n) for n in c["paragraphs"]) + "],")
        lines.append("        scores: &[")
        for m, xs in c["scores"].items():
            lines.append(f"            ({rust_str(m)}, {floats(xs)}),")
        lines.append("        ],")
        lines.append("        orders: &[")
        for m, xs in c["orders"].items():
            lines.append(f"            ({rust_str(m)}, &[" + ", ".join(str(i) for i in xs) + "]),")
        lines.append("        ],")
        lines.append("        sumy_idf: &[")
        for w in sorted(c["idf"]):
            lines.append(f"            ({rust_str(w)}, {rust_f64(c['idf'][w])}),")
        lines.append("        ],")
        lines.append(f"        lsa_half_dimensions: {c['lsa_half_dims']},")
        lines.append("    },")
    lines.append("];")
    lines.append("")
    lines.append("pub const LEXRANK_CASES: &[LexRankCase] = &[")
    for name, c in lr_cases.items():
        lines.append("    LexRankCase {")
        lines.append(f"        name: {rust_str(name)},")
        lines.append("        tokens: &[")
        for ts in c["tokens"]:
            lines.append("            &[" + ", ".join(rust_str(t) for t in ts) + "],")
        lines.append("        ],")
        lines.append("        idf: &[")
        for w, x in c["idf"].items():
            lines.append(f"            ({rust_str(w)}, {rust_f64(x)}),")
        lines.append("        ],")
        lines.append(f"        threshold: {floats(c['threshold'])},")
        lines.append(f"        continuous: {floats(c['continuous'])},")
        lines.append("    },")
    lines.append("];")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
