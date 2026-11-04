"""Generate tests/fixtures/pke_parity.rs from pke's graph keyphrase methods.

Run from the repository root:

    uv run --python 3.12 \
      --with 'git+https://github.com/boudinfl/pke' \
      --with 'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl' \
      python -I tests/fixtures/pke_parity.py tests/fixtures/pke_parity.rs

For each text, pke tokenizes, tags and stems with spaCy en_core_web_sm and
NLTK's Porter stemmer; the fixture records those tokens so the Rust test can
feed distil exactly what pke saw. It also records every candidate's score
from TextRank, SingleRank, PositionRank, TopicRank, MultipartiteRank and
TopicalPageRank.

Two changes to pke's runs, both to make the reference exact rather than
approximate:
- networkx.pagerank is called with tol=1e-12 and max_iter=10000, so scores
  are the stationary distribution rather than wherever the iteration stopped.
- TextRank and SingleRank add 1e-8 * (first offset) to each score as a tie
  breaker; the fixture subtracts it again.

TopicalPageRank needs an LDA model. A 4-topic model is fitted on the fixture
texts (sklearn, random_state=0) and passed to pke; the fixture stores, for
each graph word in the model's vocabulary, its normalized word-topic vector,
and the document-topic vector pke computed.
"""

import importlib.metadata
import json
import sys

import networkx as nx
import numpy as np
import pke
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

TEXTS = {
    "wind": (
        "Offshore wind turbines convert the kinetic energy of wind into "
        "electricity. Large turbine blades capture more energy, but long "
        "blades bend under strong gusts. Engineers model blade fatigue with "
        "finite element simulations and validate the models against strain "
        "measurements from operating wind farms. Accurate fatigue models let "
        "operators schedule blade inspections before cracks grow."
    ),
    "retrieval": (
        "Dense retrieval systems encode queries and documents as vectors and "
        "rank documents by vector similarity. Approximate nearest neighbour "
        "search makes dense retrieval fast on large collections. Sparse "
        "retrieval with inverted indexes remains a strong baseline, and hybrid "
        "systems combine sparse scores with dense scores. Evaluation on "
        "standard retrieval benchmarks measures recall at fixed cutoffs."
    ),
    "soil": (
        "Soil moisture controls crop growth in dry regions. Farmers measure "
        "soil moisture with capacitance sensors buried at several depths. "
        "Wireless sensor networks report the readings to a central server, "
        "which estimates irrigation needs from soil moisture, weather "
        "forecasts and crop type. Precise irrigation scheduling saves water "
        "and raises crop yield."
    ),
    "compiler": (
        "An optimizing compiler translates source code into efficient machine "
        "code. Register allocation assigns program variables to a small set "
        "of hardware registers. Graph coloring algorithms solve register "
        "allocation by coloring an interference graph whose nodes are "
        "variables. When the graph needs more colors than registers, the "
        "compiler spills variables to memory."
    ),
}

_pagerank = nx.pagerank


def exact_pagerank(*args, **kwargs):
    kwargs["tol"] = 1e-12
    kwargs["max_iter"] = 10000
    return _pagerank(*args, **kwargs)


nx.pagerank = exact_pagerank


def load(cls, text):
    e = cls()
    e.load_document(input=text, language="en")
    return e


def scored(e, tie_offset=False):
    out = []
    for k, w in e.weights.items():
        c = e.candidates[k]
        if tie_offset:
            w -= c.offsets[0] * 1e-8
        out.append((" ".join(c.surface_forms[0]).lower(), float(w)))
    out.sort(key=lambda x: -x[1])
    return out


def lda_model():
    docs = []
    stoplist = None
    for text in TEXTS.values():
        e = load(pke.unsupervised.TopicalPageRank, text)
        stoplist = e.stoplist
        docs.append(" ".join(s for sent in e.sentences for s in sent.stems))
    vec = CountVectorizer(stop_words=list(stoplist))
    tf = vec.fit_transform(docs)
    model = LatentDirichletAllocation(n_components=4, random_state=0)
    model.fit(tf)
    return list(vec.get_feature_names_out()), model


def pke_revision():
    dist = importlib.metadata.distribution("pke")
    url = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = url.get("vcs_info", {}).get("commit_id", "unknown commit")
    return f"{dist.version} ({commit})"


def rust_str(s):
    return json.dumps(s, ensure_ascii=False)


def rust_f64(x):
    r = repr(float(x))
    return r if any(ch in r for ch in ".eE") else r + ".0"


def main(out_path):
    dictionary, model = lda_model()
    topic_word = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    cases = []
    for name, text in TEXTS.items():
        methods = {}
        e = load(pke.unsupervised.TextRank, text)
        sentences = [
            list(zip(s.words, s.pos, s.stems)) for s in e.sentences
        ]
        e.candidate_selection()
        e.candidate_weighting(window=2, top_percent=0.33)
        methods["textrank"] = scored(e, tie_offset=True)

        e = load(pke.unsupervised.SingleRank, text)
        e.candidate_selection()
        e.candidate_weighting(window=10)
        methods["singlerank"] = scored(e, tie_offset=True)

        e = load(pke.unsupervised.PositionRank, text)
        e.candidate_selection()
        e.candidate_weighting(window=10)
        methods["positionrank"] = scored(e)

        e = load(pke.unsupervised.TopicRank, text)
        e.candidate_selection()
        e.candidate_weighting(threshold=0.74, method="average")
        methods["topicrank"] = scored(e)
        stoplist = e.stoplist

        e = load(pke.unsupervised.MultipartiteRank, text)
        e.candidate_selection()
        e.candidate_weighting(alpha=1.1, threshold=0.74, method="average")
        methods["multipartiterank"] = scored(e)

        e = load(pke.unsupervised.TopicalPageRank, text)
        e.candidate_selection()
        e.candidate_weighting(window=10, lda_model=(dictionary, model))
        methods["topicalpagerank"] = scored(e)
        stems = [s for sent in e.sentences for s in sent.stems]
        tf = CountVectorizer(stop_words=list(stoplist), vocabulary=dictionary)
        doc_topics = model.transform(tf.fit_transform([" ".join(stems)]))[0]
        word_topics = [
            (w, topic_word[:, dictionary.index(w)])
            for w in e.graph.nodes()
            if w in dictionary
        ]

        words = {w.lower() for s in sentences for (w, _, _) in s}
        stopwords = sorted(words & set(stoplist))
        cases.append((name, sentences, stopwords, word_topics, doc_topics, methods))

    lines = [
        "// @generated by tests/fixtures/pke_parity.py from pke "
        + pke_revision()
        + "; do not edit.",
        "",
        "pub struct Case {",
        "    pub name: &'static str,",
        "    pub sentences: &'static [&'static [(&'static str, &'static str, &'static str)]],",
        "    pub stopwords: &'static [&'static str],",
        "    pub word_topics: &'static [(&'static str, &'static [f64])],",
        "    pub doc_topics: &'static [f64],",
        "    pub methods: &'static [(&'static str, &'static [(&'static str, f64)])],",
        "}",
        "",
        "pub const CASES: &[Case] = &[",
    ]
    for name, sentences, stopwords, word_topics, doc_topics, methods in cases:
        lines.append("    Case {")
        lines.append(f"        name: {rust_str(name)},")
        lines.append("        sentences: &[")
        for s in sentences:
            toks = ", ".join(
                f"({rust_str(w)}, {rust_str(p)}, {rust_str(st)})" for w, p, st in s
            )
            lines.append(f"            &[{toks}],")
        lines.append("        ],")
        lines.append(
            "        stopwords: &[" + ", ".join(rust_str(w) for w in stopwords) + "],"
        )
        lines.append("        word_topics: &[")
        for w, v in word_topics:
            lines.append(
                f"            ({rust_str(w)}, &[" + ", ".join(rust_f64(x) for x in v) + "]),"
            )
        lines.append("        ],")
        lines.append(
            "        doc_topics: &[" + ", ".join(rust_f64(x) for x in doc_topics) + "],"
        )
        lines.append("        methods: &[")
        for m, ranked in methods.items():
            lines.append(f"            ({rust_str(m)}, &[")
            for p, s in ranked:
                lines.append(f"                ({rust_str(p)}, {rust_f64(s)}),")
            lines.append("            ]),")
        lines.append("        ],")
        lines.append("    },")
    lines.append("];")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
