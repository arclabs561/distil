"""Export topic distributions for TopicalPageRank on Inspec.

Uses the LDA model bundled with pke (1,000 topics trained on SemEval-2010)
and writes, next to data/inspec/test.tsv:

- lda_docs.tsv: one line per abstract, P(Z|d) as space-separated numbers,
  computed as pke's TopicalPageRank does (spaCy tokens, NLTK Porter stems,
  pke's stoplist, the model's vocabulary).
- lda_words.tsv: for each word of the abstracts, its English Snowball
  (Porter2) stem, a tab, and P(w|Z) of the Porter stem pke's model knows it
  by. distil stems with Porter2, so the file is keyed the same way.

Run after scripts/fetch_inspec.sh, from the repository root:

    uv run --python 3.12 \
      --with 'git+https://github.com/boudinfl/pke' \
      --with 'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl' \
      python -I scripts/export_inspec_lda.py data/inspec
"""

import os
import sys

import numpy as np
import pke
import spacy
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


def main(data_dir):
    model_path = os.path.join(
        os.path.dirname(pke.__file__), "models", "lda-1000-semeval2010.py3.pickle.gz"
    )
    dictionary, model = pke.utils.load_lda_model(model_path)
    index = {w: i for i, w in enumerate(dictionary)}
    topic_word = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    porter2 = SnowballStemmer("english")
    nlp = spacy.load("en_core_web_sm")
    words = {}
    with open(os.path.join(data_dir, "test.tsv")) as f, open(
        os.path.join(data_dir, "lda_docs.tsv"), "w"
    ) as docs:
        for line in f:
            text = line.split("\t", 1)[0]
            e = pke.unsupervised.TopicalPageRank()
            e.load_document(input=text, language="en", spacy_model=nlp)
            stems = []
            for s in e.sentences:
                for w, st in zip(s.words, s.stems):
                    stems.append(st)
                    if st in index and w.isalpha():
                        words.setdefault(porter2.stem(w.lower()), index[st])
            vec = CountVectorizer(stop_words=list(e.stoplist), vocabulary=dictionary)
            dist = model.transform(vec.fit_transform([" ".join(stems)]))[0]
            docs.write(" ".join(f"{x:.6g}" for x in dist) + "\n")
    with open(os.path.join(data_dir, "lda_words.tsv"), "w") as out:
        for w, i in sorted(words.items()):
            out.write(w + "\t" + " ".join(f"{x:.6g}" for x in topic_word[:, i]) + "\n")
    print(f"{len(words)} words")


if __name__ == "__main__":
    main(sys.argv[1])
