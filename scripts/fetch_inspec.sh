#!/bin/sh
# Fetch the Inspec test split (Hulth 2003; 500 abstracts with indexer-assigned
# keyphrases) from the midas/inspec mirror on Hugging Face and write
# data/inspec/test.tsv: one abstract per line, then a tab, then its keyphrases
# that occur in the text, separated by "|". Used by examples/eval_inspec.rs.
set -eu
cd "$(dirname "$0")/.."
mkdir -p data/inspec
curl -fsSL -o data/inspec/test.jsonl \
  https://huggingface.co/datasets/midas/inspec/resolve/main/test.jsonl
python3 -I - data/inspec/test.jsonl data/inspec/test.tsv <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
brackets = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]"}
with open(src) as f, open(dst, "w") as out:
    for line in f:
        row = json.loads(line)
        text = " ".join(brackets.get(t, t) for t in row["document"])
        keys = "|".join(row["extractive_keyphrases"])
        out.write(text.replace("\t", " ") + "\t" + keys + "\n")
PY
wc -l data/inspec/test.tsv
