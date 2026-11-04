#!/bin/sh
# Fetch the first 500 articles of the CNN/DailyMail 3.0.0 test split
# (Hermann et al. 2015; See et al. 2017) through the Hugging Face dataset
# viewer API and write data/cnn_dailymail/test.tsv: one article per line,
# then a tab, then its reference highlights (whitespace collapsed). Used by
# examples/eval_cnn_dailymail.rs.
set -eu
cd "$(dirname "$0")/.."
mkdir -p data/cnn_dailymail
n=${1:-500}
python3 -I - "$n" data/cnn_dailymail/test.tsv <<'PY'
import json, sys, urllib.request
n, dst = int(sys.argv[1]), sys.argv[2]
base = ("https://datasets-server.huggingface.co/rows?dataset=abisee/cnn_dailymail"
        "&config=3.0.0&split=test")
rows = []
while len(rows) < n:
    url = f"{base}&offset={len(rows)}&length={min(100, n - len(rows))}"
    req = urllib.request.Request(url, headers={"User-Agent": "distil-dev"})
    with urllib.request.urlopen(req, timeout=60) as r:
        page = json.load(r)["rows"]
    if not page:
        break
    rows.extend(p["row"] for p in page)
with open(dst, "w") as out:
    for row in rows:
        article = " ".join(row["article"].split())
        highlights = " ".join(row["highlights"].split())
        out.write(article + "\t" + highlights + "\n")
PY
wc -l data/cnn_dailymail/test.tsv
