//! End-to-end tests of the `distil` binary. URL tests use an HTTP server on
//! 127.0.0.1 inside the test process; nothing reaches the network.
#![cfg(feature = "cli")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

const ABSTRACT: &str = "Keyphrase extraction selects the phrases that best describe a \
    document. Graph-based keyphrase extraction builds a word graph from co-occurrence \
    and ranks words with PageRank. Keyphrase extraction methods such as TextRank and \
    PositionRank need no training data.";

/// A page whose navigation and footer are boilerplate around one article.
const PAGE: &str = r#"<!doctype html>
<html><head><title>Keyphrase extraction</title></head>
<body>
<div class="sidebar"><ul><li><a href="/">Home</a></li><li><a href="/shop">Discount shoes</a></li>
<li><a href="/shop">Discount shoes sale</a></li><li><a href="/news">Newsletter signup</a></li></ul></div>
<article>
<h1>Keyphrase extraction</h1>
<p>Keyphrase extraction selects the phrases that best describe a document. Graph-based
keyphrase extraction builds a word graph from co-occurrence and ranks words with PageRank,
then scores each candidate phrase by the scores of its words.</p>
<p>PositionRank biases the random walk toward words that occur early in the document,
which suits scholarly abstracts where the title and first sentences carry the topic.
TopicRank clusters similar candidate phrases into topics and ranks the topics instead.</p>
<p>Statistical keyphrase extraction methods such as RAKE and YAKE score candidate phrases
from frequency, degree and position features, and need no word graph and no training data.</p>
</article>
<div class="footer-links">Discount shoes sale. Copyright footer text. Newsletter signup.</div>
</body></html>"#;

fn distil(args: &[&str], stdin: Option<&str>) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_distil"));
    cmd.args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Keep requests to 127.0.0.1 off any configured proxy.
    for var in [
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ] {
        cmd.env_remove(var);
    }
    let mut child = cmd.spawn().expect("run distil");
    {
        let mut pipe = child.stdin.take().unwrap();
        if let Some(text) = stdin {
            pipe.write_all(text.as_bytes()).unwrap();
        }
    }
    child.wait_with_output().unwrap()
}

fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).unwrap()
}

fn temp_file(name: &str, contents: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("distil-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, contents).unwrap();
    path
}

/// `(score, phrase)` rows of plain output.
fn rows(out: &str) -> Vec<(f64, String)> {
    out.lines()
        .filter(|l| !l.is_empty() && !l.starts_with("== "))
        .map(|l| {
            let (s, p) = l.split_once('\t').expect("score<TAB>phrase");
            (s.parse().expect("numeric score"), p.to_string())
        })
        .collect()
}

#[test]
fn stdin_is_the_default_input() {
    let o = distil(&["keywords", "-k", "3"], Some(ABSTRACT));
    assert!(o.status.success(), "{}", stderr(&o));
    let r = rows(&stdout(&o));
    assert_eq!(r.len(), 3);
    assert!(r.windows(2).all(|w| w[0].0 >= w[1].0), "ranked best first");
    assert!(r.iter().any(|(_, p)| p.contains("keyphrase extraction")));

    let dash = distil(&["keywords", "-k", "3", "-"], Some(ABSTRACT));
    assert_eq!(stdout(&dash), stdout(&o));

    // `echo` and most files end with a newline after the last period.
    let newline = distil(&["keywords", "-k", "3"], Some(&format!("{ABSTRACT}\n")));
    assert!(newline.status.success(), "{}", stderr(&newline));
    assert_eq!(stdout(&newline), stdout(&o));
}

#[test]
fn file_input_matches_stdin() {
    let path = temp_file("abstract.txt", ABSTRACT);
    let file = distil(&["keywords", path.to_str().unwrap()], None);
    let piped = distil(&["keywords"], Some(ABSTRACT));
    assert!(file.status.success(), "{}", stderr(&file));
    assert_eq!(stdout(&file), stdout(&piped));
    assert!(!stdout(&file).contains("== "), "no header for one input");
}

#[test]
fn every_listed_method_runs() {
    let o = distil(&["methods"], None);
    assert!(o.status.success());
    let listed: Vec<(String, String)> = stdout(&o)
        .lines()
        .map(|l| {
            let mut f = l.split('\t');
            (f.next().unwrap().to_string(), f.next().unwrap().to_string())
        })
        .collect();
    let names: Vec<&str> = listed
        .iter()
        .filter(|(_, kind)| kind == "keyphrase")
        .map(|(n, _)| n.as_str())
        .collect();
    assert!(names.len() >= 14);
    assert_eq!(names[0], "position-rank");
    // topical-page-rank needs --topics and tf-idf two inputs; both have their
    // own tests.
    for name in names
        .iter()
        .filter(|n| !matches!(**n, "topical-page-rank" | "tf-idf" | "kp-miner"))
    {
        let o = distil(&["keywords", "-m", name, "-k", "2"], Some(ABSTRACT));
        assert!(o.status.success(), "{name}: {}", stderr(&o));
        assert!(!rows(&stdout(&o)).is_empty(), "{name} found nothing");
    }
    let sentence: Vec<&str> = listed
        .iter()
        .filter(|(_, kind)| kind == "sentence")
        .map(|(n, _)| n.as_str())
        .collect();
    assert!(sentence.len() >= 10, "{sentence:?}");
    for name in sentence {
        let o = distil(&["summarize", "-m", name, "-n", "2"], Some(ABSTRACT));
        assert!(o.status.success(), "{name}: {}", stderr(&o));
        assert_eq!(stdout(&o).lines().count(), 2, "{name}: {}", stdout(&o));
    }
}

#[test]
fn summarize_keeps_document_order_and_budgets() {
    let o = distil(&["summarize", "-m", "lead", "-n", "2"], Some(ABSTRACT));
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(
        stdout(&o),
        "Keyphrase extraction selects the phrases that best describe a document.\n\
         Graph-based keyphrase extraction builds a word graph from co-occurrence and ranks \
         words with PageRank.\n"
    );
    // The first sentence has 11 words, the second 14: a 12-word budget
    // keeps only the first.
    let o = distil(
        &["summarize", "-m", "lead", "--words", "12"],
        Some(ABSTRACT),
    );
    assert_eq!(stdout(&o).lines().count(), 1, "{}", stdout(&o));
    let o = distil(&["summarize", "-n", "1", "--words", "5"], Some(ABSTRACT));
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn summarize_json_lists_index_score_and_text() {
    let o = distil(
        &["summarize", "--json", "-m", "lead", "-n", "1"],
        Some(ABSTRACT),
    );
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(
        stdout(&o),
        "{\"source\":\"-\",\"method\":\"lead\",\"sentences\":[{\"index\":0,\"score\":3,\
         \"text\":\"Keyphrase extraction selects the phrases that best describe a document.\"}]}\n"
    );
}

#[test]
fn html_file_readability_drops_boilerplate() {
    let path = temp_file("page.html", PAGE);
    let path = path.to_str().unwrap();
    let main = distil(&["text", path], None);
    let raw = distil(&["text", "--raw", path], None);
    assert!(main.status.success() && raw.status.success());
    let (main, raw) = (stdout(&main), stdout(&raw));
    assert!(main.contains("PositionRank biases the random walk"));
    assert!(!main.contains("<p>"));
    assert!(!main.contains("Discount shoes"), "{main}");
    assert!(raw.contains("Discount shoes"));
    assert!(raw.contains("PositionRank biases the random walk"));

    let kw = distil(&["keywords", "-k", "20", "--raw", path], None);
    assert!(stdout(&kw).contains("discount shoes"));
    let kw = distil(&["keywords", "-k", "20", path], None);
    assert!(!stdout(&kw).contains("discount shoes"));
}

#[test]
fn json_lines_have_the_documented_shape() {
    let a = temp_file("a.txt", ABSTRACT);
    let b = temp_file(
        "b \"quoted\".txt",
        "Word graphs rank words. Word graphs rank phrases.",
    );
    let o = distil(
        &[
            "keywords",
            "--json",
            "-k",
            "2",
            "-m",
            "expand-rank",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
        ],
        None,
    );
    assert!(o.status.success(), "{}", stderr(&o));
    let out = stdout(&o);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2, "one line per input");
    let src_a = a.to_str().unwrap();
    assert!(lines[0].starts_with(&format!(
        "{{\"source\":\"{src_a}\",\"method\":\"expand-rank\",\"keyphrases\":[{{\"phrase\":\""
    )));
    assert!(lines[1].contains("quoted\\\".txt\""), "escaped quotes");
    assert!(lines[0].ends_with("}]}"));
    assert_eq!(lines[0].matches("\"score\":").count(), 2);
}

#[test]
fn several_inputs_get_headers() {
    let a = temp_file("h1.txt", ABSTRACT);
    let o = distil(
        &[
            "keywords",
            "-m",
            "collab-rank",
            "-k",
            "2",
            a.to_str().unwrap(),
            "-",
        ],
        Some("Word graphs rank keyphrase candidates."),
    );
    assert!(o.status.success(), "{}", stderr(&o));
    let out = stdout(&o);
    assert!(out.starts_with(&format!("== {}\n", a.to_str().unwrap())));
    assert!(out.contains("\n\n== -\n"));
}

#[test]
fn usage_errors_exit_2() {
    let o = distil(&["keywords", "-m", "no-such-method"], Some(ABSTRACT));
    assert_eq!(o.status.code(), Some(2));
    assert!(stderr(&o).contains("no-such-method"));

    let o = distil(&["keywords", "-m", "topical-page-rank"], Some(ABSTRACT));
    assert_eq!(o.status.code(), Some(2));
    assert!(stderr(&o).contains("--topics"), "{}", stderr(&o));

    let o = distil(&["keywords", "--lang", "xx"], Some(ABSTRACT));
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn lang_auto_detects_the_stopword_language() {
    let en = distil(&["keywords", "-m", "rake"], Some(ABSTRACT));
    let auto = distil(
        &["keywords", "-m", "rake", "--lang", "auto"],
        Some(ABSTRACT),
    );
    assert_eq!(stdout(&auto), stdout(&en));

    let german = "Die Extraktion von Schlüsselwörtern wählt die Phrasen aus, die ein \
        Dokument am besten beschreiben. Graphbasierte Verfahren bauen einen Wortgraphen \
        und ordnen die Wörter mit PageRank. Die Verfahren brauchen keine Trainingsdaten.";
    let o = distil(&["keywords", "-m", "rake", "--lang", "auto"], Some(german));
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    let phrases: Vec<String> = rows(&stdout(&o)).into_iter().map(|(_, p)| p).collect();
    assert!(!phrases.is_empty());
    for p in &phrases {
        for stop in ["die", "von", "und", "mit", "ein", "keine"] {
            assert!(!p.split(' ').any(|w| w == stop), "{p:?} has {stop:?}");
        }
    }
}

#[test]
fn topical_page_rank_with_a_topic_file() {
    let topics = temp_file(
        "topics.tsv",
        "# word\tP(w|z1) P(w|z2)\nkeyphrase\t0.4 0.01\nextraction\t0.3 0.01\ngraph\t0.2 0.01\nbread\t0.01 0.9\n",
    );
    let o = distil(
        &[
            "keywords",
            "-m",
            "topical-page-rank",
            "--topics",
            topics.to_str().unwrap(),
            "-k",
            "3",
        ],
        Some(ABSTRACT),
    );
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(rows(&stdout(&o)).len(), 3);

    let bad = temp_file("bad-topics.tsv", "keyphrase 0.4 0.1\n");
    let o = distil(
        &[
            "keywords",
            "-m",
            "topical-page-rank",
            "--topics",
            bad.to_str().unwrap(),
        ],
        Some(ABSTRACT),
    );
    assert_eq!(o.status.code(), Some(1));
    assert!(stderr(&o).contains("line 1"), "{}", stderr(&o));
}

#[test]
fn missing_file_exits_1_naming_it() {
    let o = distil(&["keywords", "no/such/file.txt"], None);
    assert_eq!(o.status.code(), Some(1));
    let err = stderr(&o);
    assert!(err.starts_with("distil: no/such/file.txt: "), "{err}");
    assert_eq!(err.lines().count(), 1);
    assert!(stdout(&o).is_empty());
}

#[test]
fn other_url_schemes_are_rejected() {
    let o = distil(&["text", "ftp://example.invalid/a.txt"], None);
    assert_eq!(o.status.code(), Some(1));
    assert!(stderr(&o).contains("http and https"), "{}", stderr(&o));
}

/// Serves until the test process exits. Routes: `/page` (HTML without an
/// extension), `/ua` (echoes the User-Agent), `/loop` (redirects to itself),
/// `/big` (a body over the 20 MB cap); anything else is 404.
fn serve() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            std::thread::spawn(move || respond(stream));
        }
    });
    format!("http://{addr}")
}

fn respond(mut stream: TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let path = request_line.split(' ').nth(1).unwrap_or("/").to_string();
    let mut agent = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("user-agent:") {
            agent = v.trim().to_string();
        }
    }
    let ok = |ctype: &str, body: &[u8]| {
        let mut head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        head.extend_from_slice(body);
        head
    };
    let _ = match path.as_str() {
        "/page" => stream.write_all(&ok("text/html; charset=utf-8", PAGE.as_bytes())),
        "/ua" => stream.write_all(&ok("text/plain", agent.as_bytes())),
        "/loop" => stream.write_all(
            b"HTTP/1.1 302 Found\r\nLocation: /loop\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        ),
        "/big" => {
            let size = 21 * 1024 * 1024;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n"
            );
            let chunk = vec![b'a'; 64 * 1024];
            stream.write_all(head.as_bytes()).and_then(|_| {
                (0..size / chunk.len()).try_for_each(|_| stream.write_all(&chunk))
            })
        }
        _ => stream.write_all(
            b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found",
        ),
    };
    let _ = stream.flush();
    // Drain whatever the client still sends so closing does not reset it.
    let _ = reader.read(&mut [0; 1]);
}

#[test]
fn url_html_uses_content_type_and_readability() {
    let base = serve();
    let url = format!("{base}/page");
    let main = distil(&["text", &url], None);
    assert!(main.status.success(), "{}", stderr(&main));
    assert!(stdout(&main).contains("PositionRank biases the random walk"));
    assert!(!stdout(&main).contains("Discount shoes"));
    let raw = distil(&["text", "--raw", &url], None);
    assert!(stdout(&raw).contains("Discount shoes"));
    let kw = distil(&["keywords", "-k", "3", &url], None);
    assert!(kw.status.success(), "{}", stderr(&kw));
    assert_eq!(rows(&stdout(&kw)).len(), 3);
}

#[test]
fn url_sends_the_distil_user_agent() {
    let base = serve();
    let o = distil(&["text", &format!("{base}/ua")], None);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(
        stdout(&o).trim(),
        format!("distil/{}", env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn url_404_exits_1_naming_the_url() {
    let base = serve();
    let url = format!("{base}/missing");
    let o = distil(&["keywords", &url], None);
    assert_eq!(o.status.code(), Some(1));
    let err = stderr(&o);
    assert!(err.starts_with(&format!("distil: {url}: ")), "{err}");
    assert!(err.contains("404"), "{err}");
}

#[test]
fn url_redirect_loop_is_cut_off() {
    let base = serve();
    let o = distil(&["text", &format!("{base}/loop")], None);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        stderr(&o).to_lowercase().contains("redirect"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn url_body_over_the_cap_is_rejected() {
    let base = serve();
    let o = distil(&["text", &format!("{base}/big")], None);
    assert_eq!(o.status.code(), Some(1));
    assert!(stderr(&o).contains("exceeds 20 MB"), "{}", stderr(&o));
    assert!(stdout(&o).is_empty());
}

#[test]
fn tf_idf_needs_two_inputs_and_ranks_each() {
    let one = distil(&["keywords", "-m", "tf-idf"], Some(ABSTRACT));
    assert_eq!(one.status.code(), Some(2), "{}", stderr(&one));
    let a = temp_file("tfidf-a.txt", ABSTRACT);
    let b = temp_file(
        "tfidf-b.txt",
        "Sentence extraction selects whole sentences for a summary of a document.",
    );
    let two = distil(
        &[
            "keywords",
            "-m",
            "tf-idf",
            "-k",
            "3",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
        ],
        None,
    );
    assert!(two.status.success(), "{}", stderr(&two));
    assert_eq!(stdout(&two).matches("== ").count(), 2);
}
