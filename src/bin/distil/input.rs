//! Reading inputs (files, stdin, http(s) URLs) and extracting their text.

use deformat::detect::{decode_bytes, detect_bytes, detect_mime, detect_path};
use deformat::Format;
use std::io::Read;
use std::path::Path;
use std::time::Duration;

/// Whole-request timeout for a URL, connection to last body byte.
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);
/// Largest response body read, after decompression.
pub const MAX_BODY: u64 = 20 * 1024 * 1024;
/// Redirects followed before giving up.
const MAX_REDIRECTS: u32 = 5;

/// Raw bytes of one input and what is known about their format.
struct Loaded {
    bytes: Vec<u8>,
    /// Format from the Content-Type header or the file extension.
    hint: Format,
    /// `charset` parameter of the Content-Type header.
    charset: Option<String>,
    /// URL after redirects, for readability's link resolution.
    url: Option<String>,
}

/// Plain text of `source`: a path, `-` for stdin, or an http(s) URL.
/// With `raw`, HTML keeps all its text instead of only the main content.
pub fn read_text(source: &str, raw: bool) -> Result<String, String> {
    let loaded = load(source)?;
    extract(loaded, raw)
}

fn load(source: &str) -> Result<Loaded, String> {
    if source == "-" {
        let mut bytes = Vec::new();
        std::io::stdin()
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        return Ok(Loaded {
            bytes,
            hint: Format::Unknown,
            charset: None,
            url: None,
        });
    }
    if let Some((scheme, _)) = source.split_once("://") {
        let scheme = scheme.to_ascii_lowercase();
        if scheme == "http" || scheme == "https" {
            return fetch(source);
        }
        if !Path::new(source).exists() {
            return Err(format!(
                "unsupported URL scheme {scheme:?}; only http and https are fetched"
            ));
        }
    }
    let bytes = std::fs::read(source).map_err(|e| e.to_string())?;
    Ok(Loaded {
        bytes,
        hint: detect_path(source),
        charset: None,
        url: None,
    })
}

fn fetch(url: &str) -> Result<Loaded, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(FETCH_TIMEOUT))
        .max_redirects(MAX_REDIRECTS)
        .user_agent(concat!("distil/", env!("CARGO_PKG_VERSION")))
        .http_status_as_error(false)
        .build()
        .into();
    let response = agent.get(url).call().map_err(|e| e.to_string())?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("HTTP {status}"));
    }
    let final_url = {
        use ureq::ResponseExt;
        response.get_uri().to_string()
    };
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let mut bytes = Vec::new();
    response
        .into_body()
        .into_reader()
        .take(MAX_BODY + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_BODY {
        return Err(format!(
            "response body exceeds {} MB",
            MAX_BODY / (1024 * 1024)
        ));
    }
    let hint = content_type.as_deref().map_or(Format::Unknown, detect_mime);
    let charset = content_type.as_deref().and_then(charset_param);
    Ok(Loaded {
        bytes,
        hint,
        charset,
        url: Some(final_url),
    })
}

/// The `charset` parameter of a Content-Type value.
fn charset_param(content_type: &str) -> Option<String> {
    content_type.split(';').skip(1).find_map(|p| {
        let (k, v) = p.split_once('=')?;
        k.trim()
            .eq_ignore_ascii_case("charset")
            .then(|| v.trim().trim_matches('"').to_string())
    })
}

fn extract(loaded: Loaded, raw: bool) -> Result<String, String> {
    // Binary formats are recognized by their bytes; a text Content-Type or
    // extension cannot override that, and a missing hint falls back to it.
    let sniffed = detect_bytes(&loaded.bytes);
    let format = match (loaded.hint, sniffed) {
        (
            _,
            f @ (Format::Pdf
            | Format::Docx
            | Format::Epub
            | Format::Rtf
            | Format::Xlsx
            | Format::Pptx),
        ) => f,
        (Format::Unknown, f) => f,
        (hint, _) => hint,
    };
    let label = loaded.charset.as_deref().unwrap_or("utf-8");
    let text = |bytes: &[u8]| decode_bytes(bytes, label).into_owned();
    let extracted = match format {
        Format::Pdf => deformat::pdf::extract_bytes(&loaded.bytes),
        Format::Docx => deformat::docx::extract_bytes(&loaded.bytes),
        Format::Epub => deformat::epub::extract_bytes(&loaded.bytes),
        Format::Rtf => deformat::rtf::extract_bytes(&loaded.bytes),
        Format::Xlsx => deformat::xlsx::extract_bytes(&loaded.bytes),
        Format::Pptx => deformat::pptx::extract_bytes(&loaded.bytes),
        Format::Html if !raw => {
            return Ok(deformat::extract_readable(&text(&loaded.bytes), loaded.url.as_deref()).text)
        }
        Format::Html | Format::Xml => deformat::extract_as(&text(&loaded.bytes), format),
        _ => return Ok(text(&loaded.bytes)),
    };
    extracted.map(|e| e.text).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::charset_param;

    #[test]
    fn charset_from_content_type() {
        assert_eq!(
            charset_param("text/html; charset=\"ISO-8859-1\"").as_deref(),
            Some("ISO-8859-1")
        );
        assert_eq!(charset_param("text/html"), None);
    }
}
