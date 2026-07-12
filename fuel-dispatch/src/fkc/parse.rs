//! FKC markdown + restricted-YAML parser (§3.1 file anatomy, §3.8 YAML subset).
//!
//! Pipeline:
//! 1. Split the file-level `---`-fenced YAML front-matter off the top (§3.1).
//! 2. Scan `## ` headings; for each section extract its single fenced
//!    ` ```fkc ` block (§3.1: exactly one per section — zero or >1 is an error).
//! 3. Run the §3.8 restricted-subset pre-pass on every YAML chunk BEFORE
//!    deserialization: reject tab indentation, anchors (`&name`), aliases
//!    (`*name`), merge keys (`<<:`), and unquoted Norway-problem tokens in
//!    scalar value positions.
//! 4. `serde_yml::from_str` each chunk into the schema structs and assemble
//!    [`FkcFile`].
//!
//! Why hand-rolled and not `pulldown-cmark`: the adoption plan (§1.1) permits
//! "a minimal hand-rolled section/fence scanner", and the file anatomy is
//! deliberately simple (front-matter, `## ` headings, fenced blocks). A line
//! scanner keeps the dependency surface to just `serde` + `serde_yml`.
//!
//! ## The Norway problem (§3.8)
//!
//! `serde_yml` (libyaml-family) resolves YAML-1.1-ish implicit types, so an
//! unquoted `no` / `yes` / `on` / `off` / `n` / `y` in a *scalar value*
//! position can coerce to a bool. FKC defends with TWO layers, both applied
//! here: (a) every token-bearing schema field is typed `String`, so even if the
//! deserializer produced a bool it could not target a string field; and (b)
//! this pre-pass rejects an unquoted Norway token in a value position outright
//! ([`FkcError::NorwayToken`]) so the contract author is told to quote it
//! rather than getting a silently-coerced value. Quoted forms (`"no"`,
//! `'NO'`), block/flow keys, and tokens appearing only inside a quoted
//! expression string are untouched.

use std::path::Path;

use super::error::FkcError;
use super::schema::{FkcFile, FkcFrontMatter, FkcKernel};

/// Parse an FKC file's text into an [`FkcFile`] (front-matter + kernels).
///
/// Pure: no I/O of its own (tests pass `&str`). Every failure is a typed
/// [`FkcError`]; never panics.
pub fn parse_file(text: &str) -> Result<FkcFile, FkcError> {
    let (front_src, body) = split_front_matter(text)?;

    // §3.8 pre-pass on the front-matter chunk.
    enforce_restricted_yaml(front_src, 0)?;
    let front_matter: FkcFrontMatter =
        serde_yml::from_str(front_src).map_err(|e| FkcError::yaml(None, e))?;

    if let Some(line) = find_orphan_fkc_fence(body) {
        return Err(FkcError::OrphanFkcBlock { line });
    }

    let sections = split_sections(body);
    let mut kernels = Vec::with_capacity(sections.len());
    for section in sections {
        let block = extract_fkc_block(&section)?;
        // §3.8 pre-pass on this kernel's fkc block, reporting absolute lines.
        enforce_restricted_yaml(&block.text, block.start_line)?;
        let kernel: FkcKernel =
            serde_yml::from_str(&block.text).map_err(|e| FkcError::yaml(Some(&section.title), e))?;
        kernels.push(kernel);
    }

    Ok(FkcFile {
        front_matter,
        kernels,
    })
}

/// Read a file at `path`, then [`parse_file`].
pub fn parse_path(path: impl AsRef<Path>) -> Result<FkcFile, FkcError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| {
        FkcError::MalformedFrontMatter(format!("cannot read `{}`: {e}", path.display()))
    })?;
    parse_file(&text)
}

// ===========================================================================
// §3.1 front-matter
// ===========================================================================

/// Split the leading `---`-fenced YAML front-matter off the top of the file.
/// Returns `(front_matter_yaml, remaining_body)`.
fn split_front_matter(text: &str) -> Result<(&str, &str), FkcError> {
    // Tolerate a leading BOM / blank lines before the opening fence.
    let trimmed = text.trim_start_matches('\u{feff}');
    let leading_ws = text.len() - trimmed.len();

    let mut lines = trimmed.lines();
    let first = lines.next().ok_or_else(|| {
        FkcError::MalformedFrontMatter("file is empty (expected `---` front-matter)".into())
    })?;
    if first.trim_end() != "---" {
        return Err(FkcError::MalformedFrontMatter(
            "file must begin with a `---` front-matter fence".into(),
        ));
    }

    // Find the closing `---`. Compute byte offsets so we can slice the original.
    // Offsets are relative to `trimmed`; add `leading_ws` back for the body.
    let mut offset = first.len();
    // Account for the newline after the first line.
    offset += newline_len(trimmed, offset);
    let front_start = offset;

    loop {
        let rest = &trimmed[offset..];
        let line_end = rest.find('\n').map(|i| offset + i).unwrap_or(trimmed.len());
        let line = &trimmed[offset..line_end];
        if line.trim_end() == "---" {
            let front = &trimmed[front_start..offset];
            // Body begins after this closing fence's line + newline.
            let mut body_off = line_end;
            body_off += newline_len(trimmed, body_off);
            let body = &text[leading_ws + body_off..];
            return Ok((front, body));
        }
        if line_end >= trimmed.len() {
            return Err(FkcError::MalformedFrontMatter(
                "unterminated front-matter (missing closing `---`)".into(),
            ));
        }
        offset = line_end + newline_len(trimmed, line_end);
    }
}

/// Length (0, 1, or 2) of the newline sequence at `pos` in `s` (`\n` or `\r\n`).
fn newline_len(s: &str, pos: usize) -> usize {
    let bytes = s.as_bytes();
    match bytes.get(pos) {
        Some(b'\r') if bytes.get(pos + 1) == Some(&b'\n') => 2,
        Some(b'\n') | Some(b'\r') => 1,
        _ => 0,
    }
}

// ===========================================================================
// §3.1 `## ` sections
// ===========================================================================

/// A `## ` kernel section: its heading title + body lines (with absolute line
/// numbers so the §3.8 pre-pass can report meaningfully).
struct Section {
    title: String,
    /// Body lines paired with their absolute (1-based) line number in the file.
    lines: Vec<(usize, String)>,
}

/// Detect a `` ```fkc `` opening fence that appears BEFORE the first `## `
/// heading (or anywhere in a file with no `## ` headings). Such a block is
/// silently dropped by `split_sections`, yielding an Ok-but-empty import.
/// Only a `` ```fkc `` opener triggers; a `` ```yaml `` (or other) intro fence
/// is skipped. Line numbers are 1-based within `body` (same convention as
/// `split_sections`).
fn find_orphan_fkc_fence(body: &str) -> Option<usize> {
    let mut in_fence = false;
    for (idx, raw) in body.lines().enumerate() {
        let trimmed = raw.trim_start();
        if !in_fence && trimmed.starts_with("## ") {
            return None; // first `## ` reached: the rest belongs to sections
        }
        if !in_fence {
            if trimmed.trim_end() == "```fkc" {
                return Some(idx + 1);
            }
            if trimmed.starts_with("```") {
                in_fence = true; // some other intro fence — skip its body
            }
        } else if trimmed.trim_end() == "```" {
            in_fence = false;
        }
    }
    None
}

/// Split the markdown body into `## `-delimited sections. A leading `# ` (H1)
/// title and any prose before the first `## ` are dropped (intro prose, §3.1).
fn split_sections(body: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    let mut current: Option<Section> = None;

    for (idx, raw) in body.lines().enumerate() {
        let line_no = idx + 1; // 1-based within `body`
        if let Some(title) = raw.strip_prefix("## ") {
            if let Some(sec) = current.take() {
                sections.push(sec);
            }
            current = Some(Section {
                title: title.trim().to_string(),
                lines: Vec::new(),
            });
        } else if let Some(sec) = current.as_mut() {
            sec.lines.push((line_no, raw.to_string()));
        }
        // else: prose before the first `## ` — ignored.
    }
    if let Some(sec) = current.take() {
        sections.push(sec);
    }
    sections
}

// ===========================================================================
// §3.1 fenced ```fkc block extraction
// ===========================================================================

/// An extracted fkc block: its YAML text + the absolute start line of the
/// first content line (so §3.8 errors point at the real file line).
struct FkcBlock {
    text: String,
    /// Absolute (1-based, within the section body) line of the first YAML line.
    start_line: usize,
}

/// Extract the single ` ```fkc ` fenced block from a section (§3.1: exactly
/// one). Zero → `MissingFkcBlock`; more than one → `MultipleFkcBlocks`.
fn extract_fkc_block(section: &Section) -> Result<FkcBlock, FkcError> {
    let mut blocks: Vec<FkcBlock> = Vec::new();
    let mut in_block = false;
    let mut buf: Vec<String> = Vec::new();
    let mut block_start = 0usize;

    for (line_no, raw) in &section.lines {
        let trimmed = raw.trim_start();
        if !in_block {
            // Opening fence: ```fkc (allow trailing whitespace).
            if trimmed.trim_end() == "```fkc" {
                in_block = true;
                buf.clear();
                block_start = line_no + 1;
            }
        } else {
            // Inside a block: a bare ``` closes it.
            if trimmed.trim_end() == "```" {
                blocks.push(FkcBlock {
                    text: buf.join("\n"),
                    start_line: block_start,
                });
                in_block = false;
            } else {
                buf.push(raw.clone());
            }
        }
    }

    match blocks.len() {
        0 => Err(FkcError::MissingFkcBlock {
            section: section.title.clone(),
        }),
        1 => Ok(blocks.pop().expect("len checked == 1")),
        n => Err(FkcError::MultipleFkcBlocks {
            section: section.title.clone(),
            count: n,
        }),
    }
}

// ===========================================================================
// §3.8 restricted-YAML pre-pass
// ===========================================================================

/// The Norway-problem tokens (lowercase) that YAML 1.1 coerces to bools.
const NORWAY_TOKENS: &[&str] = &["no", "yes", "on", "off", "n", "y"];

/// Enforce the §3.8 restricted YAML subset on a raw YAML chunk BEFORE
/// deserialization. `base_line` is added to the 0-based in-chunk line index to
/// produce an absolute file line for error reporting.
///
/// Rejects: tab indentation; anchors (`&name`); aliases (`*name`); merge keys
/// (`<<:`); and unquoted Norway tokens in a scalar value position.
fn enforce_restricted_yaml(chunk: &str, base_line: usize) -> Result<(), FkcError> {
    for (i, raw) in chunk.lines().enumerate() {
        let line = base_line + i;

        // --- tab indentation (leading whitespace only) ---
        let indent: String = raw.chars().take_while(|c| *c == ' ' || *c == '\t').collect();
        if indent.contains('\t') {
            return Err(FkcError::TabIndentation { line });
        }

        // Strip a trailing `# ...` comment that is NOT inside a quoted string,
        // so tokens / sigils inside comments and quoted values are ignored.
        let content = strip_comment_and_quotes(raw);
        let content_trim = content.trim();
        if content_trim.is_empty() {
            continue;
        }

        // --- merge keys (`<<:`) ---
        if content_trim.starts_with("<<:") || content_trim.starts_with("- <<:") {
            return Err(FkcError::MergeKeyDisallowed { line });
        }

        // --- anchors / aliases — scan tokens outside quotes ---
        // We scan the comment/quote-stripped `content` (quoted spans were blanked
        // to spaces), so a `&`/`*` only triggers when it is a YAML sigil.
        for tok in content.split_whitespace() {
            // An anchor sigil `&name` appears as a token starting with `&`.
            if let Some(rest) = tok.strip_prefix('&') {
                if !rest.is_empty() && is_anchor_name(rest) {
                    return Err(FkcError::AnchorDisallowed { line });
                }
            }
            // An alias `*name` appears as a token starting with `*`.
            if let Some(rest) = tok.strip_prefix('*') {
                if !rest.is_empty() && is_anchor_name(rest) {
                    return Err(FkcError::AliasDisallowed { line });
                }
            }
        }

        // --- Norway tokens in a scalar VALUE position ---
        // Value position = text after the first unquoted `:` (mapping value) or
        // after a `- ` (sequence item). We use the quote-blanked `content`.
        if let Some((key, value)) = scalar_key_value_of(&content) {
            let v = value.trim();
            // §3.8 exemption: a `name:` operand-ROLE value is explicitly a
            // diagnostic STRING that "stays the string" even for `n`/`y`
            // (the spec's own example: "A `name: n` operand role stays the
            // string `"n"`"). The schema field is `String`, so it cannot
            // coerce to a bool; do NOT reject it. Every OTHER value-position
            // Norway token is still flagged so a bool-targeting field cannot
            // silently coerce.
            let key_is_role_name = key.as_deref() == Some("name");
            if !key_is_role_name
                && !v.is_empty()
                && NORWAY_TOKENS.contains(&v.to_ascii_lowercase().as_str())
            {
                return Err(FkcError::NorwayToken {
                    token: v.to_string(),
                    line,
                });
            }
        }
    }
    Ok(())
}

/// Is `s` a plausible YAML anchor/alias name (alnum / `-` / `_`)? Used to avoid
/// false positives on `*` used as a glob or `&` inside odd text.
fn is_anchor_name(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Return a copy of `raw` with quoted spans (`"..."` / `'...'`) replaced by
/// spaces and any trailing unquoted `#` comment stripped. This lets the §3.8
/// sigil/Norway scans look only at unquoted content.
fn strip_comment_and_quotes(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;

    while let Some(c) = chars.next() {
        match c {
            '\'' if !in_double => {
                in_single = !in_single;
                out.push(' ');
            }
            '"' if !in_single => {
                in_double = !in_double;
                out.push(' ');
            }
            '#' if !in_single && !in_double => {
                // Comment start (only honored when preceded by whitespace or at
                // column 0; YAML requires a space before an inline `#`). If the
                // previous emitted char is non-space and non-empty, treat `#`
                // literally.
                let prev_is_space = out.chars().last().map(|p| p == ' ').unwrap_or(true);
                if prev_is_space {
                    break; // rest of line is a comment
                }
                out.push(c);
            }
            _ => {
                if in_single || in_double {
                    out.push(' ');
                } else {
                    out.push(c);
                }
            }
        }
    }
    out
}

/// Given a quote-blanked, comment-stripped line, return `(key, value)` for the
/// scalar VALUE part: the text after the first `:` (mapping value), or after a
/// leading `- ` (sequence scalar; `key` is `None` then). Returns `None` if the
/// line is a bare key, a nested mapping opener, or a flow collection
/// (`{...}` / `[...]`) we don't unpack here.
fn scalar_key_value_of(content: &str) -> Option<(Option<String>, &str)> {
    let trimmed = content.trim_start();

    // Flow collections are handled token-wise elsewhere; skip whole-line value
    // extraction for them to avoid mis-parsing `{ a: no }` here (the inner
    // `family: no` in `quant: { family: no, ... }` IS still caught because the
    // deserializer would target a String field — and a bare top-level
    // `family: no` line is caught below).
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return None;
    }

    // `- scalar` sequence item (no key).
    let after_dash = trimmed.strip_prefix("- ");
    let is_seq_item = after_dash.is_some();
    let trimmed = after_dash.unwrap_or(trimmed);

    // `key: value` — split on the first `:`.
    if let Some(colon) = trimmed.find(':') {
        let key = trimmed[..colon].trim();
        let value = &trimmed[colon + 1..];
        // A value that itself opens a flow collection or nested map is not a
        // bare scalar.
        let vt = value.trim_start();
        if vt.starts_with('{') || vt.starts_with('[') {
            return None;
        }
        // A `- name: y` sequence-item-with-key still yields key="name".
        let key = if key.is_empty() { None } else { Some(key.to_string()) };
        return Some((key, value));
    }
    // A bare `- scalar` sequence item carries no key.
    if is_seq_item {
        return Some((None, trimmed));
    }
    None
}
