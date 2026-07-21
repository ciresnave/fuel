//! Parse the FKC authoring DSL (`extent(role, axis)`, `const(N)`, `param(N)`,
//! `add|sub|mul|div(a, b)`) into the positional `shape_expr::Dim` AST. Role names are
//! resolved to positional operand indices via the probe combo's canonical order
//! (§6.4-0009 wire form is positional). Returns `None` on any malformed / unknown-role
//! input (the caller maps `None` → skip; never a false reject).

use crate::fkc::return_check::ProbeComboRef;
use crate::fkc::shape_expr::{Dim, LAST};

/// Position of `role` in the combo's canonical order.
fn role_pos(combo: ProbeComboRef, role: &str) -> Option<u8> {
    combo.iter().position(|(r, _, _)| r == role).and_then(|p| u8::try_from(p).ok())
}

/// Split `"a, b"` (the two args of a binary node) at the top-level comma (depth 0).
fn split_top_comma(s: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => return Some((s[..i].trim(), s[i + 1..].trim())),
            _ => {}
        }
    }
    None
}

/// Strip a `head( ... )` wrapper, returning the inner text.
fn inner<'a>(s: &'a str, head: &str) -> Option<&'a str> {
    s.trim().strip_prefix(head)?.strip_suffix(')').map(str::trim)
}

pub fn parse_dim(rule: &str, combo: ProbeComboRef) -> Option<Dim> {
    let rule = rule.trim();
    if let Some(args) = inner(rule, "extent(") {
        let (role, axis) = split_top_comma(args)?;
        let operand = role_pos(combo, role)?;
        let axis = if axis == "last" { LAST } else { axis.parse::<u8>().ok()? };
        return Some(Dim::Extent { operand, axis });
    }
    if let Some(n) = inner(rule, "const(") {
        return Some(Dim::Const(n.parse::<i64>().ok()?));
    }
    if let Some(f) = inner(rule, "param(") {
        return Some(Dim::Param(f.parse::<u8>().ok()?));
    }
    for (head, ctor) in [("add(", 0u8), ("sub(", 1), ("mul(", 2), ("div(", 3)] {
        if let Some(args) = inner(rule, head) {
            let (a, b) = split_top_comma(args)?;
            let (da, db) = (Box::new(parse_dim(a, combo)?), Box::new(parse_dim(b, combo)?));
            return Some(match ctor {
                0 => Dim::Add(da, db),
                1 => Dim::Sub(da, db),
                2 => Dim::Mul(da, db),
                _ => Dim::Div(da, db),
            });
        }
    }
    None
}
