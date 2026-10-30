// SPDX-License-Identifier: MIT OR Apache-2.0
//! The standard command-line contract for every binary in this crate (CodeRipper `OPS-011`):
//! `--version`/`-V` print the crate version and exit 0, `--help`/`-h` print usage on stdout and
//! exit 0, and any other option-looking argument is a usage error (message on stderr, exit 2).
//! Each binary calls [`standard`] first, so none of them loads a model, opens a device, touches the
//! network or writes a file before the arguments have been understood.
//!
//! Positional arguments (model ids, prompts, paths) are not options and pass through untouched; a
//! binary that takes real flags lists them in `known` (`--device=` matches `--device=0`).

/// What the arguments ask for, decided before anything is printed or exited.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Run the program.
    Run,
    /// Print the version and exit 0.
    Version,
    /// Print usage and exit 0.
    Help,
    /// An option this binary does not know; usage error, exit 2.
    Unknown(String),
}

/// Classify `args` (without the program name). The first version or help flag wins over a later
/// unknown option; an unknown option wins over nothing.
pub fn classify(args: &[String], known: &[&str]) -> Outcome {
    if let Some(o) = args.iter().find_map(|a| info_flag(a)) {
        return o;
    }
    match args.iter().find(|a| is_unknown_option(a, known)) {
        Some(a) => Outcome::Unknown(a.clone()),
        None => Outcome::Run,
    }
}

fn info_flag(a: &str) -> Option<Outcome> {
    match a {
        "--version" | "-V" => Some(Outcome::Version),
        "--help" | "-h" => Some(Outcome::Help),
        _ => None,
    }
}

fn is_unknown_option(a: &str, known: &[&str]) -> bool {
    if !a.starts_with('-') || a == "-" {
        return false;
    }
    !known
        .iter()
        .any(|k| a == *k || (k.ends_with('=') && a.starts_with(k)))
}

/// Apply [`classify`] to the process arguments: print and exit for `--version`, `--help` or an
/// unknown option, return for everything else. `usage` is the text after `Usage: <name> `.
pub fn standard(name: &str, usage: &str, known: &[&str]) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match classify(&args, known) {
        Outcome::Run => {}
        Outcome::Version => {
            println!("{name} {}", env!("CARGO_PKG_VERSION"));
            std::process::exit(0);
        }
        Outcome::Help => {
            println!("Usage: {name} {usage}");
            std::process::exit(0);
        }
        Outcome::Unknown(a) => {
            eprintln!("{name}: unrecognized option '{a}'\nTry '{name} --help' for usage.");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn version_and_help_flags_are_recognised() {
        assert_eq!(classify(&v(&["--version"]), &[]), Outcome::Version);
        assert_eq!(classify(&v(&["-V"]), &[]), Outcome::Version);
        assert_eq!(classify(&v(&["--help"]), &[]), Outcome::Help);
        assert_eq!(classify(&v(&["-h"]), &[]), Outcome::Help);
    }

    #[test]
    fn an_unknown_option_is_a_usage_error() {
        assert_eq!(
            classify(&v(&["--coderipper-no-such-flag"]), &[]),
            Outcome::Unknown("--coderipper-no-such-flag".into())
        );
    }

    #[test]
    fn positionals_and_a_lone_dash_pass_through() {
        assert_eq!(
            classify(&v(&["TinyLlama/x", "a prompt", "32", "-"]), &[]),
            Outcome::Run
        );
        assert_eq!(classify(&[], &[]), Outcome::Run);
    }

    #[test]
    fn declared_flags_pass_and_undeclared_ones_still_fail() {
        let known = ["--list-devices", "--device="];
        assert_eq!(
            classify(&v(&["--list-devices", "--device=1"]), &known),
            Outcome::Run
        );
        // `--device` without `=` is not the declared `--device=`, and a prefix of a declared flag is
        // not the flag.
        assert_eq!(
            classify(&v(&["--device"]), &known),
            Outcome::Unknown("--device".into())
        );
        assert_eq!(
            classify(&v(&["--list-devicesx"]), &known),
            Outcome::Unknown("--list-devicesx".into())
        );
    }

    #[test]
    fn help_wins_over_a_later_unknown_option() {
        assert_eq!(classify(&v(&["--bogus", "--help"]), &[]), Outcome::Help);
    }
}
