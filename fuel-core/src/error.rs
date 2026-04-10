//! Fuel-specific Error and Result
//!
//! The canonical definitions live in [`fuel_core_types::error`].
//! This module re-exports them so that `crate::Error`, `crate::Result`,
//! `crate::Context`, etc. continue to resolve within fuel-core.
pub use fuel_core_types::error::{
    zip, Context, Error, MatMulUnexpectedStriding, Result,
};

/// Returns early from a function with a formatted error message.
///
/// This is fuel-core's own `bail!` macro.  `$crate::Error` resolves to
/// `fuel_core::Error` (which re-exports `fuel_core_types::Error`), so
/// callers inside this crate can write `bail!("oops")` and get the correct
/// type.
#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()).bt())
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()).bt())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()).bt())
    };
}
