//! Error types for `lmfit`.
//!
//! Hand-written rather than derived so the crate carries no error-handling
//! dependency. Every module in the crate is allowed to depend on this one;
//! this one depends on nothing, which keeps the module graph acyclic.

use std::fmt;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that can go wrong while preparing or running a fit.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A parameter was given `min == max`, leaving no room to vary.
    MinEqualsMax { name: String, min: f64 },

    /// A parameter's lower bound exceeded its upper bound.
    InvertedBounds { name: String, min: f64, max: f64 },

    /// `x` and `y` did not have the same length.
    DimensionMismatch { x: usize, y: usize },

    /// A parameter holds a NaN or infinite value.
    NonFiniteValue { name: String, value: f64 },

    /// A parameter with this name was already added.
    DuplicateParameter { name: String },

    /// No parameter has this name — usually a misspelled runtime override.
    UnknownParameter { name: String },

    /// Fewer data points than varied parameters; the problem is underdetermined.
    TooFewDataPoints { ndata: usize, nvarys: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinEqualsMax { name, min } => write!(
                f,
                "parameter `{name}` has min == max == {min}, leaving nothing to vary"
            ),
            Self::InvertedBounds { name, min, max } => write!(
                f,
                "parameter `{name}` has min ({min}) greater than max ({max})"
            ),
            Self::DimensionMismatch { x, y } => {
                write!(f, "x and y have different lengths: {x} vs {y}")
            }
            Self::NonFiniteValue { name, value } => {
                write!(f, "parameter `{name}` holds a non-finite value: {value}")
            }
            Self::DuplicateParameter { name } => {
                write!(f, "a parameter named `{name}` already exists")
            }
            Self::UnknownParameter { name } => write!(f, "no parameter named `{name}`"),
            Self::TooFewDataPoints { ndata, nvarys } => {
                write!(f, "cannot fit {nvarys} parameters to {ndata} data points")
            }
        }
    }
}

impl std::error::Error for Error {}
