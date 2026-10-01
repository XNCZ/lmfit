//! Ready-made line shapes, as plain functions.
//!
//! A line shape is arithmetic, not a model. Define your own
//! `#[derive(Model)]` struct and call these from `eval`, combining them
//! however the physics needs:
//!
//! ```
//! use lmfit::{Curve, Model, lineshapes};
//!
//! #[derive(Model)]
//! struct PeakOnBackground {
//!     #[param(value = 5.0)]
//!     amplitude: f64,
//!     #[param(value = 5.0)]
//!     center: f64,
//!     #[param(value = 2.0, min = 0.0)]
//!     sigma: f64,
//!     #[param(value = 0.0)]
//!     background: f64,
//! }
//!
//! impl Curve for PeakOnBackground {
//!     fn eval(&self, x: f64) -> f64 {
//!         lineshapes::gaussian(x, self.amplitude, self.center, self.sigma) + self.background
//!     }
//! }
//! ```
//!
//! The parameter names and formulas match lmfit's `lineshapes` module.

mod constant;
mod gaussian;

pub use constant::constant;
pub use gaussian::gaussian;
