//! Arbitrary curve fitting in Rust, modelled on Python's `lmfit`.
//!
//! A model is a struct whose fields are its parameters, and the arithmetic is
//! the one method you write:
//!
//! ```
//! use lmfit::{Curve, Model};
//!
//! #[derive(Model)]
//! struct Gaussian {
//!     #[param(value = 5.0)]
//!     amp: f64,
//!     #[param(value = 5.0)]
//!     cen: f64,
//!     #[param(value = 2.0, min = 0.0)]
//!     wid: f64,
//! }
//!
//! impl Curve for Gaussian {
//!     fn eval(&self, x: f64) -> f64 {
//!         self.amp * (-(x - self.cen).powi(2) / self.wid).exp()
//!     }
//! }
//!
//! # let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
//! # let y: Vec<f64> = x.iter().map(|&t| 5.0 * (-(t - 5.0f64).powi(2) / 2.0).exp()).collect();
//! let result = Gaussian { amp: 4.0, cen: 4.0, wid: 1.5 }.fit(&y, &x)?;
//!
//! // Fitted values are the model's fields, not string lookups.
//! assert!((result.model.amp - 5.0).abs() < 1e-6);
//! assert!((result.model.cen - 5.0).abs() < 1e-6);
//! assert!((result.model.wid - 2.0).abs() < 1e-6);
//! # Ok::<(), lmfit::Error>(())
//! ```
//!
//! Line shapes are plain functions, so a model combines them by arithmetic
//! into whatever the physics needs:
//!
//! ```
//! # use lmfit::{Curve, Model, lineshapes};
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
//!
//! # let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
//! # let y: Vec<f64> = x.iter().map(|&t| 5.0 * (-(t - 5.0f64).powi(2) / 2.0).exp() + 0.75).collect();
//! let result = PeakOnBackground {
//!     amplitude: 4.0,
//!     center: 4.0,
//!     sigma: 1.5,
//!     background: 0.0,
//! }
//! .fit(&y, &x)?;
//!
//! assert!((result.model.background - 0.75).abs() < 1e-6);
//! assert!((result.model.center - 5.0).abs() < 1e-6);
//! # Ok::<(), lmfit::Error>(())
//! ```

// Lets the derive macro's generated `::lmfit::...` paths resolve inside
// this crate as well as in a downstream one, so the built-in models can use
// the same macro their users do.
extern crate self as lmfit;

pub mod bounds;
pub mod error;
pub mod lineshapes;
pub mod numerics;
pub mod parameter;
pub mod render;
pub mod result;
pub mod solver;
pub mod traits;

pub use bounds::Transform;
pub use error::{Error, Result};
pub use num_complex::Complex64;
pub use parameter::{Parameter, Parameters};
pub use result::{ComplexResult, ModelResult};
pub use traits::{ComplexCurve, Curve, ModelParams, NoPartials, ParamSpec, PartialValues};

/// Derive the parameter plumbing for a curve model.
///
/// See [`Curve`] for what a model looks like end to end.
pub use lmfit_derive::Model;
