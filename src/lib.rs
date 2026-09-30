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
//! Models combine with `+`, and each side's parameters are prefixed so they
//! stay distinct:
//!
//! ```
//! # use lmfit::{Curve, Model};
//! # #[derive(Model)]
//! # struct Gaussian {
//! #     #[param(value = 5.0)] amp: f64,
//! #     #[param(value = 5.0)] cen: f64,
//! #     #[param(value = 2.0, min = 0.0)] wid: f64,
//! # }
//! # impl Curve for Gaussian {
//! #     fn eval(&self, x: f64) -> f64 { self.amp * (-(x - self.cen).powi(2) / self.wid).exp() }
//! # }
//! #[derive(Model)]
//! struct Constant {
//!     #[param(value = 0.0)]
//!     c: f64,
//! }
//!
//! impl Curve for Constant {
//!     fn eval(&self, _x: f64) -> f64 {
//!         self.c
//!     }
//! }
//!
//! # let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
//! # let y: Vec<f64> = x.iter().map(|&t| 5.0 * (-(t - 5.0f64).powi(2) / 2.0).exp() + 0.75).collect();
//! let model = Gaussian { amp: 4.0, cen: 4.0, wid: 1.5 } + Constant { c: 0.0 };
//! let result = model.fit(&y, &x)?;
//!
//! assert!((result.model.b.c - 0.75).abs() < 1e-6);
//! assert!(result.params.get("gaussian_amp").is_some());
//! assert!(result.params.get("constant_c").is_some());
//! # Ok::<(), lmfit::Error>(())
//! ```

// Lets the derive macro's generated `::lmfit::...` paths resolve inside
// this crate as well as in a downstream one, so the built-in models can use
// the same macro their users do.
extern crate self as lmfit;

pub mod bounds;
pub mod composite;
pub mod error;
pub mod models;
pub mod numerics;
pub mod parameter;
pub mod report;
pub mod result;
pub mod solver;
pub mod traits;

pub use bounds::Transform;
pub use composite::Sum;
pub use error::{Error, Result};
pub use parameter::{Parameter, Parameters};
pub use result::ModelResult;
pub use traits::{Curve, ModelParams, ParamSpec};

/// Derive the parameter plumbing for a curve model.
///
/// See [`Curve`] for what a model looks like end to end.
pub use lmfit_derive::Model;
