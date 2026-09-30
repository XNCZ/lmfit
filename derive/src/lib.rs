//! Derive macro for `lmfit` curve models.
//!
//! This crate is an implementation detail. Its output is re-exported from
//! `lmfit`, so depend on that crate and write `use lmfit::Model;`
//! rather than naming this one.

mod attr;
mod expand;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// Derive the parameter plumbing for a curve model.
///
/// Every field becomes a fitted parameter, in declaration order, and the
/// struct gains a `Default` impl built from the fields' starting values.
///
/// ```ignore
/// use lmfit::{Curve, Model};
///
/// #[derive(Model)]
/// struct Gaussian {
///     #[param(value = 5.0)]
///     amp: f64,
///     #[param(value = 5.0)]
///     cen: f64,
///     #[param(value = 2.0, min = 0.0)]
///     wid: f64,
/// }
///
/// impl Curve for Gaussian {
///     fn eval(&self, x: f64) -> f64 {
///         self.amp * (-(x - self.cen).powi(2) / self.wid).exp()
///     }
/// }
/// ```
///
/// # Attributes
///
/// * `#[param(value = <expr>)]` — required; the starting value.
/// * `#[param(min = <expr>)]`, `#[param(max = <expr>)]` — bounds.
/// * `#[param(vary = <bool>)]` — set `false` to pin the parameter.
/// * `#[model(name = "...")]` on the struct — override the prefix used for
///   this model's parameters in a composite.
///
/// # What this generates
///
/// `ModelParams` and `Default`, plus `Add`, so that `gaussian + constant`
/// builds a composite. It does **not** generate `Curve`; the arithmetic is
/// yours to write.
///
/// Because `Default` is generated, a model must not also derive `Default`, and
/// because `Add` is generated it must not already implement `Add`.
#[proc_macro_derive(Model, attributes(param, model))]
pub fn derive_model(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand::expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
