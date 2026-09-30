//! Ready-made line shapes.
//!
//! Each one is defined with `#[derive(Model)]` — the same macro a user would
//! reach for — so this module doubles as the macro's most realistic test. The
//! parameter names and formulas match lmfit's `lineshapes` module, because
//! those names are user-visible API: a fit of `GaussianModel` in Python and a
//! fit of [`Gaussian`] here should be describable in the same words.

mod constant;
mod gaussian;

pub use constant::Constant;
pub use gaussian::Gaussian;
