//! The Levenberg-Marquardt adapter.
//!
//! This is the only module in the crate that knows the `levenberg-marquardt`
//! crate exists. Everything upstream deals in models and parameters; the
//! solver deals in flat vectors of unbounded doubles. Translating between the
//! two is the whole job here, and it is deliberately isolated: adding a
//! derivative-free solver later means adding a sibling to this file, not
//! touching the traits, the derive macro, or the report.
//!
//! Three translations happen here, and each is easy to get subtly wrong:
//!
//! 1. **Parameter selection.** Only fields marked `vary` reach the solver, so
//!    solver index `j` is not model index `i`. A fixed parameter keeps its
//!    value and is invisible to the fit.
//! 2. **Bounds.** The solver has no box constraints at all, so bounded
//!    parameters are carried in [`Transform`]'s unbounded space. Bounds are
//!    therefore enforced by construction, never by rejection or clamping
//!    during iteration.
//! 3. **The Jacobian.** Required, not optional — the solver does not
//!    difference on our behalf, and returning `None` aborts the fit outright.

use std::cell::{Cell, RefCell};

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{DMatrix, DVector, Dyn, Matrix, Vector, storage::Owned};

use crate::bounds::Transform;
use crate::error::{Error, Result};
use crate::numerics::forward_diff_jacobian;
use crate::parameter::Parameters;
use crate::result::{ModelResult, Statistics, statistics};
use crate::traits::Curve;

/// scipy's `leastsq` default for `ftol` and `xtol`, which lmfit inherits.
///
/// `with_tol` sets `ftol = xtol = tol` *and* `gtol = 0`, which is precisely
/// scipy's combination, so one call expresses all three. This must be set
/// explicitly: the solver's own default without the `minpack-compat` feature
/// is `eps * 30`, which is a very different stopping point.
const TOL: f64 = 1.49012e-8;

/// lmfit's budget is `max_nfev = 2000 * (nvarys + 1)` and the solver's is
/// `patience * (n + 1)`, so the two coincide here.
const PATIENCE: usize = 2000;

/// One parameter the solver is allowed to move.
struct Varying {
    /// Index into the model's parameter layout.
    index: usize,
    /// This parameter's bounds, as a map in and out of solver space.
    transform: Transform,
}

/// Residuals memoised against the solver-space point they belong to.
///
/// The solver asks for residuals and the Jacobian separately, and both are
/// anchored at the same point — so without this the base residuals would be
/// computed twice per iteration, once by each. For a three-parameter model
/// that is one wasted evaluation out of every five.
#[derive(Default)]
struct Cache {
    /// The solver-space vector `residuals` belongs to.
    at: Vec<f64>,
    residuals: Vec<f64>,
    valid: bool,
}

/// A model, its data, and its bounds, presented to the solver as a plain
/// least-squares problem in unbounded space.
pub(crate) struct LmProblem<'a, M> {
    model: M,
    x: &'a [f64],
    y: &'a [f64],
    varying: Vec<Varying>,
    cache: RefCell<Cache>,
    /// Residual evaluations performed, finite-difference probes included.
    nfev: Cell<usize>,
    /// First data point whose residual came out non-finite, if any. Preserved
    /// so that an aborted fit can say *where* it went wrong rather than
    /// reporting the solver's opaque `User("residuals")`.
    nonfinite_at: Cell<Option<usize>>,
}

impl<'a, M: Curve> LmProblem<'a, M> {
    fn new(model: M, x: &'a [f64], y: &'a [f64], varying: Vec<Varying>) -> Self {
        Self {
            model,
            x,
            y,
            varying,
            cache: RefCell::new(Cache::default()),
            nfev: Cell::new(0),
            nonfinite_at: Cell::new(None),
        }
    }

    /// The solver-space vector corresponding to the model's current values.
    fn internal_params(&self) -> Vec<f64> {
        self.varying
            .iter()
            .map(|v| v.transform.to_internal(self.model.get(v.index)))
            .collect()
    }

    /// Residuals of `model` against the data, written into `out`.
    ///
    /// Returns `false` on the first non-finite residual, recording its index
    /// so the caller can report something specific. A single NaN poisons the
    /// entire normal-equations solve, so there is nothing to gain by pushing
    /// on past it.
    fn residuals_of(
        model: &M,
        x: &[f64],
        y: &[f64],
        nfev: &Cell<usize>,
        nonfinite_at: &Cell<Option<usize>>,
        out: &mut [f64],
    ) -> bool {
        nfev.set(nfev.get() + 1);
        for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
            let r = yi - model.eval(xi);
            if !r.is_finite() {
                if nonfinite_at.get().is_none() {
                    nonfinite_at.set(Some(i));
                }
                return false;
            }
            out[i] = r;
        }
        true
    }

    /// Residuals at the model's current values, recomputing only if the cache
    /// does not already hold them for `at`.
    fn cached_residuals(&self, at: &[f64]) -> Option<Vec<f64>> {
        {
            let cache = self.cache.borrow();
            if cache.valid && cache.at == at {
                return Some(cache.residuals.clone());
            }
        }

        let mut out = vec![0.0; self.y.len()];
        if !Self::residuals_of(
            &self.model,
            self.x,
            self.y,
            &self.nfev,
            &self.nonfinite_at,
            &mut out,
        ) {
            return None;
        }

        let mut cache = self.cache.borrow_mut();
        cache.at = at.to_vec();
        cache.residuals = out.clone();
        cache.valid = true;
        Some(out)
    }
}

impl<M: Curve> LeastSquaresProblem<f64, Dyn, Dyn> for LmProblem<'_, M> {
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, p: &Vector<f64, Dyn, Self::ParameterStorage>) {
        for (j, v) in self.varying.iter().enumerate() {
            // `from_internal` cannot leave the bounds, so no clamping is
            // needed and none is applied.
            self.model.set(v.index, v.transform.from_internal(p[j]));
        }
        self.cache.borrow_mut().valid = false;
    }

    fn params(&self) -> Vector<f64, Dyn, Self::ParameterStorage> {
        DVector::from_vec(self.internal_params())
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        let at = self.internal_params();
        self.cached_residuals(&at).map(DVector::from_vec)
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, Dyn, Self::JacobianStorage>> {
        let at = self.internal_params();
        let base = self.cached_residuals(&at)?;

        let nrows = base.len();
        let ncols = self.varying.len();
        let mut out = vec![0.0; nrows * ncols];

        // The model's full parameter vector, varying entries overwritten per
        // probe. Writing every varying entry from `probe` on each call also
        // restores the ones the previous probe disturbed — `probe` is `at`
        // with exactly one entry displaced.
        let mut full: Vec<f64> = (0..M::NPARAMS).map(|i| self.model.get(i)).collect();

        // A perturbed model is *built*, not mutated: `at_values` takes
        // `&self`, which is what makes this reachable from a `&self` method
        // without interior mutability around the model itself.
        let ok = forward_diff_jacobian(
            &at,
            &base,
            |probe, buf| {
                for (j, v) in self.varying.iter().enumerate() {
                    full[v.index] = v.transform.from_internal(probe[j]);
                }
                let perturbed = self.model.at_values(&full);
                Self::residuals_of(&perturbed, self.x, self.y, &self.nfev, &self.nonfinite_at, buf)
            },
            &mut out,
        );

        if ok {
            Some(DMatrix::from_vec(nrows, ncols, out))
        } else {
            None
        }
    }
}

/// Run a fit.
///
/// Split out of [`Curve::fit`] so the trait stays a thin facade.
pub(crate) fn fit<M: Curve>(model: &M, y: &[f64], x: &[f64]) -> Result<ModelResult<M>> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch {
            x: x.len(),
            y: y.len(),
        });
    }

    let params = model.parameters()?;
    // The one place `specs()` and `NPARAMS` meet on every fit, so a
    // hand-written impl that disagrees with itself is caught here in debug
    // builds rather than corrupting a fit.
    debug_assert_eq!(params.len(), M::NPARAMS, "specs() disagrees with NPARAMS");
    let nvarys = params.no_fix_indices().len();
    if y.is_empty() {
        return Err(Error::TooFewDataPoints { ndata: 0, nvarys });
    }

    let varying: Vec<Varying> = params
        .iter()
        .enumerate()
        .filter(|(_, p)| p.vary)
        .map(|(index, p)| {
            Ok(Varying {
                index,
                transform: p.transform()?,
            })
        })
        .collect::<Result<_>>()?;

    let default_values: Vec<f64> = params.values();
    let working = model.at_values(&default_values);

    // Nothing to vary: the model is already at its answer, and handing the
    // solver an empty parameter vector would only make it report
    // `NoParameters`.
    if varying.is_empty() {
        return Ok(assemble(
            working,
            params,
            default_values,
            x,
            y,
            0,
            true,
            "Fit succeeded: there was nothing to vary.".to_string(),
        ));
    }

    let problem = LmProblem::new(working, x, y, varying);

    let (problem, report) = LevenbergMarquardt::<f64>::new()
        .with_tol(TOL)
        .with_patience(PATIENCE)
        .minimize(problem);

    let (success, message) = describe(&report.termination, problem.nonfinite_at.get());

    // Read the fitted model back out. `at_values` again, so the trait needs
    // no `Clone` bound.
    let fit_result: Vec<f64> = (0..M::NPARAMS).map(|i| problem.model.get(i)).collect();
    let fit_model = model.at_values(&fit_result);

    Ok(assemble(
        fit_model,
        params,
        fit_result,
        x,
        y,
        problem.nfev.get(),
        success,
        message,
    ))
}

/// Build the result, computing the curve and every statistic that describes it.
#[allow(clippy::too_many_arguments)]
fn assemble<M: Curve>(
    model: M,
    mut params: Parameters,
    values: Vec<f64>,
    x: &[f64],
    y: &[f64],
    nfev: usize,
    success: bool,
    message: String,
) -> ModelResult<M> {
    for (p, v) in params.iter_mut().zip(&values) {
        p.value = *v;
    }

    let best_fit: Vec<f64> = x.iter().map(|&xi| model.eval(xi)).collect();
    let residual: Vec<f64> = y
        .iter()
        .zip(&best_fit)
        .map(|(observed, predicted)| observed - predicted)
        .collect();

    let nvarys = params.no_fix_indices().len();
    let Statistics {
        chisqr,
        redchi,
        aic,
        bic,
        ndata,
        nfree,
    } = statistics(&residual, nvarys);

    ModelResult {
        model,
        params,
        x: x.to_vec(),
        y: y.to_vec(),
        best_fit,
        residual,
        chisqr,
        redchi,
        aic,
        bic,
        ndata,
        nvarys,
        nfree,
        nfev,
        success,
        message,
    }
}

/// Translate the solver's termination reason into a success flag and a message.
///
/// Every variant is handled explicitly. Collapsing them to "converged or not"
/// would throw away the one thing a user needs when a fit goes wrong: whether
/// it ran out of budget, hit a NaN, or genuinely could not improve.
fn describe(reason: &TerminationReason, nonfinite_at: Option<usize>) -> (bool, String) {
    match reason {
        TerminationReason::Converged { .. } => (true, "Fit succeeded.".to_string()),
        TerminationReason::ResidualsZero => (
            true,
            "Fit succeeded: the residuals are exactly zero.".to_string(),
        ),
        TerminationReason::Orthogonal => (
            true,
            "Fit succeeded: the residual is orthogonal to the Jacobian.".to_string(),
        ),
        TerminationReason::LostPatience => (
            false,
            format!(
                "Fit did not converge: the function evaluation budget of {} was exhausted.",
                PATIENCE
            ),
        ),
        TerminationReason::NoImprovementPossible(what) => (
            false,
            format!("Fit stopped: `{what}` cannot improve further at this precision."),
        ),
        TerminationReason::Numerical(what) => (
            false,
            format!("Fit stopped: a non-finite value appeared in `{what}`."),
        ),
        TerminationReason::NoResiduals => (
            false,
            "Fit stopped: the model produced no residuals.".to_string(),
        ),
        TerminationReason::NoParameters => (
            false,
            "Fit stopped: there were no parameters to vary.".to_string(),
        ),
        TerminationReason::WrongDimensions(what) => (
            false,
            format!("Fit stopped: `{what}` had an unexpected shape."),
        ),
        // The solver reports the same opaque reason for a failed residual and
        // a failed Jacobian. We always supply a Jacobian, so a failure here is
        // a non-finite residual, and `nonfinite_at` says which point.
        TerminationReason::User(what) => match nonfinite_at {
            Some(i) => (
                false,
                format!("Fit stopped: the model produced a non-finite residual at data point {i}."),
            ),
            None => (
                false,
                format!("Fit stopped: the `{what}` evaluation failed."),
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_termination_reason_is_described() {
        let reasons = [
            TerminationReason::Converged {
                ftol: true,
                xtol: false,
            },
            TerminationReason::ResidualsZero,
            TerminationReason::Orthogonal,
            TerminationReason::LostPatience,
            TerminationReason::NoImprovementPossible("xtol"),
            TerminationReason::Numerical("residuals"),
            TerminationReason::NoResiduals,
            TerminationReason::NoParameters,
            TerminationReason::WrongDimensions("jacobian"),
            TerminationReason::User("jacobian"),
        ];
        for reason in reasons {
            let (success, message) = describe(&reason, None);
            assert!(!message.is_empty(), "{reason:?} produced no message");
            // Only the three convergence criteria count as success.
            let expected = matches!(
                reason,
                TerminationReason::Converged { .. }
                    | TerminationReason::ResidualsZero
                    | TerminationReason::Orthogonal
            );
            assert_eq!(
                success, expected,
                "{reason:?} mapped to success = {success}"
            );
        }
    }

    /// A failed residual is reported by data point rather than with the
    /// solver's opaque `User("residuals")`.
    #[test]
    fn a_failed_residual_names_the_offending_point() {
        let (success, message) = describe(&TerminationReason::User("residuals"), Some(17));
        assert!(!success);
        assert!(message.contains("17"), "message was {message:?}");
    }
}
