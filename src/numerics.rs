//! Numerical differentiation.
//!
//! The solver needs a Jacobian, but a model is a user's closure — there are no
//! analytic derivatives to be had. So they are differenced, exactly as
//! MINPACK's `fdjac2` does internally for scipy, and therefore exactly as
//! lmfit's `leastsq` obtains them.
//!
//! Everything here is plain array arithmetic that knows nothing about models,
//! parameters, or the solver: it takes a vector in, calls a closure, and
//! writes a matrix out. Swapping forward differences for central differences
//! or complex-step differentiation would touch this file and nothing else.

/// MINPACK's `epsfcn` — the relative step from which [`fd_step`] is derived.
///
/// lmfit leaves scipy's `leastsq` at its default, so matching that default is
/// what keeps the two implementations' iteration counts and step sequences
/// comparable.
pub const EPSFCN: f64 = 1.0e-10;

/// The forward-difference step to use for a parameter currently at `value`.
///
/// Mirrors MINPACK's `fdjac2`: the step scales with the parameter's magnitude,
/// so a parameter of order `1e6` is not perturbed below the resolution of its
/// own floating-point representation. A parameter at exactly zero has no scale
/// to borrow, so the bare `sqrt(epsfcn)` is used.
pub fn fd_step(value: f64) -> f64 {
    let eps = EPSFCN.max(f64::EPSILON).sqrt();
    let h = eps * value.abs();
    if h == 0.0 { eps } else { h }
}

/// Fill `out` with a forward-difference Jacobian of the residual vector.
///
/// `params` is the point to differentiate at and `base` the residuals already
/// evaluated there — the caller has almost always computed them, and reusing
/// them is the difference between `n` and `n + 1` evaluations per Jacobian.
///
/// `residuals_at` receives a candidate parameter vector and a buffer to fill,
/// returning `false` if the evaluation failed.
///
/// `out` is **column-major** with `base.len()` rows and `params.len()`
/// columns — the layout `nalgebra` stores matrices in and the one
/// `DMatrix::from_vec` expects. Filling it row-major would transpose the
/// Jacobian into something the solver happily accepts and then fits against,
/// converging to a wrong answer with no error anywhere.
///
/// Returns `false` as soon as a perturbed evaluation fails or produces a
/// non-finite difference. A `false` here means the Jacobian was not fully
/// written and must not be used.
pub fn forward_diff_jacobian<R>(
    params: &[f64],
    base: &[f64],
    mut residuals_at: R,
    out: &mut [f64],
) -> bool
where
    R: FnMut(&[f64], &mut [f64]) -> bool,
{
    let nrows = base.len();
    let ncols = params.len();
    debug_assert_eq!(out.len(), nrows * ncols);

    let mut probe = params.to_vec();
    let mut resid = vec![0.0; nrows];

    for j in 0..ncols {
        let h = fd_step(params[j]);
        probe[j] = params[j] + h;
        let ok = residuals_at(&probe, &mut resid);
        probe[j] = params[j];
        if !ok {
            return false;
        }

        let inv_h = 1.0 / h;
        for i in 0..nrows {
            let d = (resid[i] - base[i]) * inv_h;
            if !d.is_finite() {
                return false;
            }
            // Column-major: column j, row i.
            out[j * nrows + i] = d;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The step must track the parameter's magnitude, and fall back to the
    /// bare `sqrt(epsfcn)` at zero where there is no magnitude to track.
    #[test]
    fn step_scales_with_magnitude_and_falls_back_at_zero() {
        let eps = EPSFCN.sqrt();
        assert!((fd_step(0.0) - eps).abs() < 1e-20);
        assert!((fd_step(2.0) - 2.0 * eps).abs() < 1e-20);
        assert!((fd_step(-3.0) - 3.0 * eps).abs() < 1e-20);
        // Order of magnitude check: epsfcn = 1e-10 gives a 1e-5 step.
        assert!((fd_step(0.0) - 1.0e-5).abs() < 1.0e-15);
    }

    /// For the linear model `m(x) = slope*x + intercept` the residual
    /// `r_i = y_i - m(x_i)` has the constant Jacobian `[-x_i, -1]`, so the
    /// difference is exact up to floating-point noise rather than merely close.
    #[test]
    fn matches_the_analytic_jacobian_of_a_line() {
        let x = [0.0, 1.0, 2.5, -4.0];
        let params = vec![3.0, 0.5];

        let residuals_at = |p: &[f64], out: &mut [f64]| {
            for (i, &xi) in x.iter().enumerate() {
                out[i] = -(p[0] * xi + p[1]);
            }
            true
        };

        let mut base = vec![0.0; x.len()];
        residuals_at(&params, &mut base);

        let mut jac = vec![0.0; x.len() * params.len()];
        assert!(forward_diff_jacobian(
            &params,
            &base,
            residuals_at,
            &mut jac
        ));

        let nrows = x.len();
        for (i, &xi) in x.iter().enumerate() {
            // Column-major: parameter j, data point i.
            let d_slope = jac[i];
            let d_intercept = jac[nrows + i];
            // A forward difference on a linear function is exact, so the only
            // error is the division by h.
            assert!(
                (d_slope - (-xi)).abs() < 1e-9,
                "row {i}: {d_slope} vs {}",
                -xi
            );
            assert!(
                (d_intercept - (-1.0)).abs() < 1e-9,
                "row {i}: {d_intercept}"
            );
        }
    }

    /// One Jacobian costs exactly `n + 1` residual evaluations when the base
    /// residuals are supplied — the caller's already-computed values are
    /// reused rather than recomputed.
    #[test]
    fn uses_exactly_one_evaluation_per_parameter() {
        let params = vec![1.0, 2.0, 3.0, 4.0];
        let base = vec![0.0; 5];
        let mut calls = 0usize;

        let mut jac = vec![0.0; 5 * 4];
        assert!(forward_diff_jacobian(
            &params,
            &base,
            |_p: &[f64], out: &mut [f64]| {
                calls += 1;
                out.fill(0.0);
                true
            },
            &mut jac,
        ));

        assert_eq!(calls, params.len());
    }

    /// A non-finite difference aborts the whole Jacobian rather than leaving
    /// the caller to fit against a matrix with a NaN column in it.
    #[test]
    fn aborts_on_non_finite_values() {
        let params = vec![1.0, 2.0];
        let base = vec![1.0, 2.0];

        // Residuals blow up at the second parameter's perturbation.
        let mut jac = vec![0.0; 2 * 2];
        let ok = forward_diff_jacobian(
            &params,
            &base,
            |p: &[f64], out: &mut [f64]| {
                if p[1] != 2.0 {
                    out[0] = f64::NAN;
                    out[1] = 0.0;
                } else {
                    out.copy_from_slice(&base);
                }
                true
            },
            &mut jac,
        );
        assert!(!ok);
    }

    /// An evaluation that reports failure stops the sweep immediately.
    #[test]
    fn aborts_when_an_evaluation_fails() {
        let params = vec![1.0, 2.0, 3.0];
        let base = vec![1.0];
        let mut calls = 0usize;

        let mut jac = vec![0.0; 3];
        let ok = forward_diff_jacobian(
            &params,
            &base,
            |_p: &[f64], out: &mut [f64]| {
                calls += 1;
                out[0] = 0.0;
                calls < 2
            },
            &mut jac,
        );

        assert!(!ok);
        assert_eq!(calls, 2, "should stop at the first failure, not sweep on");
    }
}
