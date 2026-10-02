//! Fit results and the statistics reported alongside them.
//!
//! This module holds data only. It knows nothing about the solver, which is
//! what lets the fit report and any downstream analysis be written and tested
//! without a solver in the loop — and lets a future second solver reuse it
//! unchanged.

use crate::parameter::Parameters;
use num_complex::Complex64;

/// Everything a completed fit produced.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelResult<M> {
    /// The fitted model. Its fields are the parameter values, so reading a
    /// result is `result.model.amplitude` rather than a string lookup.
    pub model: M,

    /// The fitted parameters, in the same order as the model's fields. Carries
    /// bounds and initial values, which the model's fields do not.
    pub params: Parameters,

    /// The independent variable the fit was given.
    pub x: Vec<f64>,
    /// The dependent variable the fit was given.
    pub y: Vec<f64>,

    /// The model evaluated at `x` with the fitted parameters.
    pub best_fit: Vec<f64>,
    /// `y - best_fit`, one entry per data point.
    pub residual: Vec<f64>,

    /// 每个参数的标准误,插入序;固定参数或协方差不可得时为 None。
    pub stderr: Vec<Option<f64>>,
    /// 变参数协方差(外部空间),不可得时为 None。
    pub covar: Option<crate::numerics::Covariance>,

    /// Sum of squared residuals.
    pub chisqr: f64,
    /// Reduced chi-square, `chisqr / nfree`.
    pub redchi: f64,
    /// Akaike information criterion.
    pub aic: f64,
    /// Bayesian information criterion.
    pub bic: f64,

    /// Number of data points.
    pub ndata: usize,
    /// Number of parameters the solver was allowed to vary.
    pub nvarys: usize,
    /// Degrees of freedom, `ndata - nvarys`. Saturates at zero rather than
    /// underflowing when a fit is given more parameters than data points;
    /// lmfit allows `nfree` to go negative there, but every use of it guards
    /// with `max(1, ..)`, so the two agree on every computed quantity and
    /// differ only in a reported value that has no meaning either way.
    pub nfree: usize,

    /// Number of residual evaluations the solver performed.
    ///
    /// Deliberately **not** comparable to lmfit's `nfev`: that counts calls
    /// into the user's objective, whereas this counts residual evaluations
    /// including the ones spent on finite differences. Do not "fix" it to
    /// match — the difference is inherent to where the two implementations
    /// draw the line, not an accounting bug.
    pub nfev: usize,

    /// Whether the solver reached one of its convergence criteria.
    pub success: bool,
    /// Human-readable account of how the fit ended.
    pub message: String,
}

/// Everything a completed complex fit produced.
///
/// 字段镜像 [`ModelResult`],`y`/`best_fit`/`residual` 为复数;复数残差按
/// 实部、虚部逐点交错进入统计(与 lmfit 对齐):`chisqr = Σ|r_i|²`,
/// `ndata = 2n`(数据点数的两倍,lmfit 口径),`nfree = ndata - nvarys`。
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexResult<M> {
    /// The fitted model. Its fields are the parameter values.
    pub model: M,

    /// The fitted parameters, in the same order as the model's fields.
    pub params: Parameters,

    /// The complex independent variable the fit was given(实自变量以虚部 0 嵌入)。
    pub x: Vec<Complex64>,
    /// The complex dependent variable the fit was given.
    pub y: Vec<Complex64>,

    /// The model evaluated at `x` with the fitted parameters.
    pub best_fit: Vec<Complex64>,
    /// `y - best_fit`, one entry per data point.
    pub residual: Vec<Complex64>,

    /// 每个参数的标准误,插入序;固定参数或协方差不可得时为 None。
    pub stderr: Vec<Option<f64>>,
    /// 变参数协方差(外部空间),不可得时为 None。
    pub covar: Option<crate::numerics::Covariance>,

    /// Sum of squared complex residuals, `Σ|r_i|²`.
    pub chisqr: f64,
    /// Reduced chi-square, `chisqr / nfree`.
    pub redchi: f64,
    /// Akaike information criterion.
    pub aic: f64,
    /// Bayesian information criterion.
    pub bic: f64,

    /// Number of real residual slots, `2n` — 与 lmfit 对齐。
    pub ndata: usize,
    /// Number of parameters the solver was allowed to vary.
    pub nvarys: usize,
    /// Degrees of freedom, `ndata - nvarys`.
    pub nfree: usize,

    /// Number of residual evaluations the solver performed.
    pub nfev: usize,

    /// Whether the solver reached one of its convergence criteria.
    pub success: bool,
    /// Human-readable account of how the fit ended.
    pub message: String,
}

/// The statistics lmfit reports for a fit, in the order it computes them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Statistics {
    pub chisqr: f64,
    pub redchi: f64,
    pub aic: f64,
    pub bic: f64,
    pub ndata: usize,
    pub nfree: usize,
}

/// Compute [`Statistics`] from a residual vector.
///
/// The ordering here is load-bearing and mirrors lmfit's
/// `MinimizerResult._calculate_statistics` exactly:
///
/// 1. `redchi` is computed from the **raw** sum of squares.
/// 2. Only then is chi-square floored at `1e-250 * ndata`, so that the
///    logarithm below cannot be handed a zero. A perfect fit would otherwise
///    report `aic = bic = -inf`, which is worse than useless for comparing
///    models — the whole point of the two criteria.
/// 3. `aic` and `bic` share one `-2·log-likelihood` term, built from the
///    *floored* chi-square.
///
/// # Panics
///
/// Panics if `residual` is empty; a fit with no data points is rejected before
/// it reaches here.
pub(crate) fn statistics(residual: &[f64], nvarys: usize) -> Statistics {
    let ndata = residual.len();
    assert!(ndata > 0, "statistics require at least one data point");

    let nfree = ndata.saturating_sub(nvarys);
    let raw_chisqr: f64 = residual.iter().map(|r| r * r).sum();

    // Step 1: reduced chi-square sees the unfloored value.
    let redchi = raw_chisqr / nfree.max(1) as f64;

    // Step 2: floor, so step 3's logarithm stays finite on a perfect fit.
    let chisqr = raw_chisqr.max(1.0e-250 * ndata as f64);

    // Step 3: both information criteria share this term.
    let neg2_log_likel = ndata as f64 * (chisqr / ndata as f64).ln();

    Statistics {
        chisqr,
        redchi,
        aic: neg2_log_likel + 2.0 * nvarys as f64,
        bic: neg2_log_likel + (ndata as f64).ln() * nvarys as f64,
        ndata,
        nfree,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The formulas written out independently from lmfit's source, so that a
    /// change to the implementation has to disagree with something other than
    /// itself to pass.
    fn reference(residual: &[f64], nvarys: usize) -> (f64, f64, f64, f64, usize) {
        let ndata = residual.len();
        let nfree = ndata.saturating_sub(nvarys);
        let raw: f64 = residual.iter().map(|r| r * r).sum();
        let redchi = raw / (if nfree > 1 { nfree } else { 1 }) as f64;
        let chisqr = raw.max(1.0e-250 * ndata as f64);
        let neg2 = ndata as f64 * (chisqr / ndata as f64).ln();
        let aic = neg2 + 2.0 * nvarys as f64;
        let bic = neg2 + (ndata as f64).ln() * nvarys as f64;
        (chisqr, redchi, aic, bic, nfree)
    }

    fn assert_matches(residual: &[f64], nvarys: usize) {
        let got = statistics(residual, nvarys);
        let (chisqr, redchi, aic, bic, nfree) = reference(residual, nvarys);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * (1.0 + b.abs());
        assert!(
            close(got.chisqr, chisqr),
            "chisqr {} vs {chisqr}",
            got.chisqr
        );
        assert!(
            close(got.redchi, redchi),
            "redchi {} vs {redchi}",
            got.redchi
        );
        assert!(close(got.aic, aic), "aic {} vs {aic}", got.aic);
        assert!(close(got.bic, bic), "bic {} vs {bic}", got.bic);
        assert_eq!(got.ndata, residual.len());
        assert_eq!(got.nfree, nfree);
    }

    #[test]
    fn matches_the_reference_formulas() {
        assert_matches(&[0.5, -1.25, 2.0, 0.0, -0.125], 3);
        assert_matches(&[1.0], 0);
        assert_matches(&[3.0, 4.0], 0);
        assert_matches(&[0.1; 50], 5);
    }

    /// Hand-computed check of a small case, independent of both
    /// implementations' shape.
    #[test]
    fn hand_computed_values() {
        // chisqr = 1 + 4 + 9 = 14, ndata = 3, nvarys = 1, nfree = 2
        let s = statistics(&[1.0, -2.0, 3.0], 1);

        assert_eq!(s.ndata, 3);
        assert_eq!(s.nfree, 2);
        assert!((s.chisqr - 14.0).abs() < 1e-12);
        assert!((s.redchi - 7.0).abs() < 1e-12);

        let neg2 = 3.0 * (14.0_f64 / 3.0).ln();
        assert!((s.aic - (neg2 + 2.0)).abs() < 1e-12);
        assert!((s.bic - (neg2 + 3.0_f64.ln())).abs() < 1e-12);
    }

    /// The regression that a naive implementation fails: an exact fit gives
    /// `chisqr == 0`, and taking its logarithm yields `-inf`. lmfit floors
    /// chi-square first, and so must we.
    #[test]
    fn perfect_fit_stays_finite() {
        let s = statistics(&[0.0, 0.0, 0.0, 0.0], 2);

        assert!(s.aic.is_finite(), "aic was {}", s.aic);
        assert!(s.bic.is_finite(), "bic was {}", s.bic);
        assert!((s.chisqr - 1.0e-250 * 4.0).abs() < 1e-260);

        // The floor is applied *after* redchi, so redchi reports the true
        // zero rather than a floored stand-in.
        assert_eq!(s.redchi, 0.0);
    }

    /// More parameters than data points is degenerate but must not divide by
    /// zero or underflow.
    #[test]
    fn more_parameters_than_points_is_handled() {
        let s = statistics(&[1.0, 2.0], 7);

        assert_eq!(s.nfree, 0);
        assert!((s.redchi - 5.0).abs() < 1e-12, "redchi was {}", s.redchi);
        assert!(s.aic.is_finite() && s.bic.is_finite());
    }
}
