//! End-to-end fits, driven by a hand-written model.
//!
//! This is the milestone that proves the architecture: it exercises the whole
//! chain — parameter layout, bounds, the solver adapter, the residual cache,
//! the finite-difference Jacobian, and the statistics — without any macro
//! existing yet. If the internal/external assembly or the cache is wrong, it
//! shows up here.
//!
//! Milestone M4 adds the same coverage for `#[derive(Model)]` and asserts the
//! two agree, so this file stays as the hand-written reference.

use lmfit::{Curve, ModelParams, ParamSpec};

/// A Gaussian, written out by hand — exactly what `#[derive(Model)]` will
/// generate in M4, so the two can be compared directly.
#[derive(Debug, Clone, PartialEq)]
struct Gaussian {
    amp: f64,
    cen: f64,
    wid: f64,
}

impl Gaussian {
    fn new(amp: f64, cen: f64, wid: f64) -> Self {
        Self { amp, cen, wid }
    }

    /// The data-generating function, independent of the model's `eval`.
    fn truth(x: f64, amp: f64, cen: f64, wid: f64) -> f64 {
        amp * (-(x - cen).powi(2) / wid).exp()
    }
}

impl ModelParams for Gaussian {
    const MODEL_NAME: &'static str = "gaussian";

    fn specs(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec {
                name: "amp".to_string(),
                value: self.amp,
                min: None,
                max: None,
                vary: true,
            },
            ParamSpec {
                name: "cen".to_string(),
                value: self.cen,
                min: None,
                max: None,
                vary: true,
            },
            ParamSpec {
                name: "wid".to_string(),
                value: self.wid,
                // A width must stay positive; this exercises the one-sided
                // bounds transform inside a real fit.
                min: Some(0.0),
                max: None,
                vary: true,
            },
        ]
    }

    const NPARAMS: usize = 3;

    fn get(&self, index: usize) -> f64 {
        match index {
            0 => self.amp,
            1 => self.cen,
            2 => self.wid,
            _ => panic!("index {index} out of range"),
        }
    }

    fn set(&mut self, index: usize, value: f64) {
        match index {
            0 => self.amp = value,
            1 => self.cen = value,
            2 => self.wid = value,
            _ => panic!("index {index} out of range"),
        }
    }

    fn at_values(&self, values: &[f64]) -> Self {
        Self::new(values[0], values[1], values[2])
    }
}

impl Curve for Gaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amp * (-(x - self.cen).powi(2) / self.wid).exp()
    }
}

/// Sample points spanning the peak.
fn xs(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 * 10.0 / (n - 1) as f64).collect()
}

/// Deterministic pseudo-noise in `[-0.5, 0.5)`.
///
/// Deliberately not an RNG: an explicit arithmetic sequence cannot drift
/// between runs or toolchains, so a failure here always means a real
/// regression rather than a changed seed.
fn noise(i: usize) -> f64 {
    let h = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((h >> 33) as f64 / (1u64 << 31) as f64) - 0.5
}

fn data(n: usize, scale: f64, amp: f64, cen: f64, wid: f64) -> (Vec<f64>, Vec<f64>) {
    let x = xs(n);
    let y = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| Gaussian::truth(xi, amp, cen, wid) + scale * noise(i))
        .collect();
    (x, y)
}

/// A starting guess that overlaps the data.
///
/// The peak sits at `x = 5` with a width of 2, so a model started at
/// `cen = 1, wid = 1` is a narrow spike three widths away from any data — the
/// two curves barely overlap and the gradient is almost flat. That is a real
/// local minimum of the problem, not a solver defect, and lmfit stalls there
/// too. Every test below therefore starts somewhere the model can see the
/// data, which is what `guess()`-style heuristics exist to arrange.
fn start() -> Gaussian {
    Gaussian::new(3.0, 4.0, 1.5)
}

#[test]
fn recovers_known_parameters_from_noiseless_data() {
    let (x, y) = data(101, 0.0, 5.0, 5.0, 2.0);

    let result = start().fit(&y, &x).expect("fit ran");

    assert!(result.success, "fit failed: {}", result.message);
    assert!(
        (result.model.amp - 5.0).abs() < 1e-6,
        "amp = {}",
        result.model.amp
    );
    assert!(
        (result.model.cen - 5.0).abs() < 1e-6,
        "cen = {}",
        result.model.cen
    );
    assert!(
        (result.model.wid - 2.0).abs() < 1e-6,
        "wid = {}",
        result.model.wid
    );

    // An exact fit drives chi-square to zero, which is the case the statistics
    // floor exists for.
    assert!(result.chisqr < 1e-12, "chisqr = {}", result.chisqr);
    assert!(result.aic.is_finite() && result.bic.is_finite());
    assert_eq!(result.ndata, 101);
    assert_eq!(result.nvarys, 3);
    assert_eq!(result.nfree, 98);
}

#[test]
fn recovers_known_parameters_from_noisy_data() {
    let (x, y) = data(201, 0.2, 5.0, 5.0, 2.0);

    let result = start().fit(&y, &x).expect("fit ran");

    assert!(result.success, "fit failed: {}", result.message);
    assert!(
        (result.model.amp - 5.0).abs() < 0.1,
        "amp = {}",
        result.model.amp
    );
    assert!(
        (result.model.cen - 5.0).abs() < 0.1,
        "cen = {}",
        result.model.cen
    );
    assert!(
        (result.model.wid - 2.0).abs() < 0.2,
        "wid = {}",
        result.model.wid
    );

    // The curve and residual must be consistent with the reported parameters.
    assert_eq!(result.best_fit.len(), result.y.len());
    assert_eq!(result.residual.len(), result.y.len());
    for i in 0..result.y.len() {
        assert!(
            (result.residual[i] - (result.y[i] - result.best_fit[i])).abs() < 1e-12,
            "residual {i} disagrees with y - best_fit"
        );
    }
}

/// The reported statistics must describe the reported parameters.
#[test]
fn statistics_are_internally_consistent() {
    let (x, y) = data(201, 0.2, 5.0, 5.0, 2.0);
    let result = start().fit(&y, &x).expect("fit ran");

    let summed: f64 = result.residual.iter().map(|r| r * r).sum();
    assert!(
        (result.chisqr - summed).abs() <= 1e-9 * (1.0 + summed),
        "chisqr {} vs summed squared residual {}",
        result.chisqr,
        summed
    );

    assert!((result.redchi - result.chisqr / result.nfree as f64).abs() < 1e-9);

    let neg2 = result.ndata as f64 * (result.chisqr / result.ndata as f64).ln();
    assert!((result.aic - (neg2 + 2.0 * result.nvarys as f64)).abs() < 1e-9);
    assert!((result.bic - (neg2 + (result.ndata as f64).ln() * result.nvarys as f64)).abs() < 1e-9);
}

/// The fitted model must agree with `best_fit`, and the parameters it carries
/// must be the ones the fit ended on.
#[test]
fn fitted_model_reproduces_the_curve() {
    let (x, y) = data(101, 0.1, 4.0, 6.0, 3.0);
    let result = Gaussian::new(4.0, 5.0, 3.0).fit(&y, &x).expect("fit ran");

    for (i, &xi) in x.iter().enumerate() {
        let from_model = result.model.eval(xi);
        assert!(
            (from_model - result.best_fit[i]).abs() < 1e-12,
            "model and best_fit disagree at {xi}"
        );
    }

    // And the reported parameters agree with the model's fields.
    assert_eq!(result.params.get("amp").unwrap().value, result.model.amp);
    assert_eq!(result.params.get("wid").unwrap().value, result.model.wid);
    assert_eq!(result.params.get("wid").unwrap().min, Some(0.0));
}

/// A bound that excludes the true optimum must pin the parameter there, and
/// the reported value must never sit outside the bound.
///
/// This is the test that would catch a sign error in the one-sided transform:
/// the parameter would still fit, still look plausible, and simply settle
/// somewhere it was told it could not go.
#[test]
fn bounds_pin_a_parameter_that_cannot_reach_the_optimum() {
    let (x, y) = data(101, 0.0, 5.0, 5.0, 2.0);
    // The peak is at 5.0 but the bound forbids anything below 5.5, so the best
    // reachable centre is the bound itself. Starting above it, so the fit has
    // to travel down onto the bound rather than sit on it from the outset.
    let model = BoundedGaussian {
        inner: Gaussian::new(5.0, 6.5, 2.0),
        cen_min: 5.5,
    };

    let result = model.fit(&y, &x).expect("fit ran");

    assert!(
        result.model.inner.cen >= 5.5 - 1e-9,
        "centre escaped its lower bound: {}",
        result.model.inner.cen
    );
    // It approaches the bound but need not land exactly on it: once the centre
    // is this close, moving it the rest of the way changes the sum of squares
    // by less than `ftol`, so the solver stops. lmfit behaves the same way.
    // What matters is that it came *down* to the bound from 6.5 and stopped.
    assert!(
        (result.model.inner.cen - 5.5).abs() < 0.01,
        "expected the centre to pin at the bound, got {}",
        result.model.inner.cen
    );
    // The reported parameter must agree with the model, and respect the bound.
    let reported = result.params.get("cen").unwrap();
    assert_eq!(reported.value, result.model.inner.cen);
    assert_eq!(reported.min, Some(5.5));
}

/// A bound that contains the optimum must not disturb it — the transform has
/// to be inert when it is not binding.
#[test]
fn interior_bounds_leave_the_optimum_alone() {
    let (x, y) = data(101, 0.0, 5.0, 5.0, 2.0);
    let model = BoundedGaussian {
        inner: start(),
        cen_min: 2.0,
    };

    let result = model.fit(&y, &x).expect("fit ran");

    assert!(result.success, "fit failed: {}", result.message);
    assert!(
        (result.model.inner.cen - 5.0).abs() < 1e-6,
        "a non-binding bound shifted the centre to {}",
        result.model.inner.cen
    );
}

/// A wrapper whose parameter spec carries a lower bound on the centre.
#[derive(Debug, Clone, PartialEq)]
struct BoundedGaussian {
    inner: Gaussian,
    cen_min: f64,
}

impl ModelParams for BoundedGaussian {
    const MODEL_NAME: &'static str = "bounded_gaussian";

    fn specs(&self) -> Vec<ParamSpec> {
        let mut specs = self.inner.specs();
        specs[1].min = Some(self.cen_min);
        specs
    }
    const NPARAMS: usize = Gaussian::NPARAMS;
    fn get(&self, index: usize) -> f64 {
        self.inner.get(index)
    }
    fn set(&mut self, index: usize, value: f64) {
        self.inner.set(index, value);
    }
    fn at_values(&self, values: &[f64]) -> Self {
        Self {
            inner: self.inner.at_values(values),
            cen_min: self.cen_min,
        }
    }
}

impl Curve for BoundedGaussian {
    fn eval(&self, x: f64) -> f64 {
        self.inner.eval(x)
    }
}

/// A parameter marked `vary = false` keeps its value and is excluded from the
/// varied count.
#[test]
fn fixed_parameters_are_left_alone() {
    let (x, y) = data(101, 0.0, 5.0, 5.0, 2.0);

    // `amp` is pinned at 1.0 although the data reaches 5.0, so the remaining
    // parameters can only fit the shape, not the height.
    let model = FixedAmp {
        inner: Gaussian::new(1.0, 4.0, 1.5),
    };
    let result = model.fit(&y, &x).expect("fit ran");

    assert_eq!(result.model.inner.amp, 1.0, "a fixed value was moved");
    assert_eq!(result.nvarys, 2);
    assert!(!result.params.get("amp").unwrap().vary);

    // The peak position is set by symmetry, so it is recoverable even with the
    // amplitude pinned.
    assert!(
        (result.model.inner.cen - 5.0).abs() < 0.05,
        "centre should still find the peak, got {}",
        result.model.inner.cen
    );
}

#[derive(Debug, Clone, PartialEq)]
struct FixedAmp {
    inner: Gaussian,
}

impl ModelParams for FixedAmp {
    const MODEL_NAME: &'static str = "fixed_amp";

    fn specs(&self) -> Vec<ParamSpec> {
        let mut specs = self.inner.specs();
        specs[0].vary = false;
        specs
    }
    const NPARAMS: usize = Gaussian::NPARAMS;
    fn get(&self, index: usize) -> f64 {
        self.inner.get(index)
    }
    fn set(&mut self, index: usize, value: f64) {
        self.inner.set(index, value);
    }
    fn at_values(&self, values: &[f64]) -> Self {
        Self {
            inner: self.inner.at_values(values),
        }
    }
}

impl Curve for FixedAmp {
    fn eval(&self, x: f64) -> f64 {
        self.inner.eval(x)
    }
}

#[test]
fn mismatched_input_is_rejected() {
    let model = Gaussian::new(1.0, 1.0, 1.0);
    let err = model.fit(&[1.0, 2.0, 3.0], &[1.0, 2.0]).unwrap_err();
    assert!(matches!(
        err,
        lmfit::Error::DimensionMismatch { x: 2, y: 3 }
    ));

    let err = model.fit(&[], &[]).unwrap_err();
    assert!(matches!(
        err,
        lmfit::Error::TooFewDataPoints { ndata: 0, .. }
    ));
}
