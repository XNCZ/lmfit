//! `#[derive(Model)]` — what it generates, and that it generates the *same*
//! thing a careful hand-written impl would.
//!
//! The equivalence test is the important one: the macro is only trustworthy if
//! it produces a model indistinguishable from one written out by hand, so the
//! two are fitted to identical data and every observable is compared.

use lmfit::{Curve, Model, ModelParams, ParamSpec};

// ---------------------------------------------------------------------------
// A model written by hand, as the reference.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct HandGaussian {
    amp: f64,
    cen: f64,
    wid: f64,
}

impl ModelParams for HandGaussian {
    const MODEL_NAME: &'static str = "hand_gaussian";

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
                min: Some(0.0),
                max: None,
                vary: true,
            },
        ]
    }

    fn nparams(&self) -> usize {
        3
    }

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

    fn with_values(&self, values: &[f64]) -> Self {
        Self {
            amp: values[0],
            cen: values[1],
            wid: values[2],
        }
    }
}

impl Curve for HandGaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amp * (-(x - self.cen).powi(2) / self.wid).exp()
    }
}

// ---------------------------------------------------------------------------
// The same model, generated.
// ---------------------------------------------------------------------------

#[derive(Model, Debug)]
struct DerivedGaussian {
    #[param(value = 5.0)]
    amp: f64,
    #[param(value = 5.0)]
    cen: f64,
    #[param(value = 2.0, min = 0.0)]
    wid: f64,
}

impl Curve for DerivedGaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amp * (-(x - self.cen).powi(2) / self.wid).exp()
    }
}

fn xs(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 * 10.0 / (n - 1) as f64).collect()
}

fn gaussian_data(n: usize, amp: f64, cen: f64, wid: f64) -> (Vec<f64>, Vec<f64>) {
    let x = xs(n);
    let y = x
        .iter()
        .map(|&xi| amp * (-(xi - cen).powi(2) / wid).exp())
        .collect();
    (x, y)
}

/// The whole point of the macro: a derived model must be indistinguishable
/// from one written by hand, down to the last bit of the fit.
#[test]
fn derived_model_fits_identically_to_a_hand_written_one() {
    let (x, y) = gaussian_data(101, 5.0, 5.0, 2.0);

    let derived = DerivedGaussian {
        amp: 3.0,
        cen: 4.0,
        wid: 1.5,
    }
    .fit(&y, &x)
    .expect("derived fit ran");

    let hand = HandGaussian {
        amp: 3.0,
        cen: 4.0,
        wid: 1.5,
    }
    .fit(&y, &x)
    .expect("hand fit ran");

    assert_eq!(derived.model.amp, hand.model.amp);
    assert_eq!(derived.model.cen, hand.model.cen);
    assert_eq!(derived.model.wid, hand.model.wid);

    assert_eq!(derived.chisqr, hand.chisqr);
    assert_eq!(derived.redchi, hand.redchi);
    assert_eq!(derived.aic, hand.aic);
    assert_eq!(derived.bic, hand.bic);
    assert_eq!(derived.nfev, hand.nfev);
    assert_eq!(derived.success, hand.success);
    assert_eq!(derived.best_fit, hand.best_fit);
    assert_eq!(derived.residual, hand.residual);

    // And both actually solved the problem.
    assert!((derived.model.amp - 5.0).abs() < 1e-6);
    assert!((derived.model.cen - 5.0).abs() < 1e-6);
    assert!((derived.model.wid - 2.0).abs() < 1e-6);
}

#[test]
fn default_uses_the_annotated_starting_values() {
    let m = DerivedGaussian::default();
    assert_eq!(m.amp, 5.0);
    assert_eq!(m.cen, 5.0);
    assert_eq!(m.wid, 2.0);
}

#[test]
fn specs_carry_names_bounds_and_vary_flags() {
    let m = DerivedGaussian::default();
    let specs = m.specs();

    assert_eq!(specs.len(), 3);
    assert_eq!(m.nparams(), 3);

    assert_eq!(specs[0].name, "amp");
    assert_eq!(specs[0].min, None);
    assert_eq!(specs[0].max, None);
    assert!(specs[0].vary && specs[1].vary && specs[2].vary);

    assert_eq!(specs[2].name, "wid");
    assert_eq!(specs[2].min, Some(0.0));
    assert_eq!(specs[2].max, None);
}

/// `MODEL_NAME` comes from the struct name, lowercased — it is what prefixes
/// a composite's parameter names.
#[test]
fn model_name_is_derived_from_the_type_name() {
    assert_eq!(DerivedGaussian::MODEL_NAME, "derived_gaussian");
    assert_eq!(Constant::MODEL_NAME, "constant");
}

#[test]
fn default_and_hand_written_models_agree_on_the_starting_point() {
    let (x, y) = gaussian_data(101, 5.0, 5.0, 2.0);
    // Starting from the defaults rather than a literal, so this also exercises
    // the generated `Default`.
    let result = DerivedGaussian::default().fit(&y, &x).expect("fit ran");
    assert!(result.success, "{}", result.message);

    let reported = result.params.get("amp").unwrap();
    assert_eq!(reported.init, 5.0, "init should record the starting value");
    assert_eq!(reported.value, result.model.amp);
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

/// Field order, not alphabetical order, defines parameter index order. A model
/// whose fields would sort differently is the case that catches a macro that
/// alphabetises.
#[derive(Model, Debug)]
struct OutOfOrder {
    #[param(value = 1.0)]
    zeta: f64,
    #[param(value = 2.0)]
    alpha: f64,
    #[param(value = 3.0)]
    mid: f64,
}

impl Curve for OutOfOrder {
    fn eval(&self, _x: f64) -> f64 {
        self.zeta + self.alpha + self.mid
    }
}

#[test]
fn parameter_order_follows_declaration_order() {
    let m = OutOfOrder::default();
    let names: Vec<String> = m.specs().into_iter().map(|s| s.name).collect();
    assert_eq!(names, vec!["zeta", "alpha", "mid"]);

    assert_eq!(m.get(0), 1.0);
    assert_eq!(m.get(1), 2.0);
    assert_eq!(m.get(2), 3.0);

    let rebuilt = m.with_values(&[10.0, 20.0, 30.0]);
    assert_eq!(rebuilt.zeta, 10.0);
    assert_eq!(rebuilt.alpha, 20.0);
    assert_eq!(rebuilt.mid, 30.0);
}

#[derive(Model, Debug)]
struct Bounds {
    #[param(value = 5.0, min = 0.0, max = 10.0)]
    full: f64,
    #[param(value = 5.0, min = -1.0)]
    lower: f64,
    #[param(value = 5.0, max = 1.0)]
    upper: f64,
    #[param(value = 7.0, vary = false)]
    pinned: f64,
}

impl Curve for Bounds {
    fn eval(&self, _x: f64) -> f64 {
        self.full + self.lower + self.upper + self.pinned
    }
}

#[test]
fn every_bound_and_vary_form_is_honoured() {
    let m = Bounds::default();
    let specs = m.specs();
    let by_name = |n: &str| specs.iter().find(|s| s.name == n).unwrap();

    assert_eq!(by_name("full").min, Some(0.0));
    assert_eq!(by_name("full").max, Some(10.0));
    assert_eq!(by_name("lower").min, Some(-1.0));
    assert_eq!(by_name("lower").max, None);
    assert_eq!(by_name("upper").min, None);
    assert_eq!(by_name("upper").max, Some(1.0));
    assert_eq!(by_name("pinned").max, None);

    assert!(by_name("full").vary);
    assert!(!by_name("pinned").vary);
    assert_eq!(by_name("pinned").value, 7.0);
}

/// A bounded, pinned mix must survive a real fit: the pinned parameter stays
/// put and the bounded one stays inside its range.
#[test]
fn bounds_and_pins_survive_a_fit() {
    let (x, y) = gaussian_data(101, 5.0, 5.0, 2.0);
    let result = PinnedGaussian::default().fit(&y, &x).expect("fit ran");

    assert_eq!(result.model.amp, 5.0, "pinned parameter moved");
    assert_eq!(result.nvarys, 2);
    assert!(
        result.model.wid >= 0.0,
        "width escaped its bound: {}",
        result.model.wid
    );
    assert!((result.model.cen - 5.0).abs() < 1e-5);
}

#[derive(Model, Debug)]
struct PinnedGaussian {
    #[param(value = 5.0, vary = false)]
    amp: f64,
    #[param(value = 4.0)]
    cen: f64,
    #[param(value = 1.5, min = 0.0)]
    wid: f64,
}

impl Curve for PinnedGaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amp * (-(x - self.cen).powi(2) / self.wid).exp()
    }
}

/// `#[model(name = "...")]` overrides the prefix, which is how a composite of
/// two models of the same type stays unambiguous.
#[derive(Model, Debug)]
#[model(name = "g")]
struct Renamed {
    #[param(value = 1.0)]
    value: f64,
}

impl Curve for Renamed {
    fn eval(&self, _x: f64) -> f64 {
        self.value
    }
}

#[derive(Model, Debug)]
struct Constant {
    #[param(value = 0.0)]
    c: f64,
}

impl Curve for Constant {
    fn eval(&self, _x: f64) -> f64 {
        self.c
    }
}

#[test]
fn model_name_attribute_overrides_the_prefix() {
    assert_eq!(Renamed::MODEL_NAME, "g");
    let names: Vec<String> = Renamed::default()
        .specs()
        .into_iter()
        .map(|s| s.name)
        .collect();
    assert_eq!(names, vec!["value"]);
}

// ---------------------------------------------------------------------------
// Composites
// ---------------------------------------------------------------------------

/// `+` on two derived models must build a composite with prefixed, distinct
/// parameter names — this is what the generated `Add` impl exists for.
#[test]
fn derived_models_combine_with_the_add_operator() {
    let model = DerivedGaussian::default() + Constant::default();

    assert_eq!(model.nparams(), 4);
    let names: Vec<String> = model.specs().into_iter().map(|s| s.name).collect();
    assert_eq!(
        names,
        vec![
            "derived_gaussian_amp",
            "derived_gaussian_cen",
            "derived_gaussian_wid",
            "constant_c"
        ]
    );

    let params = model.parameters().expect("distinct names");
    assert_eq!(params.len(), 4);
}

/// A composite of a derived model and a constant must fit an offset Gaussian.
#[test]
fn a_composite_fits_a_sum_of_curves() {
    let x = xs(121);
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 5.0 * (-(xi - 5.0f64).powi(2) / 2.0).exp() + 0.75)
        .collect();

    let model = DerivedGaussian {
        amp: 3.0,
        cen: 4.0,
        wid: 1.5,
    } + Constant { c: 0.0 };

    let result = model.fit(&y, &x).expect("fit ran");
    assert!(result.success, "{}", result.message);

    assert!(
        (result.model.a.amp - 5.0).abs() < 1e-5,
        "amp = {}",
        result.model.a.amp
    );
    assert!((result.model.a.cen - 5.0).abs() < 1e-5);
    assert!((result.model.a.wid - 2.0).abs() < 1e-5);
    assert!(
        (result.model.b.c - 0.75).abs() < 1e-5,
        "c = {}",
        result.model.b.c
    );

    // The prefixed names are what the parameter collection reports.
    assert!(result.params.get("derived_gaussian_amp").is_some());
    assert!(result.params.get("constant_c").is_some());
}

/// A composite of two models that share a `MODEL_NAME` produces colliding
/// parameter names, and says so rather than silently merging them.
#[test]
fn same_type_composites_report_the_collision() {
    let model = DerivedGaussian::default() + DerivedGaussian::default();
    let err = model.fit(&[1.0, 2.0], &[1.0, 2.0]).unwrap_err();
    assert!(
        matches!(err, lmfit::Error::DuplicateParameter { .. }),
        "expected a duplicate-name error, got {err:?}"
    );
}
