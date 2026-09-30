//! The fit report: structure, alignment, and the numbers it repeats back.
//!
//! Alignment is asserted by measurement rather than by a full-string golden.
//! A golden would fail on every legitimate change to a value, while telling us
//! nothing about the thing that actually breaks in a table like this — a
//! label of a novel length pushing its `=` out of column.

use lmfit::{Curve, Model, ModelResult};

#[derive(Model, Debug)]
struct Gaussian {
    #[param(value = 5.0)]
    amp: f64,
    #[param(value = 5.0)]
    cen: f64,
    #[param(value = 2.0, min = 0.0)]
    wid: f64,
}

impl Curve for Gaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amp * (-(x - self.cen).powi(2) / self.wid).exp()
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

/// A model with a deliberately long parameter name, to exercise column widths.
#[derive(Model, Debug)]
struct Wide {
    #[param(value = 1.0)]
    a: f64,
    #[param(value = 1.0)]
    a_considerably_longer_parameter_name: f64,
}

impl Curve for Wide {
    fn eval(&self, _x: f64) -> f64 {
        self.a + self.a_considerably_longer_parameter_name
    }
}

#[derive(Model, Debug)]
struct Pinned {
    #[param(value = 5.0, vary = false)]
    amp: f64,
    #[param(value = 4.0)]
    cen: f64,
    #[param(value = 1.5, min = 0.0)]
    wid: f64,
}

impl Curve for Pinned {
    fn eval(&self, x: f64) -> f64 {
        self.amp * (-(x - self.cen).powi(2) / self.wid).exp()
    }
}

fn fit_gaussian() -> ModelResult<Gaussian> {
    let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| 5.0 * (-(t - 5.0f64).powi(2) / 2.0).exp())
        .collect();
    Gaussian::default().fit(&y, &x).expect("fit ran")
}

/// Every `=` in the statistics block must sit in the same column.
fn equals_columns(block: &str) -> Vec<usize> {
    block.lines().filter_map(|line| line.find(" = ")).collect()
}

#[test]
fn report_has_the_expected_sections_in_order() {
    let report = fit_gaussian().fit_report();

    let model = report.find("[[Model]]").expect("model section");
    let stats = report
        .find("[[Fit Statistics]]")
        .expect("statistics section");
    let vars = report.find("[[Variables]]").expect("variables section");

    assert!(model < stats, "model section should come first");
    assert!(stats < vars, "statistics should precede variables");

    assert!(report.contains("leastsq"), "solver name missing");
}

#[test]
fn statistics_labels_line_up() {
    let report = fit_gaussian().fit_report();
    let block: String = report
        .lines()
        .skip_while(|l| !l.starts_with("[[Fit Statistics]]"))
        .take_while(|l| !l.starts_with("[[Variables]]"))
        .collect::<Vec<_>>()
        .join("\n");

    let columns = equals_columns(&block);
    assert_eq!(columns.len(), 8, "expected eight statistic rows");
    assert!(
        columns.windows(2).all(|w| w[0] == w[1]),
        "equals signs are not aligned: {columns:?}"
    );
}

/// A longer parameter name must widen the whole column, not just its own line.
#[test]
fn variable_values_line_up_whatever_the_name_length() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![3.0, 3.0, 3.0, 3.0, 3.0];
    let report = Wide::default().fit(&y, &x).expect("fit ran").fit_report();

    let lines: Vec<&str> = report
        .lines()
        .skip_while(|l| !l.starts_with("[[Variables]]"))
        .skip(1)
        .collect();

    assert_eq!(lines.len(), 2);
    let value_columns: Vec<usize> = lines
        .iter()
        .map(|l| l.find(" (init").expect("each line has an (init = ...)"))
        .collect();
    assert_eq!(
        value_columns[0],
        value_columns[1],
        "values not aligned:\n{}",
        lines.join("\n")
    );
}

#[test]
fn variables_show_the_fitted_value_and_the_starting_point() {
    let result = fit_gaussian();
    let report = result.fit_report();

    for name in ["amp", "cen", "wid"] {
        let line = report
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("{name}:")))
            .unwrap_or_else(|| panic!("no line for {name} in\n{report}"));
        assert!(
            line.contains("(init = 5)") || line.contains("(init = 2)"),
            "{line}"
        );
    }

    // The values reported are the fitted ones.
    assert!(report.contains(&format!("amp: {}", trim(&result.model.amp))));
}

/// A pinned parameter is marked as such rather than shown with a starting
/// point it never moved away from.
#[test]
fn fixed_parameters_are_marked() {
    let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| 5.0 * (-(t - 5.0f64).powi(2) / 2.0).exp())
        .collect();

    let report = Pinned::default().fit(&y, &x).expect("fit ran").fit_report();

    let amp_line = report
        .lines()
        .find(|l| l.trim_start().starts_with("amp:"))
        .expect("amp line");
    assert!(amp_line.ends_with("(fixed)"), "{amp_line}");

    let cen_line = report
        .lines()
        .find(|l| l.trim_start().starts_with("cen:"))
        .expect("cen line");
    assert!(cen_line.contains("(init = 4)"), "{cen_line}");
}

/// A composite names its structure rather than showing the generic prefix.
#[test]
fn composites_describe_their_structure() {
    let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| 5.0 * (-(t - 5.0f64).powi(2) / 2.0).exp() + 0.75)
        .collect();

    let model = Gaussian {
        amp: 4.0,
        cen: 4.0,
        wid: 1.5,
    } + Constant { c: 0.0 };

    let report = model.fit(&y, &x).expect("fit ran").fit_report();

    assert!(
        report.contains("(gaussian + constant)"),
        "composite description missing from\n{report}"
    );
    // And parameter names carry their prefixes.
    assert!(report.contains("gaussian_amp"));
    assert!(report.contains("constant_c"));
}

/// Format a float the way the report does, for use in assertions.
fn trim(v: &f64) -> String {
    if *v == v.trunc() && v.abs() < 1.0e5 {
        format!("{}", *v as i64)
    } else {
        format!("{v}")
    }
}
