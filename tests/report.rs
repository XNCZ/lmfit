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
    let report = fit_gaussian().to_string();

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
    let report = fit_gaussian().to_string();
    let block: String = report
        .lines()
        .skip_while(|l| !l.starts_with("[[Fit Statistics]]"))
        .take_while(|l| !l.starts_with("[[Variables]]"))
        .collect::<Vec<_>>()
        .join("\n");

    let columns = equals_columns(&block);
    assert_eq!(columns.len(), 9, "expected nine statistic rows");
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
    let report = Wide::default().fit(&y, &x).expect("fit ran").to_string();

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
    let report = result.to_string();

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

    let report = Pinned::default().fit(&y, &x).expect("fit ran").to_string();

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

/// The `[[Model]]` line carries the model's name, so reports for different
/// models are distinguishable at a glance.
#[test]
fn model_line_carries_the_model_name() {
    let report = fit_gaussian().to_string();
    assert!(
        report.contains("[[Model]]\n    gaussian"),
        "model name missing from\n{report}"
    );
}

/// Format a float the way the report does, for use in assertions.
fn trim(v: &f64) -> String {
    if *v == v.trunc() && v.abs() < 1.0e5 {
        format!("{}", *v as i64)
    } else {
        format!("{v}")
    }
}

/// stderr 出现在变量行;固定参数仍显示 (fixed),无 +/-。
#[test]
fn variables_show_stderr_when_present() {
    let report = fit_gaussian().to_string();
    let amp_line = match report
        .lines()
        .find(|l| l.trim_start().starts_with("amp:"))
    {
        Some(l) => l,
        None => panic!("amp line missing"),
    };
    assert!(amp_line.contains("+/-"), "{amp_line}");
}

/// 峰与本底模型,示例同款:本底与峰参数的交换关系产生确定性强相关。
#[derive(Model, Debug)]
struct PeakOnBackground {
    #[param(value = 4.0)]
    amplitude: f64,
    #[param(value = 4.0)]
    center: f64,
    #[param(value = 1.0, min = 0.0)]
    sigma: f64,
    #[param(value = 0.0)]
    background: f64,
}

impl Curve for PeakOnBackground {
    fn eval(&self, x: f64) -> f64 {
        lmfit::lineshapes::gaussian(x, self.amplitude, self.center, self.sigma) + self.background
    }
}

/// 相关块:表头与 C(a, b) = 值 行;无噪精确拟合的奇偶对称使相关全低于
/// 0.1 阈值,故此处用带本底与噪声的数据(示例同款,相关确定性强)。
#[test]
fn correlations_block_lists_qualified_pairs() {
    let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let h = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let n = ((h >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            lmfit::lineshapes::gaussian(t, 5.0, 4.5, 0.8) + 0.25 + 0.05 * n
        })
        .collect();
    let result = match PeakOnBackground::default().fit(&y, &x) {
        Ok(r) => r,
        Err(e) => panic!("fit failed: {e}"),
    };
    let report = result.to_string();
    assert!(
        report.contains("[[Correlations]] (unreported correlations are < 0.100)"),
        "{report}"
    );
    assert!(
        report.contains("C(amplitude, sigma)") || report.contains("C(sigma, amplitude)"),
        "{report}"
    );

    // `C(a, b)` 按最长标签补齐,`=` 与报表其余两块一样成列。
    let block: String = report
        .lines()
        .skip_while(|l| !l.starts_with("[[Correlations]]"))
        .collect::<Vec<_>>()
        .join("\n");
    let columns = equals_columns(&block);
    assert!(
        columns.len() >= 2,
        "need at least two reported pairs to test alignment:\n{report}"
    );
    assert!(
        columns.windows(2).all(|w| w[0] == w[1]),
        "correlation equals signs are not aligned: {columns:?}\n{report}"
    );
}

/// 统计块的 `=` 列必须对齐,取某段块内所有 ` = ` 的列号。
fn block_equals(report: &str) -> Vec<usize> {
    let block: String = report
        .lines()
        .skip_while(|l| !l.starts_with("[[Fit Statistics]]"))
        .take_while(|l| !l.starts_with("[[Variables]]"))
        .collect::<Vec<_>>()
        .join("\n");
    equals_columns(&block)
}

/// 收敛的拟合:`state` 为 success,且不带 `reason` 行。
#[test]
fn successful_fit_reports_state_without_reason() {
    let report = fit_gaussian().to_string();
    let state = match report.lines().find(|l| l.trim_start().starts_with("state")) {
        Some(line) => line,
        None => panic!("no state line in\n{report}"),
    };
    assert!(state.trim_end().ends_with("= success"), "{state}");
    assert!(
        !report.lines().any(|l| l.trim_start().starts_with("reason")),
        "successful fit must not carry a reason line:\n{report}"
    );
}

/// 残差出现非有限值的模型:拟合以失败收场(而非 Err)。
#[derive(Model, Debug)]
struct Exploding {
    #[param(value = 1.0)]
    slope: f64,
}

impl Curve for Exploding {
    fn eval(&self, x: f64) -> f64 {
        if x > 5.0 {
            f64::NAN
        } else {
            self.slope * x
        }
    }
}

/// 失败的拟合:`state` 为 failure,紧跟一行 `reason`,且对齐不被长文本破坏。
#[test]
fn failed_fit_reports_state_and_reason() {
    let x: Vec<f64> = (0..11).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&t| t).collect();
    let result = Exploding::default().fit(&y, &x).expect("fit returns a result");
    assert!(!result.success, "a NaN residual cannot converge");

    let report = result.to_string();
    let lines: Vec<&str> = report.lines().collect();
    let idx = match lines.iter().position(|l| l.trim_start().starts_with("state")) {
        Some(i) => i,
        None => panic!("no state line in\n{report}"),
    };
    assert!(
        lines[idx].trim_end().ends_with("= failure"),
        "{}",
        lines[idx]
    );
    assert!(
        lines[idx + 1].trim_start().starts_with("reason"),
        "reason must follow state:\n{report}"
    );
    assert!(lines[idx + 1].contains("non-finite"), "{}", lines[idx + 1]);

    let columns = block_equals(&report);
    assert_eq!(columns.len(), 10, "expected ten rows when a reason is present");
    assert!(
        columns.windows(2).all(|w| w[0] == w[1]),
        "equals signs are not aligned: {columns:?}"
    );
}
