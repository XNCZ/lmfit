//! 复数拟合:端到端与统计口径测试。
//!
//! 复数残差按实部、虚部逐点交错进入实数最小二乘(与 numpy `view(float)`
//! 同序);统计口径与 lmfit 对齐:`chisqr = Σ|r_i|²`,`ndata = 2n`。

use lmfit::{Complex64, ComplexCurve, Model};

/// 复高斯:实数四参数,相位因子产生虚部。
#[derive(Model, Debug)]
struct ComplexGaussian {
    #[param(value = 5.0)]
    amplitude: f64,
    #[param(value = 5.0)]
    center: f64,
    #[param(value = 2.0, min = 0.0)]
    sigma: f64,
    #[param(value = 0.3)]
    phase: f64,
}

impl ComplexCurve for ComplexGaussian {
    fn eval(&self, x: Complex64) -> Complex64 {
        let e = (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp();
        e * self.amplitude * Complex64::from_polar(1.0, self.phase)
    }
}

/// 采样点横跨峰位。
fn xs(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 * 10.0 / (n - 1) as f64).collect()
}

/// 确定性伪噪声在 [-0.5, 0.5)。
fn noise(i: usize) -> f64 {
    let h = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((h >> 33) as f64 / (1u64 << 31) as f64) - 0.5
}

/// 数据生成函数,与模型的 `eval` 保持独立。
fn truth(x: f64) -> Complex64 {
    let g = 5.0 * (-(x - 5.0f64).powi(2) / 8.0).exp();
    Complex64::from_polar(g, 0.3)
}

/// 与数据重叠的起点。
fn start() -> ComplexGaussian {
    ComplexGaussian {
        amplitude: 4.0,
        center: 4.0,
        sigma: 1.5,
        phase: 0.0,
    }
}

/// 两个正交分量各自注入确定性噪声。
fn noisy_data(x: &[f64], scale: f64) -> Vec<Complex64> {
    x.iter()
        .enumerate()
        .map(|(i, &t)| truth(t) + Complex64::new(scale * noise(i), scale * noise(i + 7)))
        .collect()
}

#[test]
fn recovers_known_parameters_from_noiseless_complex_data() {
    let x = xs(101);
    let y: Vec<Complex64> = x.iter().map(|&t| truth(t)).collect();

    let result = match start().fit(&y, &x) {
        Ok(r) => r,
        Err(e) => panic!("fit failed: {e}"),
    };

    assert!(result.success, "{}", result.message);
    assert!(
        (result.model.amplitude - 5.0).abs() < 1e-6,
        "amplitude = {}",
        result.model.amplitude
    );
    assert!((result.model.center - 5.0).abs() < 1e-6, "center = {}", result.model.center);
    assert!((result.model.sigma - 2.0).abs() < 1e-6, "sigma = {}", result.model.sigma);
    assert!((result.model.phase - 0.3).abs() < 1e-6, "phase = {}", result.model.phase);
}

#[test]
fn chisqr_sums_both_quadratures_and_ndata_doubles() {
    let x = xs(201);
    let y = noisy_data(&x, 0.05);

    let result = match start().fit(&y, &x) {
        Ok(r) => r,
        Err(e) => panic!("fit failed: {e}"),
    };
    assert!(result.success, "{}", result.message);

    // 口径:chisqr 为两侧平方和,ndata 为槽位数(2n)。
    let manual: f64 = result
        .y
        .iter()
        .zip(&result.best_fit)
        .map(|(observed, predicted)| (*observed - *predicted).norm_sqr())
        .sum();
    assert!(
        (result.chisqr - manual).abs() <= 1e-9 * (1.0 + manual),
        "chisqr {} vs 手算 {}",
        result.chisqr,
        manual
    );
    assert_eq!(result.ndata, 2 * result.y.len());
    assert_eq!(result.nfree, result.ndata - result.nvarys);
}

#[test]
fn stderr_is_some_for_varied_complex_fit() {
    let x = xs(201);
    let y = noisy_data(&x, 0.05);

    let result = match start().fit(&y, &x) {
        Ok(r) => r,
        Err(e) => panic!("fit failed: {e}"),
    };

    for (i, se) in result.stderr.iter().enumerate() {
        match se {
            Some(s) => assert!(*s > 0.0, "stderr[{i}] = {s}"),
            None => panic!("stderr[{i}] 应为 Some"),
        }
    }
    match &result.covar {
        Some(c) => assert_eq!(c.matrix.len(), 16), // nvarys = 4
        None => panic!("应有协方差"),
    }
}

/// 除零模型:在 x = 5 处产生非有限残差。
#[derive(Model, Debug)]
struct Singular {
    #[param(value = 1.0)]
    scale: f64,
}

impl ComplexCurve for Singular {
    fn eval(&self, x: Complex64) -> Complex64 {
        self.scale / (x - 5.0)
    }
}

#[test]
fn a_non_finite_complex_residual_names_the_data_point() {
    let x = vec![1.0, 2.0, 5.0, 7.0];
    let y = vec![Complex64::new(0.0, 0.0); 4];

    let result = match Singular::default().fit(&y, &x) {
        Ok(r) => r,
        Err(e) => panic!("fit returned an error: {e}"),
    };

    assert!(!result.success, "message = {}", result.message);
    assert!(
        result.message.contains("data point 2"),
        "message = {}",
        result.message
    );
}

#[test]
fn mismatched_complex_input_is_rejected() {
    let y = vec![Complex64::new(1.0, 0.0); 3];
    let x = vec![0.0, 1.0];

    let err = match ComplexGaussian::default().fit(&y, &x) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(matches!(err, lmfit::Error::DimensionMismatch { x: 2, y: 3 }));
}

/// 复平面极点:`a / (x - p)`,极点的实部虚部都是拟合参数。
#[derive(Model, Debug)]
struct Pole {
    #[param(value = 1.0)]
    amplitude: f64,
    #[param(value = 5.0)]
    pole_re: f64,
    #[param(value = 1.0)]
    pole_im: f64,
}

impl ComplexCurve for Pole {
    fn eval(&self, x: Complex64) -> Complex64 {
        Complex64::new(self.amplitude, 0.0) / (x - Complex64::new(self.pole_re, self.pole_im))
    }
}

/// 复数自变量采样下的极点恢复:实轴网格叠加固定虚部偏移。
#[test]
fn recovers_a_pole_in_the_complex_plane() {
    let x: Vec<Complex64> = (0..101)
        .map(|i| Complex64::new(i as f64 / 10.0, 0.25))
        .collect();
    let truth = Pole {
        amplitude: 2.0,
        pole_re: 5.0,
        pole_im: 0.8,
    };
    let y: Vec<Complex64> = x.iter().map(|&t| truth.eval(t)).collect();

    let start = Pole {
        amplitude: 1.5,
        pole_re: 4.5,
        pole_im: 0.5,
    };
    let result = match start.fit(&y, &x) {
        Ok(r) => r,
        Err(e) => panic!("fit failed: {e}"),
    };

    assert!(result.success, "{}", result.message);
    assert!(
        (result.model.amplitude - 2.0).abs() < 1e-6,
        "amplitude = {}",
        result.model.amplitude
    );
    assert!(
        (result.model.pole_re - 5.0).abs() < 1e-6,
        "pole_re = {}",
        result.model.pole_re
    );
    assert!(
        (result.model.pole_im - 0.8).abs() < 1e-6,
        "pole_im = {}",
        result.model.pole_im
    );
}

#[test]
fn complex_fit_report_has_all_sections_and_stderr() {
    let x = xs(201);
    let y = noisy_data(&x, 0.05);

    let result = match start().fit(&y, &x) {
        Ok(r) => r,
        Err(e) => panic!("fit failed: {e}"),
    };
    let report = result.to_string();

    assert!(report.contains("[[Model]]"), "{report}");
    assert!(report.contains("[[Fit Statistics]]"), "{report}");
    assert!(report.contains("[[Variables]]"), "{report}");

    let amp_line = match report
        .lines()
        .find(|l| l.trim_start().starts_with("amplitude:"))
    {
        Some(l) => l,
        None => panic!("amplitude line missing in\n{report}"),
    };
    assert!(amp_line.contains("+/-"), "{amp_line}");
}
