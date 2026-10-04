//! 加权拟合:σ 通道的端到端语义与错误路径。

use lmfit::{ComplexCurve, Curve, Model};

#[derive(Model, Debug)]
struct Line {
    #[param(value = 0.0)]
    slope: f64,
    #[param(value = 0.0)]
    intercept: f64,
}

impl Curve for Line {
    fn eval(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

/// σ 小的点必须把直线拉过去:第一点 (0, 0) 测得极准,后两点则把截距往上拉。
#[test]
fn a_certain_point_pins_the_line() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 2.0, 5.0];

    let plain = Line::default().fit(&y, &x).expect("fit ran");
    assert!(
        plain.model.intercept.abs() > 0.1,
        "不加权时截距本应被后两点拉走,实际 = {}",
        plain.model.intercept
    );

    let sigma = vec![1.0e-4, 1.0, 1.0];
    let weighted = Line::default().fit_sigma(&y, &x, &sigma).expect("fit ran");
    assert!(
        weighted.model.intercept.abs() < 1.0e-3,
        "加权后直线应几乎穿过 (0, 0),实际截距 = {}",
        weighted.model.intercept
    );
}

/// 同一条直线的两份定义:一份只走有限差分,一份附带解析偏导。
#[derive(Model, Debug)]
struct FdLine {
    #[param(value = 0.0)]
    slope: f64,
    #[param(value = 0.0)]
    intercept: f64,
}

impl Curve for FdLine {
    fn eval(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

#[derive(Model, Debug)]
struct AnLine {
    #[param(value = 0.0)]
    slope: f64,
    #[param(value = 0.0)]
    intercept: f64,
}

impl Curve for AnLine {
    fn eval(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    fn partials_at(&self, x: f64) -> Option<impl lmfit::PartialValues<Scalar = f64>> {
        Some(AnLinePartials {
            slope: x,
            intercept: 1.0,
        })
    }
}

/// 异方差数据上的黄金:加权直线回归有闭式解 `(XᵀWX)⁻¹XᵀWy`,手算对照。
#[test]
fn weighted_line_matches_the_closed_form() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.9, 2.2, 4.9, 6.8, 9.3];
    let sigma = vec![0.5, 0.05, 1.5, 0.1, 2.0];

    // 闭式解:2×2 正规方程显式求逆。
    let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for ((&xi, &yi), &si) in x.iter().zip(&y).zip(&sigma) {
        let w = 1.0 / (si * si);
        sw += w;
        sx += w * xi;
        sy += w * yi;
        sxx += w * xi * xi;
        sxy += w * xi * yi;
    }
    let det = sw * sxx - sx * sx;
    let slope = (sw * sxy - sx * sy) / det;
    let intercept = (sxx * sy - sx * sxy) / det;

    let r = Line::default().fit_sigma(&y, &x, &sigma).expect("fit ran");
    assert!((r.model.slope - slope).abs() < 1e-10, "{} vs {slope}", r.model.slope);
    assert!(
        (r.model.intercept - intercept).abs() < 1e-10,
        "{} vs {intercept}",
        r.model.intercept
    );

    // χ² 也是加权口径:Σ((y − ŷ)/σ)²。
    let chi2: f64 = x
        .iter()
        .zip(&y)
        .zip(&sigma)
        .map(|((&xi, &yi), &si)| {
            let resid = yi - (slope * xi + intercept);
            (resid / si) * (resid / si)
        })
        .sum();
    assert!((r.chisqr - chi2).abs() < 1e-12 * chi2.abs().max(1.0));
}

/// 解析偏导通道必须与有限差分通道在加权下收敛到同一点——漏乘 1/σ 会在这里现形。
#[test]
fn analytic_and_finite_difference_paths_agree_under_weights() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.1, 2.4, 4.2, 7.1, 8.0];
    let sigma = vec![1.0, 0.05, 1.0, 0.05, 1.0];

    let fd = FdLine::default().fit_sigma(&y, &x, &sigma).expect("fd fit");
    let an = AnLine::default().fit_sigma(&y, &x, &sigma).expect("an fit");

    // 两条路必须落在同一个极小点上:χ² 是目标函数,可严判;参数受停止容差影响,
    // 在平坦极小值上本就有 ~1e-9 的抖动,故放到 1e-7。漏乘 1/σ 时 χ² 会差 O(1)。
    assert!(
        (fd.chisqr - an.chisqr).abs() < 1e-12 * fd.chisqr.abs().max(1.0),
        "解析 χ² {} vs 差分 χ² {}",
        an.chisqr,
        fd.chisqr
    );
    assert!(
        (fd.model.slope - an.model.slope).abs() < 1e-7,
        "解析 {} vs 差分 {}",
        an.model.slope,
        fd.model.slope
    );
    assert!((fd.model.intercept - an.model.intercept).abs() < 1e-7);
}

/// σ 全相等时:参数与协方差与不加权完全一致,只有 χ²/redchi 按 1/σ² 缩放。
#[test]
fn uniform_sigma_only_rescales_the_statistics() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![1.0, 3.1, 5.2, 6.9];
    let sigma = vec![0.25; 4];

    let plain = Line::default().fit(&y, &x).expect("fit ran");
    let weighted = Line::default()
        .fit_sigma(&y, &x, &sigma)
        .expect("fit ran");

    assert!((plain.model.slope - weighted.model.slope).abs() < 1e-10);
    assert!((plain.chisqr - weighted.chisqr * 0.25 * 0.25).abs() < 1e-10);
    for (i, (a, b)) in plain.params.iter().zip(weighted.params.iter()).enumerate() {
        match (a.stderr, b.stderr) {
            (Some(va), Some(vb)) => assert!((va - vb).abs() < 1e-10, "stderr[{i}]"),
            (None, None) => {}
            (x, y) => panic!("stderr[{i}] 口径不一致: {x:?} vs {y:?}"),
        }
    }
}

/// 复数通道的同一条守门测试:复高斯带相位,含噪异方差数据。
#[derive(Model, Debug)]
struct FdComplexGaussian {
    #[param(value = 5.0)]
    amplitude: f64,
    #[param(value = 5.0)]
    center: f64,
    #[param(value = 2.0, min = 0.0)]
    sigma: f64,
    #[param(value = 0.3)]
    phase: f64,
}

impl lmfit::ComplexCurve for FdComplexGaussian {
    fn eval(&self, x: lmfit::Complex64) -> lmfit::Complex64 {
        let e = (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp();
        e * self.amplitude * lmfit::Complex64::from_polar(1.0, self.phase)
    }
}

#[derive(Model, Debug)]
struct AnComplexGaussian {
    #[param(value = 5.0)]
    amplitude: f64,
    #[param(value = 5.0)]
    center: f64,
    #[param(value = 2.0, min = 0.0)]
    sigma: f64,
    #[param(value = 0.3)]
    phase: f64,
}

impl lmfit::ComplexCurve for AnComplexGaussian {
    fn eval(&self, x: lmfit::Complex64) -> lmfit::Complex64 {
        let e = (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp();
        e * self.amplitude * lmfit::Complex64::from_polar(1.0, self.phase)
    }

    fn partials_at(
        &self,
        x: lmfit::Complex64,
    ) -> Option<impl lmfit::PartialValues<Scalar = lmfit::Complex64>> {
        use lmfit::Complex64 as C;
        let d = x - self.center;
        let s2 = self.sigma * self.sigma;
        let e = (-d * d / (2.0 * s2)).exp();
        let p = C::from_polar(1.0, self.phase);
        Some(AnComplexGaussianPartials {
            amplitude: e * p,
            center: e * p * self.amplitude * d / s2,
            sigma: e * p * self.amplitude * d * d / (s2 * self.sigma),
            phase: C::new(0.0, 1.0) * e * self.amplitude * p,
        })
    }
}

#[test]
fn complex_paths_agree_under_weights() {
    let x: Vec<f64> = (0..21).map(|i| i as f64 * 0.5).collect();
    // 确定性伪噪声,两端比峰附近糙一个量级。
    let mut seed = 7u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    let truth = AnComplexGaussian {
        amplitude: 5.0,
        center: 5.0,
        sigma: 2.0,
        phase: 0.3,
    };
    let y: Vec<lmfit::Complex64> = x
        .iter()
        .map(|&t| truth.eval(lmfit::Complex64::new(t, 0.0)) + lmfit::Complex64::new(0.02 * next(), 0.02 * next()))
        .collect();
    let sigma: Vec<f64> = x.iter().map(|&t| if (t - 5.0).abs() > 2.0 { 0.2 } else { 0.02 }).collect();

    let fd = FdComplexGaussian::default()
        .fit_sigma(&y, &x, &sigma)
        .expect("fd fit");
    let an = AnComplexGaussian::default()
        .fit_sigma(&y, &x, &sigma)
        .expect("an fit");

    assert!(
        (fd.chisqr - an.chisqr).abs() < 1e-10 * fd.chisqr.abs().max(1.0),
        "解析 χ² {} vs 差分 χ² {}",
        an.chisqr,
        fd.chisqr
    );
    assert!((fd.model.amplitude - an.model.amplitude).abs() < 1e-6);
    assert!((fd.model.center - an.model.center).abs() < 1e-6);
    assert!((fd.model.sigma - an.model.sigma).abs() < 1e-6);
    assert!((fd.model.phase - an.model.phase).abs() < 1e-6);
}

/// σ 长度必须等于数据点数。
#[test]
fn sigma_length_must_match_the_data() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 2.0];
    let sigma = vec![1.0, 1.0];
    match Line::default().fit_sigma(&y, &x, &sigma) {
        Err(lmfit::Error::SigmaMismatch { npoints, sigma }) => {
            assert_eq!(npoints, 3);
            assert_eq!(sigma, 2);
        }
        other => panic!("expected SigmaMismatch, got {other:?}"),
    }
}

/// σ 必须是有限的正数:零、负数、NaN、无穷都要被挡下,并指出是第几项。
#[test]
fn sigma_must_be_finite_and_positive() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 2.0];
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let sigma = vec![1.0, bad, 1.0];
        match Line::default().fit_sigma(&y, &x, &sigma) {
            Err(lmfit::Error::InvalidSigma { index, value }) => {
                assert_eq!(index, 1, "bad = {bad}");
                assert_eq!(value.is_nan(), bad.is_nan(), "bad = {bad}");
            }
            other => panic!("expected InvalidSigma for {bad}, got {other:?}"),
        }
    }
}
