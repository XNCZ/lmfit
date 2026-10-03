//! 解析偏导通道端到端:与有限差分版对拍、回退与失败语义、复数通道。

use lmfit::{Complex64, ComplexCurve, Curve, Model, PartialValues};

/// 仅有限差分(不覆写 `partials_at`),作为对照。
#[derive(Model, Debug)]
struct FdGaussian {
    #[param(value = 3.0)]
    amplitude: f64,
    #[param(value = 4.0)]
    center: f64,
    #[param(value = 1.5, min = 0.0)]
    sigma: f64,
}

impl Curve for FdGaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amplitude * (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp()
    }
}

/// 同一公式,附带解析偏导。
#[derive(Model, Debug)]
struct AnGaussian {
    #[param(value = 3.0)]
    amplitude: f64,
    #[param(value = 4.0)]
    center: f64,
    #[param(value = 1.5, min = 0.0)]
    sigma: f64,
}

impl Curve for AnGaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amplitude * (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp()
    }

    fn partials_at(&self, x: f64) -> Option<impl PartialValues<Scalar = f64>> {
        let d = x - self.center;
        let s2 = self.sigma * self.sigma;
        let e = (-d * d / (2.0 * s2)).exp();
        Some(AnGaussianPartials {
            amplitude: e,
            center: self.amplitude * e * d / s2,
            sigma: self.amplitude * e * d * d / (s2 * self.sigma),
        })
    }
}

/// 无噪声高斯样本(真值 amp=5, center=5, sigma=2)。
fn data() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..101).map(|i| i as f64 * 10.0 / 100.0).collect();
    let y = x
        .iter()
        .map(|&xi| 5.0 * (-(xi - 5.0).powi(2) / 8.0).exp())
        .collect();
    (x, y)
}

#[test]
fn analytic_partials_match_the_finite_difference_fit() {
    let (x, y) = data();
    let fd = FdGaussian {
        amplitude: 3.0,
        center: 4.0,
        sigma: 1.5,
    }
    .fit(&y, &x)
    .expect("fd fit ran");
    let an = AnGaussian {
        amplitude: 3.0,
        center: 4.0,
        sigma: 1.5,
    }
    .fit(&y, &x)
    .expect("analytic fit ran");

    assert!(fd.success, "{}", fd.message);
    assert!(an.success, "{}", an.message);
    assert!((an.model.amplitude - fd.model.amplitude).abs() < 1e-6);
    assert!((an.model.center - fd.model.center).abs() < 1e-6);
    assert!((an.model.sigma - fd.model.sigma).abs() < 1e-6);

    // 解析路径不产生探测求值:nfev 显著更低。
    assert!(
        an.nfev < fd.nfev,
        "analytic nfev={} fd nfev={}",
        an.nfev,
        fd.nfev
    );
}

/// 带活动下界的对照:最优在界外,两个版本应落在同一停止邻域,且都不越界。
#[derive(Model, Debug)]
struct FdBounded {
    #[param(value = 5.0)]
    amplitude: f64,
    #[param(value = 6.5, min = 5.5)]
    center: f64,
    #[param(value = 1.5)]
    sigma: f64,
}

impl Curve for FdBounded {
    fn eval(&self, x: f64) -> f64 {
        self.amplitude * (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp()
    }
}

#[derive(Model, Debug)]
struct AnBounded {
    #[param(value = 5.0)]
    amplitude: f64,
    #[param(value = 6.5, min = 5.5)]
    center: f64,
    #[param(value = 1.5)]
    sigma: f64,
}

impl Curve for AnBounded {
    fn eval(&self, x: f64) -> f64 {
        self.amplitude * (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp()
    }

    fn partials_at(&self, x: f64) -> Option<impl PartialValues<Scalar = f64>> {
        let d = x - self.center;
        let s2 = self.sigma * self.sigma;
        let e = (-d * d / (2.0 * s2)).exp();
        Some(AnBoundedPartials {
            amplitude: e,
            center: self.amplitude * e * d / s2,
            sigma: self.amplitude * e * d * d / (s2 * self.sigma),
        })
    }
}

#[test]
fn analytic_partials_respect_bounds() {
    let (x, y) = data();
    let fd = FdBounded {
        amplitude: 5.0,
        center: 6.5,
        sigma: 1.5,
    }
    .fit(&y, &x)
    .expect("fd fit ran");
    let an = AnBounded {
        amplitude: 5.0,
        center: 6.5,
        sigma: 1.5,
    }
    .fit(&y, &x)
    .expect("analytic fit ran");

    assert!(fd.success && an.success);
    // 最优在界外:两个版本都被下界挡住,且都不越界。
    assert!(an.model.center >= 5.5 - 1e-9, "center = {}", an.model.center);
    assert!(
        (an.model.center - fd.model.center).abs() < 1e-2,
        "analytic center={} fd center={}",
        an.model.center,
        fd.model.center
    );
    assert!((an.model.amplitude - fd.model.amplitude).abs() < 1e-6);
}

/// 半途拒供(数据横跨 x=5,任一张雅可比都会撞上 `None`):整张退回 FD,
/// 结果与纯 FD 版逐位一致。
#[derive(Model, Debug)]
struct HalfDeclineGaussian {
    #[param(value = 3.0)]
    amplitude: f64,
    #[param(value = 4.0)]
    center: f64,
    #[param(value = 1.5, min = 0.0)]
    sigma: f64,
}

impl Curve for HalfDeclineGaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amplitude * (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp()
    }

    fn partials_at(&self, x: f64) -> Option<impl PartialValues<Scalar = f64>> {
        if x > 5.0 {
            return None;
        }
        let d = x - self.center;
        let s2 = self.sigma * self.sigma;
        let e = (-d * d / (2.0 * s2)).exp();
        Some(HalfDeclineGaussianPartials {
            amplitude: e,
            center: self.amplitude * e * d / s2,
            sigma: self.amplitude * e * d * d / (s2 * self.sigma),
        })
    }
}

#[test]
fn a_mid_way_decline_falls_back_to_finite_difference() {
    let (x, y) = data();
    let fd = FdGaussian {
        amplitude: 3.0,
        center: 4.0,
        sigma: 1.5,
    }
    .fit(&y, &x)
    .expect("fd fit ran");
    let half = HalfDeclineGaussian {
        amplitude: 3.0,
        center: 4.0,
        sigma: 1.5,
    }
    .fit(&y, &x)
    .expect("fallback fit ran");

    assert!(half.success, "{}", half.message);
    assert_eq!(half.nfev, fd.nfev, "整张回退后应与纯 FD 路径同计数");
    assert!((half.model.amplitude - fd.model.amplitude).abs() < 1e-12);
    assert!((half.model.center - fd.model.center).abs() < 1e-12);
    assert!((half.model.sigma - fd.model.sigma).abs() < 1e-12);
}

/// 非有限偏导:与有限差分的非有限语义一致,判雅可比失败。
#[derive(Model, Debug)]
struct NonFiniteGaussian {
    #[param(value = 3.0)]
    amplitude: f64,
    #[param(value = 4.0)]
    center: f64,
    #[param(value = 1.5, min = 0.0)]
    sigma: f64,
}

impl Curve for NonFiniteGaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amplitude * (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp()
    }

    fn partials_at(&self, _x: f64) -> Option<impl PartialValues<Scalar = f64>> {
        Some(NonFiniteGaussianPartials {
            amplitude: f64::NAN,
            center: 0.0,
            sigma: 1.0,
        })
    }
}

#[test]
fn non_finite_partials_fail_the_jacobian() {
    let (x, y) = data();
    let result = NonFiniteGaussian {
        amplitude: 3.0,
        center: 4.0,
        sigma: 1.5,
    }
    .fit(&y, &x)
    .expect("fit ran");

    assert!(!result.success, "message = {}", result.message);
    assert!(
        result.message.contains("jacobian"),
        "message = {}",
        result.message
    );
}

// ---------------------------------------------------------------------------
// 复数通道
// ---------------------------------------------------------------------------

/// 复高斯(与 tests/complex.rs 同构):四个实参数,相位因子产生虚部。
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

impl ComplexCurve for FdComplexGaussian {
    fn eval(&self, x: Complex64) -> Complex64 {
        let e = (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp();
        e * self.amplitude * Complex64::from_polar(1.0, self.phase)
    }
}

/// 同一公式,附带复数解析偏导。
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

impl ComplexCurve for AnComplexGaussian {
    fn eval(&self, x: Complex64) -> Complex64 {
        let e = (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp();
        e * self.amplitude * Complex64::from_polar(1.0, self.phase)
    }

    fn partials_at(&self, x: Complex64) -> Option<impl PartialValues<Scalar = Complex64>> {
        let d = x - self.center;
        let s2 = self.sigma * self.sigma;
        let e = (-d.powi(2) / (2.0 * s2)).exp();
        let rot = Complex64::from_polar(1.0, self.phase);
        Some(AnComplexGaussianPartials {
            amplitude: e * rot,
            center: self.amplitude * e * d / s2 * rot,
            sigma: self.amplitude * e * d.powi(2) / (s2 * self.sigma) * rot,
            phase: Complex64::i() * self.amplitude * e * rot,
        })
    }
}

#[test]
fn analytic_complex_partials_match_the_finite_difference_fit() {
    let x: Vec<f64> = (0..101).map(|i| i as f64 * 10.0 / 100.0).collect();
    let truth = FdComplexGaussian {
        amplitude: 5.0,
        center: 5.0,
        sigma: 2.0,
        phase: 0.3,
    };
    let y: Vec<Complex64> = x
        .iter()
        .map(|&xi| truth.eval(Complex64::new(xi, 0.0)))
        .collect();

    let fd = FdComplexGaussian {
        amplitude: 4.0,
        center: 4.0,
        sigma: 1.5,
        phase: 0.0,
    }
    .fit(&y, &x)
    .expect("fd fit ran");
    let an = AnComplexGaussian {
        amplitude: 4.0,
        center: 4.0,
        sigma: 1.5,
        phase: 0.0,
    }
    .fit(&y, &x)
    .expect("analytic fit ran");

    assert!(fd.success, "{}", fd.message);
    assert!(an.success, "{}", an.message);
    assert!((an.model.amplitude - fd.model.amplitude).abs() < 1e-6);
    assert!((an.model.center - fd.model.center).abs() < 1e-6);
    assert!((an.model.sigma - fd.model.sigma).abs() < 1e-6);
    assert!((an.model.phase - fd.model.phase).abs() < 1e-6);
    assert!(
        an.nfev < fd.nfev,
        "analytic nfev={} fd nfev={}",
        an.nfev,
        fd.nfev
    );
}
