//! 派生参数:值由同名方法给出,不参与拟合。

use lmfit::{ComplexCurve, Curve, Model, ModelParams};

/// 按名字取标准误:它与值同住在 `Parameter` 上。
fn stderr_of<M: ModelParams>(r: &lmfit::ModelResult<M>, name: &str) -> Option<f64> {
    r.params.get(name).expect("参数在表内").stderr
}

/// 直线:两个普通参数 + 两个派生量(和与积)。
#[derive(Model, Debug)]
struct Line {
    #[param(value = 1.0)]
    slope: f64,
    #[param(value = 1.0)]
    intercept: f64,
    #[param(derive)]
    sum: f64,
    #[param(derive)]
    product: f64,
}

impl Line {
    /// 派生量:斜率与截距之和。
    fn sum(&self) -> f64 {
        self.slope + self.intercept
    }

    /// 派生量:斜率与截距之积。
    fn product(&self) -> f64 {
        self.slope * self.intercept
    }
}

impl Curve for Line {
    fn eval(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

/// 同一模型的对照写法:不带派生字段。
#[derive(Model, Debug)]
struct PlainLine {
    #[param(value = 1.0)]
    slope: f64,
    #[param(value = 1.0)]
    intercept: f64,
}

impl Curve for PlainLine {
    fn eval(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

/// 确定性伪噪声直线样本:斜率 2、截距 3。
fn data() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..11).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let h = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let n = ((h >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            2.0 * t + 3.0 + 0.2 * n
        })
        .collect();
    (x, y)
}

#[test]
fn a_derived_parameter_equals_its_formula() {
    let (x, y) = data();
    let r = Line {
        slope: 1.0,
        intercept: 1.0,
        ..Line::default()
    }
    .fit(&y, &x)
    .expect("拟合");

    assert!((r.model.sum - (r.model.slope + r.model.intercept)).abs() < 1e-12);
    assert!((r.model.product - r.model.slope * r.model.intercept).abs() < 1e-12);
    // 字段是公式的缓存:库内任何写入之后两者一致。
    assert_eq!(r.model.sum, r.model.sum());
}

#[test]
fn adding_a_derived_parameter_does_not_move_the_fit() {
    let (x, y) = data();
    let plain = PlainLine {
        slope: 1.0,
        intercept: 1.0,
    }
    .fit(&y, &x)
    .expect("拟合");
    let derived = Line {
        slope: 1.0,
        intercept: 1.0,
        ..Line::default()
    }
    .fit(&y, &x)
    .expect("拟合");

    assert_eq!(plain.nvarys, derived.nvarys);
    assert_eq!(plain.nfev, derived.nfev);
    assert_eq!(plain.chisqr, derived.chisqr);
    assert_eq!(plain.model.slope, derived.model.slope);
    assert_eq!(plain.model.intercept, derived.model.intercept);
    assert_eq!(stderr_of(&plain, "slope"), stderr_of(&derived, "slope"));
    assert_eq!(
        stderr_of(&plain, "intercept"),
        stderr_of(&derived, "intercept")
    );
}

#[test]
fn a_derived_parameter_is_not_varied() {
    let (x, y) = data();
    let r = Line {
        slope: 1.0,
        intercept: 1.0,
        ..Line::default()
    }
    .fit(&y, &x)
    .expect("拟合");

    let p = r.params.get("sum").expect("派生参数在参数表内");
    assert!(!p.vary);
    assert!(p.derive);
    assert_eq!(r.nvarys, 2);
    assert_eq!(r.params.no_fix_indices(), vec![0, 1]);
    assert_eq!(r.params.len(), 4);
}

#[test]
fn writing_a_parameter_refreshes_the_derived_fields() {
    let mut m = Line {
        slope: 3.0,
        intercept: 4.0,
        ..Line::default()
    };
    m.set(0, 10.0);
    assert_eq!(m.sum, 10.0 + m.intercept);
    assert_eq!(m.product, 10.0 * m.intercept);

    // 写派生字段本身是空操作:重算随即覆盖它。
    m.set(2, 999.0);
    assert_eq!(m.sum, m.slope + m.intercept);
}

/// 链式:后者可以引用先声明的派生量。
#[derive(Model, Debug)]
struct Chained {
    #[param(value = 2.0)]
    a: f64,
    #[param(derive)]
    double: f64,
    #[param(derive)]
    quadruple: f64,
}

impl Chained {
    fn double(&self) -> f64 {
        2.0 * self.a
    }

    fn quadruple(&self) -> f64 {
        2.0 * self.double
    }
}

impl Curve for Chained {
    fn eval(&self, x: f64) -> f64 {
        self.a * x
    }
}

#[test]
fn derived_fields_chain_in_declaration_order() {
    let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&t| 3.0 * t).collect();
    let r = Chained {
        a: 1.0,
        ..Chained::default()
    }
    .fit(&y, &x)
    .expect("拟合");

    assert!((r.model.a - 3.0).abs() < 1e-9);
    assert_eq!(r.model.double, 2.0 * r.model.a);
    assert_eq!(r.model.quadruple, 4.0 * r.model.a);
}

/// delta 方法黄金:解析偏导 × 结果里的协方差,对比库传播出的标准误。
#[test]
fn derived_stderr_matches_the_delta_method() {
    let (x, y) = data();
    let r = Line {
        slope: 1.0,
        intercept: 1.0,
        ..Line::default()
    }
    .fit(&y, &x)
    .expect("拟合");
    let c = r.covar.as_ref().expect("协方差可得");

    // ∂sum/∂s = ∂sum/∂b = 1;∂product/∂s = b,∂product/∂b = s。
    let (s, b) = (r.model.slope, r.model.intercept);
    for (p, name) in [([1.0, 1.0], "sum"), ([b, s], "product")] {
        let mut var = 0.0;
        for j in 0..2 {
            for k in 0..2 {
                var += p[j] * p[k] * c.matrix[j * c.nvarys + k];
            }
        }
        let want = var.max(0.0).sqrt();
        let got = stderr_of(&r, name).expect("派生的标准误已传播");
        assert!(
            (got - want).abs() < 1e-6 * want,
            "{name}: {got} vs {want}"
        );
    }
}

/// 固定参数进公式:有值、无方差贡献。
#[derive(Model, Debug)]
struct FixedFeed {
    #[param(value = 1.0)]
    slope: f64,
    #[param(value = 1.0)]
    intercept: f64,
    #[param(value = 2.0, vary = false)]
    known: f64,
    #[param(derive)]
    scaled: f64,
}

impl FixedFeed {
    /// 派生量:把已知常数按斜率缩放。
    fn scaled(&self) -> f64 {
        self.known / self.slope
    }
}

impl Curve for FixedFeed {
    fn eval(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

#[test]
fn a_fixed_parameter_feeds_the_formula_without_adding_variance() {
    let (x, y) = data();
    let r = FixedFeed {
        slope: 1.0,
        intercept: 1.0,
        known: 2.0,
        ..FixedFeed::default()
    }
    .fit(&y, &x)
    .expect("拟合");

    assert_eq!(r.model.scaled, r.model.known / r.model.slope);
    // 固定参数自身没有标准误;派生量有,且等于 ∂/∂s = -known/s² 的单参数传播。
    assert_eq!(stderr_of(&r, "known"), None);
    let c = r.covar.as_ref().expect("协方差可得");
    let d = -r.model.known / (r.model.slope * r.model.slope);
    let want = (d * d * c.matrix[0]).max(0.0).sqrt();
    let got = stderr_of(&r, "scaled").expect("派生的标准误已传播");
    assert!((got - want).abs() < 1e-6 * want, "{got} vs {want}");
}

/// 复数通道的同一条守门测试。
#[derive(Model, Debug)]
struct ComplexLine {
    #[param(value = 1.0)]
    slope: f64,
    #[param(value = 1.0)]
    offset: f64,
    #[param(derive)]
    span: f64,
}

impl ComplexLine {
    fn span(&self) -> f64 {
        self.slope - self.offset
    }
}

impl lmfit::ComplexCurve for ComplexLine {
    fn eval(&self, x: lmfit::Complex64) -> lmfit::Complex64 {
        lmfit::Complex64::new(self.slope, self.offset) * x
    }
}

#[test]
fn complex_models_propagate_derived_stderr() {
    let x: Vec<f64> = (0..11).map(|i| i as f64).collect();
    let y: Vec<lmfit::Complex64> = x
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let h = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let re = ((h >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            let im = ((h >> 11) as f64 / (1u64 << 31) as f64) - 0.5;
            lmfit::Complex64::new(2.0 * t + 0.05 * re, 1.0 * t + 0.05 * im)
        })
        .collect();

    let r = ComplexLine {
        slope: 1.0,
        offset: 1.0,
        ..ComplexLine::default()
    }
    .fit(&y, &x)
    .expect("拟合");

    assert!((r.model.span - (r.model.slope - r.model.offset)).abs() < 1e-12);
    let c = r.covar.as_ref().expect("协方差可得");
    // ∂span/∂s = 1,∂span/∂o = -1。
    let p: [f64; 2] = [1.0, -1.0];
    let mut var = 0.0;
    for j in 0..2 {
        for k in 0..2 {
            var += p[j] * p[k] * c.matrix[j * c.nvarys + k];
        }
    }
    let want = var.max(0.0).sqrt();
    let got = r
        .params
        .get("span")
        .expect("参数在表内")
        .stderr
        .expect("派生的标准误已传播");
    assert!((got - want).abs() < 1e-6 * want, "{got} vs {want}");
}

/// 无变参数 → 无协方差 → 派生标准误同样为 None,而不是 0。
#[derive(Model, Debug)]
struct AllFixed {
    #[param(value = 2.0, vary = false)]
    a: f64,
    #[param(derive)]
    twice: f64,
}

impl AllFixed {
    fn twice(&self) -> f64 {
        2.0 * self.a
    }
}

impl Curve for AllFixed {
    fn eval(&self, x: f64) -> f64 {
        self.a * x
    }
}

#[test]
fn without_a_covariance_derived_stderr_stays_none() {
    let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&t| 4.0 * t).collect();
    let r = AllFixed::default().fit(&y, &x).expect("拟合");

    assert!(r.success);
    assert_eq!(r.covar, None);
    assert_eq!(r.model.twice, 4.0);
    assert!(r.params.iter().all(|p| p.stderr.is_none()));
    assert_eq!(r.params.len(), 2);
}

/// 全部字段都是派生量的模型:一个普通字段都没有,也必须能编译、能拟合。
#[derive(Model, Debug)]
struct AllDerived {
    #[param(derive)]
    only: f64,
}

impl AllDerived {
    fn only(&self) -> f64 {
        2.0
    }
}

impl Curve for AllDerived {
    fn eval(&self, _x: f64) -> f64 {
        self.only
    }
}

#[test]
fn a_model_made_only_of_derived_fields_compiles() {
    let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|_| 2.0).collect();

    let r = AllDerived::default().fit(&y, &x).expect("拟合");
    assert_eq!(r.model.only, 2.0);
    assert_eq!(r.nvarys, 0);
}

/// 无变参数的拟合同样要回填重算后的派生值,而不是构造时的缓存。
#[test]
fn a_fit_with_nothing_to_vary_reports_fresh_derived_values() {
    let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&t| 4.0 * t).collect();

    // 手写构造:派生字段带的是 Default 按默认字段算出的缓存,与被覆盖的字段不一致。
    let stale = AllFixed {
        a: 5.0,
        ..AllFixed::default()
    };
    assert_eq!(stale.twice, 4.0, "构造出来的缓存本就该是旧的");

    let r = stale.fit(&y, &x).expect("拟合");
    assert_eq!(r.model.twice, 10.0);
    assert_eq!(r.params.get("twice").expect("在表内").value, 10.0);
}

/// 派生量在收敛点非有限 → 标准误报 NaN(显式可见),而不是伪造的 0。
#[derive(Model, Debug)]
struct OutOfOrder {
    #[param(value = 2.0)]
    a: f64,
    #[param(derive)]
    early: f64,
    #[param(derive)]
    late: f64,
}

impl OutOfOrder {
    /// 引用后声明的 `late`:重算到它时还是占位值,结果恒为 NaN。
    fn early(&self) -> f64 {
        1.0 + self.late
    }

    fn late(&self) -> f64 {
        3.0 * self.a
    }
}

impl Curve for OutOfOrder {
    fn eval(&self, x: f64) -> f64 {
        self.a * x
    }
}

#[test]
fn a_non_finite_derived_value_yields_a_nan_stderr() {
    let (x, y) = data();
    let r = OutOfOrder {
        a: 2.0,
        ..OutOfOrder::default()
    }
    .fit(&y, &x)
    .expect("拟合");

    assert!(r.model.early.is_nan(), "逆序引用本就该得 NaN");
    match stderr_of(&r, "early") {
        Some(se) => assert!(se.is_nan(), "非有限的派生量必须报 NaN,实际 {se}"),
        None => panic!("协方差可得,派生槽位不该是 None"),
    }
}

/// 加权 + 复数 + 派生量三者叠加,仍要落在解析 delta 方法上。
#[test]
fn weighted_complex_fits_propagate_derived_stderr() {
    let x: Vec<f64> = (0..11).map(|i| i as f64).collect();
    let y: Vec<lmfit::Complex64> = x
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let h = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let re = ((h >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            let im = ((h >> 11) as f64 / (1u64 << 31) as f64) - 0.5;
            lmfit::Complex64::new(2.0 * t + 0.1 * re, 1.0 * t + 0.1 * im)
        })
        .collect();
    // 异方差:每三点的 σ 小一个量级。
    let sigma: Vec<f64> = (0..11)
        .map(|i| if i % 3 == 0 { 0.02 } else { 0.2 })
        .collect();

    let r = ComplexLine {
        slope: 1.0,
        offset: 1.0,
        ..ComplexLine::default()
    }
    .fit_sigma(&y, &x, &sigma)
    .expect("拟合");

    let c = r.covar.as_ref().expect("协方差可得");
    // ∂span/∂s = 1,∂span/∂o = -1。
    let p: [f64; 2] = [1.0, -1.0];
    let mut var = 0.0;
    for j in 0..2 {
        for k in 0..2 {
            var += p[j] * p[k] * c.matrix[j * c.nvarys + k];
        }
    }
    let want = var.max(0.0).sqrt();
    let got = r
        .params
        .get("span")
        .expect("参数在表内")
        .stderr
        .expect("派生的标准误已传播");
    assert!((got - want).abs() < 1e-6 * want, "{got} vs {want}");
}

/// 派生模型不得要求调用方导入 `ModelParams`:生成的代码必须走全限定路径。
/// 本模块刻意只导入 `Curve` 与 `Model`。
mod without_trait_in_scope {
    use lmfit::{Curve, Model};

    #[derive(Model, Debug)]
    struct Pick {
        #[param(value = 2.0)]
        a: f64,
        #[param(derive)]
        twice: f64,
    }

    impl Pick {
        fn twice(&self) -> f64 {
            2.0 * self.a
        }
    }

    impl Curve for Pick {
        fn eval(&self, x: f64) -> f64 {
            self.a * x
        }
    }

    #[test]
    fn a_derived_model_needs_no_trait_import() {
        let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&t| 3.0 * t).collect();
        let r = Pick::default().fit(&y, &x).expect("拟合");
        assert!((r.model.a - 3.0).abs() < 1e-9);
        assert_eq!(r.model.twice, 2.0 * r.model.a);
    }
}

/// 报表:派生参数带 `+/-` 与 `(derive)` 标记,而不是 `(fix)`。
#[test]
fn the_report_marks_derived_parameters() {
    let (x, y) = data();
    let r = Line {
        slope: 1.0,
        intercept: 1.0,
        ..Line::default()
    }
    .fit(&y, &x)
    .expect("拟合");
    let report = r.to_string();

    let line = report
        .lines()
        .find(|l| l.trim_start().starts_with("sum:"))
        .expect("派生参数有独立行");
    assert!(line.contains("+/-"), "{line}");
    assert!(line.ends_with("(derive)"), "{line}");
}

/// 三态并存:变参数 `(init = …)`、固定参数 `(fix)`、派生 `(derive)`。
#[test]
fn the_three_parameter_kinds_are_marked_distinctly() {
    let (x, y) = data();
    let r = FixedFeed {
        slope: 1.0,
        intercept: 1.0,
        known: 2.0,
        ..FixedFeed::default()
    }
    .fit(&y, &x)
    .expect("拟合");
    let report = r.to_string();

    let of = |name: &str| -> String {
        match report
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("{name}:")))
        {
            Some(line) => line.to_string(),
            None => panic!("{name} 行缺失"),
        }
    };
    assert!(of("slope").contains("(init = "), "{}", of("slope"));
    assert!(of("known").ends_with("(fix)"), "{}", of("known"));
    assert!(of("scaled").ends_with("(derive)"), "{}", of("scaled"));
}
