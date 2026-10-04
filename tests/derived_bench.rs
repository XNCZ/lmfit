//! 派生参数的代价基准:`nfev` 逐位不变;热路径的额外开销只有 `set` 里的表达式
//! 重算(实测约 1%),一次性传播随派生个数线性且量级更低。
//!
//!     cargo test --release --test derived_bench -- --ignored --nocapture

use std::time::Instant;

use lmfit::{Curve, Model};

/// 数据点数与重复次数:与上版权重基准同规模。
const NPOINTS: usize = 3000;
const REPS: usize = 200;

/// 不带派生字段的对照:四个拟合参数,起点刻意偏离真值。
#[derive(Model, Debug)]
struct Plain {
    #[param(value = 50_000.0)]
    ql: f64,
    #[param(value = 25_000.0)]
    qc: f64,
    #[param(value = 0.05)]
    cos_theta: f64,
    #[param(value = 5.0e9)]
    fr: f64,
}

impl Curve for Plain {
    fn eval(&self, x: f64) -> f64 {
        let d = (x - self.fr) / self.fr;
        self.cos_theta * (self.ql / self.qc) / (1.0 + (2.0 * self.ql * d).powi(2))
    }
}

/// 两个派生量:实际口径。
#[derive(Model, Debug)]
struct Two {
    #[param(value = 50_000.0)]
    ql: f64,
    #[param(value = 25_000.0)]
    qc: f64,
    #[param(value = 0.05)]
    cos_theta: f64,
    #[param(value = 5.0e9)]
    fr: f64,
    #[param(derive)]
    qi: f64,
    #[param(derive)]
    kappa_ex: f64,
}

impl Two {
    fn qi(&self) -> f64 {
        1.0 / (1.0 / self.ql - self.cos_theta / self.qc)
    }

    fn kappa_ex(&self) -> f64 {
        self.fr / self.qc
    }
}

impl Curve for Two {
    fn eval(&self, x: f64) -> f64 {
        let d = (x - self.fr) / self.fr;
        self.cos_theta * (self.ql / self.qc) / (1.0 + (2.0 * self.ql * d).powi(2))
    }
}

/// 八个派生量,后六个引用先两个:顺带覆盖链式。
#[derive(Model, Debug)]
struct Eight {
    #[param(value = 50_000.0)]
    ql: f64,
    #[param(value = 25_000.0)]
    qc: f64,
    #[param(value = 0.05)]
    cos_theta: f64,
    #[param(value = 5.0e9)]
    fr: f64,
    #[param(derive)]
    qi: f64,
    #[param(derive)]
    kappa_ex: f64,
    #[param(derive)]
    qi_double: f64,
    #[param(derive)]
    kappa_double: f64,
    #[param(derive)]
    qi_ratio: f64,
    #[param(derive)]
    kappa_ratio: f64,
    #[param(derive)]
    qi_plus: f64,
    #[param(derive)]
    kappa_plus: f64,
}

impl Eight {
    fn qi(&self) -> f64 {
        1.0 / (1.0 / self.ql - self.cos_theta / self.qc)
    }

    fn kappa_ex(&self) -> f64 {
        self.fr / self.qc
    }

    fn qi_double(&self) -> f64 {
        2.0 * self.qi
    }

    fn kappa_double(&self) -> f64 {
        2.0 * self.kappa_ex
    }

    fn qi_ratio(&self) -> f64 {
        self.qi / self.kappa_ex
    }

    fn kappa_ratio(&self) -> f64 {
        self.kappa_ex / self.qi
    }

    fn qi_plus(&self) -> f64 {
        self.qi + self.ql
    }

    fn kappa_plus(&self) -> f64 {
        self.kappa_ex + self.qc
    }
}

impl Curve for Eight {
    fn eval(&self, x: f64) -> f64 {
        let d = (x - self.fr) / self.fr;
        self.cos_theta * (self.ql / self.qc) / (1.0 + (2.0 * self.ql * d).powi(2))
    }
}

/// 确定性伪噪声样本:真值 `ql=60000, qc=30000, cos_theta=0.1, fr=5.0e9`。
fn data() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..NPOINTS)
        .map(|i| 4.9e9 + 2.0e5 * i as f64 / NPOINTS as f64)
        .collect();
    let truth = Plain {
        ql: 60_000.0,
        qc: 30_000.0,
        cos_theta: 0.1,
        fr: 5.0e9,
    };
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let h = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let n = ((h >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            truth.eval(t) + 1.0e-4 * n
        })
        .collect();
    (x, y)
}

/// 跑 `REPS` 次拟合并返回 `(中位耗时 µs, nfev)`。
fn median_us<M: Curve>(model: &M, y: &[f64], x: &[f64]) -> (f64, usize) {
    let mut times = Vec::with_capacity(REPS);
    let mut nfev = 0;
    for _ in 0..REPS {
        let started = Instant::now();
        let r = model.fit(y, x).expect("拟合");
        times.push(started.elapsed().as_secs_f64() * 1.0e6);
        nfev = r.nfev;
    }
    times.sort_by(|a, b| a.partial_cmp(b).expect("耗时有限"));
    (times[REPS / 2], nfev)
}

#[test]
#[ignore = "基准:用 --release --ignored --nocapture 手动跑"]
fn derived_parameters_cost_nothing_in_the_fit_loop() {
    let (x, y) = data();

    // 预热:第一批计时会连带付掉缓存与主频爬坡的代价,足以盖过本基准要测的
    // 量级,故先空跑一批,再计时。
    let _ = median_us(&Plain::default(), &y, &x);

    let (t0, n0) = median_us(&Plain::default(), &y, &x);
    let (t2, n2) = median_us(&Two::default(), &y, &x);
    let (t8, n8) = median_us(&Eight::default(), &y, &x);

    // 结构不变量:派生量不在求解器路径上,求值次数逐位相同。
    assert_eq!(n0, n2);
    assert_eq!(n2, n8);

    println!("0 派生:{t0:.1} µs/次(nfev = {n0})");
    println!(
        "2 派生:{t2:.1} µs/次,相对 0 派生 {:+.2}%",
        100.0 * (t2 - t0) / t0
    );
    println!(
        "8 派生:{t8:.1} µs/次,相对 2 派生 {:+.2}%(多出的 6 个派生的一次性传播)",
        100.0 * (t8 - t2) / t2
    );
}
