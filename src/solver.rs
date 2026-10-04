//! The Levenberg-Marquardt adapter.
//!
//! This is the only module in the crate that knows the `levenberg-marquardt`
//! crate exists. Everything upstream deals in models and parameters; the
//! solver deals in flat vectors of unbounded doubles. Translating between the
//! two is the whole job here, and it is deliberately isolated: adding a
//! derivative-free solver later means adding a sibling to this file, not
//! touching the traits, the derive macro, or the report.
//!
//! Three translations happen here, and each is easy to get subtly wrong:
//!
//! 1. **Parameter selection.** Only fields marked `vary` reach the solver, so
//!    solver index `j` is not model index `i`. A fixed parameter keeps its
//!    value and is invisible to the fit.
//! 2. **Bounds.** The solver has no box constraints at all, so bounded
//!    parameters are carried in [`Transform`]'s unbounded space. Bounds are
//!    therefore enforced by construction, never by rejection or clamping
//!    during iteration.
//! 3. **The Jacobian.** Required, not optional — the solver does not
//!    difference on our behalf, and returning `None` aborts the fit outright.

use std::cell::{Cell, RefCell};

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{DMatrix, DVector, Dyn, Matrix, Vector, storage::Owned};
use num_complex::Complex64;

use crate::bounds::Transform;
use crate::error::{Error, Result};
use crate::numerics::{Covariance, central_fd_step, covariance, forward_diff_jacobian};
use crate::parameter::{Parameter, Parameters};
use crate::result::{ComplexResult, ModelResult, Statistics, statistics};
use crate::traits::{ComplexCurve, Curve, ModelParams, PartialValues};

/// scipy's `leastsq` default for `ftol` and `xtol`, which lmfit inherits.
///
/// `with_tol` sets `ftol = xtol = tol` *and* `gtol = 0`, which is precisely
/// scipy's combination, so one call expresses all three. This must be set
/// explicitly: the solver's own default without the `minpack-compat` feature
/// is `eps * 30`, which is a very different stopping point.
const TOL: f64 = 1.49012e-8;

/// lmfit's budget is `max_nfev = 2000 * (nvarys + 1)` and the solver's is
/// `patience * (n + 1)`, so the two coincide here.
const PATIENCE: usize = 2000;

/// 解析雅可比组装的结果。
pub(crate) enum Analytic {
    /// 模型未提供(缺省,或中途拒供):整张雅可比退回有限差分。
    Default,
    /// 已按列主序完整写入,可直接使用。
    Available,
    /// 偏导含非有限值:与有限差分的非有限语义一致,判雅可比失败。
    NonFinite,
}

/// 解析组装器的函数指针形态:实数/复数入口各自注入。
///
/// 用注入而非双 trait 实现,是因为同时实现 `Curve` 与 `ComplexCurve` 的模型
/// 会让两个实现重叠(E0119)——组装方式在入口按数据选择,绕开一致性检查。
pub(crate) type AnalyticAssembler<M, D, X> =
    for<'p> fn(&LmProblem<'p, M, D, X>, &[f64], &mut [f64]) -> Analytic;

/// One parameter the solver is allowed to move.
struct Varying {
    /// Index into the model's parameter layout.
    index: usize,
    /// This parameter's bounds, as a map in and out of solver space.
    transform: Transform,
}

/// Residuals memoised against the solver-space point they belong to.
///
/// The solver asks for residuals and the Jacobian separately, and both are
/// anchored at the same point — so without this the base residuals would be
/// computed twice per iteration, once by each. For a three-parameter model
/// that is one wasted evaluation out of every five.
#[derive(Default)]
struct Cache {
    /// The solver-space vector `residuals` belongs to.
    at: Vec<f64>,
    residuals: Vec<f64>,
    valid: bool,
}

/// 观测数据如何把逐点残差写入求解器的实数残差缓冲。
///
/// `X` 为自变量元素类型:实数曲线为 `f64`,复数模型接受 `f64` 或 `Complex64`。
/// 实数观测每个数据点占 1 个槽位;复数观测占 2 个槽位(实部、虚部逐点交错,
/// 与 numpy `view(float)` 同序)。求解器的其余部分只认识实数残差向量。
pub(crate) trait ResidualSrc<M: ModelParams, X> {
    /// 数据点个数。
    fn npoints(&self) -> usize;

    /// 实数残差缓冲的槽位数(实数 1 倍、复数 2 倍)。
    fn nslots(&self) -> usize;

    /// 逐点测量不确定度;不提供时为 `None`(等价于全 1)。
    ///
    /// 解析雅可比组装据此决定是否乘 `1/σ`——只把加权做在残差里会让有限差分
    /// 路径正确、解析路径静默走偏。
    fn sigmas(&self) -> Option<&[f64]> {
        None
    }

    /// 将每个数据点处的残差写入 `out`。
    ///
    /// 返回:出现首个非有限残差时记录其数据点下标并返回 false —— 单个 NaN
    /// 会毒化整条法方程,继续推进没有收益。
    fn override_residuals(
        &self,
        model: &M,
        x: &[X],
        nfev: &Cell<usize>,
        nonfinite_at: &Cell<Option<usize>>,
        out: &mut [f64],
    ) -> bool;
}

/// 带不确定度的观测:逐点残差按 `1/σ` 加权。
///
/// 权重挂在数据源上而不是求解器的形参上,因为它有**两个**消费者:残差写入
/// (除以 σ)与解析雅可比组装(乘 `1/σ`)。做成独立类型而非可选参数,也让
/// 未加权路径保持零开销——两个形状单态化成两份代码,热循环里没有分支,
/// 也没有乘以 1.0 的乘法。
pub(crate) struct Measurement<'a, Y: ?Sized> {
    /// 观测值(`[f64]` 或 `[Complex64]`)。
    pub y: &'a Y,
    /// 逐点测量不确定度,长度等于数据点数。
    pub sigma: &'a [f64],
}

impl<M: Curve> ResidualSrc<M, f64> for Measurement<'_, [f64]> {
    fn npoints(&self) -> usize {
        self.y.len()
    }

    fn nslots(&self) -> usize {
        self.y.len()
    }

    fn sigmas(&self) -> Option<&[f64]> {
        Some(self.sigma)
    }

    fn override_residuals(
        &self,
        model: &M,
        x: &[f64],
        nfev: &Cell<usize>,
        nonfinite_at: &Cell<Option<usize>>,
        out: &mut [f64],
    ) -> bool {
        nfev.set(nfev.get() + 1);
        for (i, (&xi, &yi)) in x.iter().zip(self.y).enumerate() {
            let r = (yi - model.eval(xi)) / self.sigma[i];
            if !r.is_finite() {
                if nonfinite_at.get().is_none() {
                    nonfinite_at.set(Some(i));
                }
                return false;
            }
            out[i] = r;
        }
        true
    }
}

impl<M: ComplexCurve, X: Copy + Into<Complex64>> ResidualSrc<M, X> for Measurement<'_, [Complex64]> {
    fn npoints(&self) -> usize {
        self.y.len()
    }

    fn nslots(&self) -> usize {
        self.y.len() * 2
    }

    fn sigmas(&self) -> Option<&[f64]> {
        Some(self.sigma)
    }

    fn override_residuals(
        &self,
        model: &M,
        x: &[X],
        nfev: &Cell<usize>,
        nonfinite_at: &Cell<Option<usize>>,
        out: &mut [f64],
    ) -> bool {
        nfev.set(nfev.get() + 1);
        for (i, (&xi, &yi)) in x.iter().zip(self.y).enumerate() {
            let r = (yi - model.eval(xi.into())) / self.sigma[i];
            if !r.re.is_finite() || !r.im.is_finite() {
                if nonfinite_at.get().is_none() {
                    nonfinite_at.set(Some(i));
                }
                return false;
            }
            out[2 * i] = r.re;
            out[2 * i + 1] = r.im;
        }
        true
    }
}

impl<'a, M: Curve> ResidualSrc<M, f64> for &'a [f64] {
    fn npoints(&self) -> usize {
        self.len()
    }

    fn nslots(&self) -> usize {
        self.len()
    }

    fn override_residuals(
        &self,
        model: &M,
        x: &[f64],
        nfev: &Cell<usize>,
        nonfinite_at: &Cell<Option<usize>>,
        out: &mut [f64],
    ) -> bool {
        nfev.set(nfev.get() + 1);
        for (i, (&xi, &yi)) in x.iter().zip(*self).enumerate() {
            let r = yi - model.eval(xi);
            if !r.is_finite() {
                if nonfinite_at.get().is_none() {
                    nonfinite_at.set(Some(i));
                }
                return false;
            }
            out[i] = r;
        }
        true
    }
}

impl<'a, M: ComplexCurve, X: Copy + Into<Complex64>> ResidualSrc<M, X> for &'a [Complex64] {
    fn npoints(&self) -> usize {
        self.len()
    }

    fn nslots(&self) -> usize {
        self.len() * 2
    }

    fn override_residuals(
        &self,
        model: &M,
        x: &[X],
        nfev: &Cell<usize>,
        nonfinite_at: &Cell<Option<usize>>,
        out: &mut [f64],
    ) -> bool {
        nfev.set(nfev.get() + 1);
        for (i, (&xi, &yi)) in x.iter().zip(*self).enumerate() {
            let r = yi - model.eval(xi.into());
            if !r.re.is_finite() || !r.im.is_finite() {
                if nonfinite_at.get().is_none() {
                    nonfinite_at.set(Some(i));
                }
                return false;
            }
            out[2 * i] = r.re;
            out[2 * i + 1] = r.im;
        }
        true
    }
}

/// A model, its data, and its bounds, presented to the solver as a plain
/// least-squares problem in unbounded space.
pub(crate) struct LmProblem<'a, M, D, X> {
    model: M,
    x: &'a [X],
    data: D,
    varying: Vec<Varying>,
    cache: RefCell<Cache>,
    /// 解析雅可比组装器;由实数/复数入口在构造时注入,None 表示恒走有限差分。
    analytic: Option<AnalyticAssembler<M, D, X>>,
    /// Residual evaluations performed, finite-difference probes included.
    nfev: Cell<usize>,
    /// First data point whose residual came out non-finite, if any. Preserved
    /// so that an aborted fit can say *where* it went wrong rather than
    /// reporting the solver's opaque `User("residuals")`.
    nonfinite_at: Cell<Option<usize>>,
}

impl<'a, M: ModelParams, D: ResidualSrc<M, X>, X> LmProblem<'a, M, D, X> {
    fn new(
        model: M,
        x: &'a [X],
        data: D,
        varying: Vec<Varying>,
        analytic: Option<AnalyticAssembler<M, D, X>>,
    ) -> Self {
        Self {
            model,
            x,
            data,
            varying,
            cache: RefCell::new(Cache::default()),
            analytic,
            nfev: Cell::new(0),
            nonfinite_at: Cell::new(None),
        }
    }

    /// The solver-space vector corresponding to the model's current values.
    fn internal_params(&self) -> Vec<f64> {
        self.varying
            .iter()
            .map(|v| v.transform.to_internal(self.model.get(v.index)))
            .collect()
    }

    /// 尝试解析组装雅可比。
    ///
    /// * `at` —— 内部空间点(模型的当前参数);`out` —— 列主序雅可比缓冲。
    ///
    /// 返回:组装结果;未注入组装器时为 `NotProvided`。
    fn analytic_jacob(&self, at: &[f64], out: &mut [f64]) -> Analytic {
        match self.analytic {
            Some(assemble) => assemble(self, at, out),
            None => Analytic::Default,
        }
    }

    /// Residuals at the model's current values, recomputing only if the cache
    /// does not already hold them for `at`.
    fn residuals_cache(&self, at: &[f64]) -> Option<Vec<f64>> {
        {
            let cache = self.cache.borrow();
            if cache.valid && cache.at == at {
                return Some(cache.residuals.clone());
            }
        }

        let mut out = vec![0.0; self.data.nslots()];
        if !self
            .data
            .override_residuals(&self.model, self.x, &self.nfev, &self.nonfinite_at, &mut out)
        {
            return None;
        }

        let mut cache = self.cache.borrow_mut();
        cache.at = at.to_vec();
        cache.residuals = out.clone();
        cache.valid = true;
        Some(out)
    }
}

impl<M: ModelParams, D: ResidualSrc<M, X>, X> LeastSquaresProblem<f64, Dyn, Dyn>
    for LmProblem<'_, M, D, X>
{
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, p: &Vector<f64, Dyn, Self::ParameterStorage>) {
        for (j, v) in self.varying.iter().enumerate() {
            // `from_internal` cannot leave the bounds, so no clamping is
            // needed and none is applied.
            self.model.set(v.index, v.transform.from_internal(p[j]));
        }
        self.cache.borrow_mut().valid = false;
    }

    fn params(&self) -> Vector<f64, Dyn, Self::ParameterStorage> {
        DVector::from_vec(self.internal_params())
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        let at = self.internal_params();
        self.residuals_cache(&at).map(DVector::from_vec)
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, Dyn, Self::JacobianStorage>> {
        let at = self.internal_params();
        let nrows = self.data.nslots();
        let ncols = self.varying.len();
        let mut out = vec![0.0; nrows * ncols];

        // 解析优先:模型提供并可组装时,不产生任何残差求值。
        match self.analytic_jacob(&at, &mut out) {
            Analytic::Available => return Some(DMatrix::from_vec(nrows, ncols, out)),
            Analytic::NonFinite => return None,
            Analytic::Default => {}
        }

        let base = self.residuals_cache(&at)?;
        debug_assert_eq!(base.len(), nrows);

        // The model's full parameter vector, varying entries overwritten per
        // probe. Writing every varying entry from `probe` on each call also
        // restores the ones the previous probe disturbed — `probe` is `at`
        // with exactly one entry displaced.
        let mut full: Vec<f64> = (0..M::NPARAMS).map(|i| self.model.get(i)).collect();

        // A perturbed model is *built*, not mutated: `at_values` takes
        // `&self`, which is what makes this reachable from a `&self` method
        // without interior mutability around the model itself.
        let ok = forward_diff_jacobian(
            &at,
            &base,
            |probe, buf| {
                for (j, v) in self.varying.iter().enumerate() {
                    full[v.index] = v.transform.from_internal(probe[j]);
                }
                let perturbed = self.model.at_values(&full);
                self.data.override_residuals(
                    &perturbed,
                    self.x,
                    &self.nfev,
                    &self.nonfinite_at,
                    buf,
                )
            },
            &mut out,
        );

        if ok {
            Some(DMatrix::from_vec(nrows, ncols, out))
        } else {
            None
        }
    }
}

/// 实数解析组装:内部空间列 j、数据点 i 处为 `−∂f/∂θⱼ · scale_gradient(bⱼ)`。
///
/// * `this` —— 问题(模型已在当前参数处);`at` —— 内部空间点;
///   `out` —— 列主序雅可比缓冲(m×n)。
///
/// 返回:组装结果;任一点拒供即 `NotProvided`,任一偏导非有限即 `Failed`。
fn assemble_analytic_real<M: Curve, D: ResidualSrc<M, f64>>(
    this: &LmProblem<'_, M, D, f64>,
    at: &[f64],
    out: &mut [f64],
) -> Analytic {
    let m = this.data.npoints();
    // 各列的换算因子对整张雅可比不变:循环外一次算出。
    let scale: Vec<f64> = this
        .varying
        .iter()
        .enumerate()
        .map(|(j, v)| v.transform.scale_gradient(at[j]))
        .collect();
    // 加权拟合的雅可比是**加权残差**的雅可比:列上要多乘 1/σ。漏掉它的后果是
    // 有限差分路径正确、解析路径静默走偏——两条路各自自洽,极难发现。
    let sigmas = this.data.sigmas();
    for i in 0..m {
        let weight = match sigmas {
            Some(s) => 1.0 / s[i],
            None => 1.0,
        };
        let p = match this.model.partials_at(this.x[i]) {
            Some(p) => p,
            None => return Analytic::Default,
        };
        debug_assert_eq!(p.len(), M::NPARAMS, "PartialValues::len 与 NPARAMS 不符");
        for (j, v) in this.varying.iter().enumerate() {
            let d = -p.get(v.index) * scale[j] * weight;
            if !d.is_finite() {
                return Analytic::NonFinite;
            }
            out[j * m + i] = d;
        }
    }
    Analytic::Available
}

/// 复数解析组装:每个数据点的复数偏导按实部、虚部交错写入两行
/// (与 [`ResidualSrc`] 的 `[re, im]` 槽位同序)。
///
/// * `this` —— 问题(模型已在当前参数处);`at` —— 内部空间点;
///   `out` —— 列主序雅可比缓冲(2m×n)。
///
/// 返回:组装结果;任一点拒供即 `NotProvided`,任一偏导非有限即 `Failed`。
fn assemble_analytic_complex<M: ComplexCurve, D: ResidualSrc<M, X>, X: Copy + Into<Complex64>>(
    this: &LmProblem<'_, M, D, X>,
    at: &[f64],
    out: &mut [f64],
) -> Analytic {
    let m = this.data.npoints();
    let rows = this.data.nslots();
    // 各列的换算因子对整张雅可比不变:循环外一次算出。
    let scale: Vec<f64> = this
        .varying
        .iter()
        .enumerate()
        .map(|(j, v)| v.transform.scale_gradient(at[j]))
        .collect();
    // 同实数路径:每个复频点一个 σ,实部与虚部同权。
    let sigmas = this.data.sigmas();
    for i in 0..m {
        let weight = match sigmas {
            Some(s) => 1.0 / s[i],
            None => 1.0,
        };
        let p = match this.model.partials_at(this.x[i].into()) {
            Some(p) => p,
            None => return Analytic::Default,
        };
        debug_assert_eq!(p.len(), M::NPARAMS, "PartialValues::len 与 NPARAMS 不符");
        for (j, v) in this.varying.iter().enumerate() {
            let d = p.get(v.index) * (-scale[j]) * weight;
            if !d.re.is_finite() || !d.im.is_finite() {
                return Analytic::NonFinite;
            }
            out[j * rows + 2 * i] = d.re;
            out[j * rows + 2 * i + 1] = d.im;
        }
    }
    Analytic::Available
}

/// 拟合过程中收敛出的共享结果(实数域),供实数/复数两条装配壳使用。
pub(crate) struct Solution<M> {
    model: M,
    params: Parameters,
    values: Vec<f64>,
    nfev: usize,
    success: bool,
    message: String,
    jac_for_cov: Option<(Vec<f64>, Vec<f64>)>,
}

/// 运行一次拟合的共享骨架:校验、装配、求解、读回与最终雅可比。
///
/// * `model` —— 起点模型;`data` —— 观测(实数或复数);`x` —— 自变量。
///
/// 返回:收敛点的共享结果;长度不符或数据为空时报错。
pub(crate) fn fit_core<M: ModelParams, D, X>(
    model: &M,
    data: D,
    x: &[X],
    analytic: Option<AnalyticAssembler<M, D, X>>,
) -> Result<Solution<M>>
where
    D: ResidualSrc<M, X>,
{
    if x.len() != data.npoints() {
        return Err(Error::DimensionMismatch {
            x: x.len(),
            y: data.npoints(),
        });
    }

    let params = model.parameters()?;
    // The one place `specs()` and `NPARAMS` meet on every fit, so a
    // hand-written impl that disagrees with itself is caught here in debug
    // builds rather than corrupting a fit.
    debug_assert_eq!(params.len(), M::NPARAMS, "specs() disagrees with NPARAMS");
    let nvarys = params.no_fix_indices().len();
    if data.npoints() == 0 {
        return Err(Error::TooFewDataPoints { ndata: 0, nvarys });
    }

    let varying: Vec<Varying> = params
        .iter()
        .enumerate()
        .filter(|(_, p)| p.vary)
        .map(|(index, p)| {
            Ok(Varying {
                index,
                transform: p.transform()?,
            })
        })
        .collect::<Result<_>>()?;

    let default_values: Vec<f64> = params.values();
    let working = model.at_values(&default_values);

    // Nothing to vary: the model is already at its answer, and handing the
    // solver an empty parameter vector would only make it report
    // `NoParameters`.
    if varying.is_empty() {
        // 同主路径:派生字段取自重算后的模型,而不是构造时的缓存。
        let values: Vec<f64> = (0..M::NPARAMS).map(|i| working.get(i)).collect();
        return Ok(Solution {
            model: working,
            params,
            values,
            nfev: 0,
            success: true,
            message: "Fit succeeded: there was nothing to vary.".to_string(),
            jac_for_cov: None,
        });
    }

    let problem = LmProblem::new(working, x, data, varying, analytic);

    let (problem, report) = LevenbergMarquardt::<f64>::new()
        .with_tol(TOL)
        .with_patience(PATIENCE)
        .minimize(problem);

    let (success, message) = describe(&report.termination, problem.nonfinite_at.get());

    // Read the fitted model back out. `at_values` again, so the trait needs
    // no `Clone` bound.
    let fit_result: Vec<f64> = (0..M::NPARAMS).map(|i| problem.model.get(i)).collect();
    let fit_model = model.at_values(&fit_result);
    // 派生字段由 `at_values` 重算,这里取回权威值;普通字段逐位不变。
    let values: Vec<f64> = (0..M::NPARAMS).map(|i| fit_model.get(i)).collect();

    // 最终雅可比:在收敛点差分,供协方差使用。探测次数计入 nfev。
    let at: Vec<f64> = problem.internal_params();
    let base = match problem.residuals_cache(&at) {
        Some(r) => r,
        None => Vec::new(),
    };
    let ncols = problem.varying.len();
    let mut jac = vec![0.0; base.len() * ncols];
    let jac_ok = base.len() > 0
        && ncols > 0
        && match problem.analytic_jacob(&at, &mut jac) {
            // 解析优先:模型提供时同样不产生探测求值。
            Analytic::Available => true,
            Analytic::NonFinite => false,
            Analytic::Default => {
                let mut full: Vec<f64> = (0..M::NPARAMS).map(|i| problem.model.get(i)).collect();
                forward_diff_jacobian(
                    &at,
                    &base,
                    |probe, buf| {
                        for (j, v) in problem.varying.iter().enumerate() {
                            full[v.index] = v.transform.from_internal(probe[j]);
                        }
                        let perturbed = problem.model.at_values(&full);
                        problem.data.override_residuals(
                            &perturbed,
                            problem.x,
                            &problem.nfev,
                            &problem.nonfinite_at,
                            buf,
                        )
                    },
                    &mut jac,
                )
            }
        };
    let gradients: Vec<f64> = problem
        .varying
        .iter()
        .enumerate()
        .map(|(j, v)| v.transform.scale_gradient(at[j]))
        .collect();
    let jac_for_cov = match jac_ok {
        true => Some((jac, gradients)),
        false => None,
    };

    Ok(Solution {
        model: fit_model,
        params,
        values,
        nfev: problem.nfev.get(),
        success,
        message,
        jac_for_cov,
    })
}

/// Run a fit.
///
/// Split out of [`Curve::fit`] so the trait stays a thin facade.
pub(crate) fn fit<M: Curve>(model: &M, y: &[f64], x: &[f64]) -> Result<ModelResult<M>> {
    let solved = fit_core(model, y, x, Some(assemble_analytic_real::<M, &[f64]>))?;
    Ok(assemble_real(solved, y, x, None))
}

/// Run a complex fit.
///
/// [`ComplexCurve::fit`] 的薄壳:与实数 [`fit`] 共享 `fit_core` 骨架。
pub(crate) fn fit_complex<M: ComplexCurve, X: Copy + Into<Complex64>>(
    model: &M,
    y: &[Complex64],
    x: &[X],
) -> Result<ComplexResult<M>> {
    let solved = fit_core(model, y, x, Some(assemble_analytic_complex::<M, &[Complex64], X>))?;
    Ok(assemble_complex(solved, y, x, None))
}

/// 校验逐点 σ:长度与数据点一致,且每项有限为正。
///
/// * `npoints` —— 数据点个数;`sigma` —— 待校验的逐点不确定度。
///
/// 返回:全部合法时为 `Ok`;否则为对应的 [`Error`]。
fn check_sigma(npoints: usize, sigma: &[f64]) -> Result<()> {
    if sigma.len() != npoints {
        return Err(Error::SigmaMismatch {
            npoints,
            sigma: sigma.len(),
        });
    }
    for (index, &value) in sigma.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::InvalidSigma { index, value });
        }
    }
    Ok(())
}

/// 以逐点不确定度加权运行一次实数拟合。
///
/// * `model` —— 起点模型;`y` —— 观测;`x` —— 自变量;`sigma` —— 逐点不确定度。
///
/// 返回:拟合结果;σ 非法时报错。
pub(crate) fn fit_sigma<M: Curve>(
    model: &M,
    y: &[f64],
    x: &[f64],
    sigma: &[f64],
) -> Result<ModelResult<M>> {
    check_sigma(y.len(), sigma)?;
    let source = Measurement { y, sigma };
    let solved = fit_core(model, source, x, Some(assemble_analytic_real::<M, _>))?;
    Ok(assemble_real(solved, y, x, Some(sigma)))
}

/// 以逐点不确定度加权运行一次复数拟合。
///
/// 每个复频点一个 σ,实部与虚部同权。
///
/// * `model` —— 起点模型;`y` —— 复观测;`x` —— 自变量;`sigma` —— 逐点不确定度。
///
/// 返回:拟合结果;σ 非法时报错。
pub(crate) fn fit_complex_sigma<M: ComplexCurve, X: Copy + Into<Complex64>>(
    model: &M,
    y: &[Complex64],
    x: &[X],
    sigma: &[f64],
) -> Result<ComplexResult<M>> {
    check_sigma(y.len(), sigma)?;
    let source = Measurement { y, sigma };
    let solved = fit_core(model, source, x, Some(assemble_analytic_complex::<M, _, X>))?;
    Ok(assemble_complex(solved, y, x, Some(sigma)))
}

/// 由共享结果装配实数拟合结果:重建曲线与残差,补齐统计、协方差与标准误。
fn assemble_real<M: Curve>(
    solved: Solution<M>,
    y: &[f64],
    x: &[f64],
    sigma: Option<&[f64]>,
) -> ModelResult<M> {
    let Solution {
        model,
        mut params,
        values,
        nfev,
        success,
        message,
        jac_for_cov,
    } = solved;

    for (p, v) in params.iter_mut().zip(&values) {
        p.value = *v;
    }

    let best_fit: Vec<f64> = x.iter().map(|&xi| model.eval(xi)).collect();
    let residual: Vec<f64> = y
        .iter()
        .zip(&best_fit)
        .map(|(observed, predicted)| observed - predicted)
        .collect();

    let nvarys = params.no_fix_indices().len();
    // 统计按加权残差算;`residual` 字段本身保持未加权(与 `best_fit` 同口径)。
    let (stats, covar, stderr) = match sigma {
        Some(s) => {
            let scaled: Vec<f64> = residual.iter().zip(s).map(|(r, si)| r / si).collect();
            finish(&model, &params, nvarys, &scaled, jac_for_cov)
        }
        None => finish(&model, &params, nvarys, &residual, jac_for_cov),
    };

    ModelResult {
        model,
        params,
        x: x.to_vec(),
        y: y.to_vec(),
        best_fit,
        residual,
        stderr,
        covar,
        chisqr: stats.chisqr,
        redchi: stats.redchi,
        aic: stats.aic,
        bic: stats.bic,
        ndata: stats.ndata,
        nvarys,
        nfree: stats.nfree,
        nfev,
        success,
        message,
    }
}

/// 由共享结果装配复数拟合结果:重建复曲线与复残差,交错为实数残差后
/// 复用统计、协方差与标准误计算。
fn assemble_complex<M: ComplexCurve, X: Copy + Into<Complex64>>(
    solved: Solution<M>,
    y: &[Complex64],
    x: &[X],
    sigma: Option<&[f64]>,
) -> ComplexResult<M> {
    let Solution {
        model,
        mut params,
        values,
        nfev,
        success,
        message,
        jac_for_cov,
    } = solved;

    for (p, v) in params.iter_mut().zip(&values) {
        p.value = *v;
    }

    let best_fit: Vec<Complex64> = x.iter().map(|&xi| model.eval(xi.into())).collect();
    let residual: Vec<Complex64> = y
        .iter()
        .zip(&best_fit)
        .map(|(observed, predicted)| observed - predicted)
        .collect();

    // 逐点交错为实数残差 [re, im](与 numpy view(float) 同序);NaN 自然传播。
    let mut real_residual = Vec::with_capacity(residual.len() * 2);
    match sigma {
        // 每个复频点一个 σ,实部与虚部同权。
        Some(s) => {
            for (r, si) in residual.iter().zip(s) {
                real_residual.push(r.re / si);
                real_residual.push(r.im / si);
            }
        }
        None => {
            for r in &residual {
                real_residual.push(r.re);
                real_residual.push(r.im);
            }
        }
    }

    let nvarys = params.no_fix_indices().len();
    let (stats, covar, stderr) = finish(&model, &params, nvarys, &real_residual, jac_for_cov);

    ComplexResult {
        model,
        params,
        x: x.iter().map(|&v| v.into()).collect(),
        y: y.to_vec(),
        best_fit,
        residual,
        stderr,
        covar,
        chisqr: stats.chisqr,
        redchi: stats.redchi,
        aic: stats.aic,
        bic: stats.bic,
        ndata: stats.ndata,
        nvarys,
        nfree: stats.nfree,
        nfev,
        success,
        message,
    }
}

/// 派生参数的标准误,按 delta 方法传播。
///
/// 传播式为 `σ_g² = pᵀCp`:`C` 是外部空间协方差(变参数序,与
/// [`Parameters::no_fix_indices`] 同序),`p` 是派生量对各变参数的一阶偏导。
/// 偏导取中心差商,每侧先经 [`Transform::to_internal`](先夹入界内)再
/// [`Transform::from_internal`] 折回,故参数贴界时自动退化为单侧差商,不会
/// 产生越界点;往返把两侧塌缩成同一点时该项贡献 0(参数在外部空间里动不了)。
///
/// * `model` —— 收敛点的模型,派生字段已由 `at_values` 重算;
/// * `params` —— 已回填拟合值的参数表;`covar` —— 变参数协方差。
///
/// 返回:派生参数的 `(参数下标, 标准误)` 表;非派生参数不在其中。
fn derive_stderr<M: ModelParams>(
    model: &M,
    params: &Parameters,
    covar: &Covariance,
) -> Vec<(usize, f64)> {
    // 变参数的序即协方差矩阵的序,与 `no_fix_indices` 一致。
    let varying: Vec<(usize, &Parameter)> = params
        .iter()
        .enumerate()
        .filter(|entry| entry.1.vary)
        .collect();
    let base = params.values();
    let mut out = Vec::new();

    for (d, p) in params.iter().enumerate() {
        if !p.derive {
            continue;
        }
        let mut grad = vec![0.0; varying.len()];
        for (j, &(v, vp)) in varying.iter().enumerate() {
            let transform = match vp.transform() {
                Ok(t) => t,
                Err(err) => {
                    // fit_core 已对同一批参数校验过变换,此处不可达;真发生时
                    // 不写任何派生标准误,而不是写下可疑的数。
                    debug_assert!(false, "变参数变换在传播时失败: {err:?}");
                    return Vec::new();
                }
            };
            let theta = base[v];
            let h = central_fd_step(theta);
            let up = transform.from_internal(transform.to_internal(theta + h));
            let down = transform.from_internal(transform.to_internal(theta - h));
            let dtheta = up - down;
            grad[j] = match dtheta == 0.0 {
                true => 0.0,
                false => {
                    let mut values = base.clone();
                    values[v] = up;
                    let g_up = model.at_values(&values).get(d);
                    values[v] = down;
                    let g_down = model.at_values(&values).get(d);
                    (g_up - g_down) / dtheta
                }
            };
        }
        let mut var = 0.0;
        for (j, gj) in grad.iter().enumerate() {
            for (k, gk) in grad.iter().enumerate() {
                var += gj * gk * covar.matrix[j * covar.nvarys + k];
            }
        }
        // 负值只可能来自舍入(精确算术下二次型半正定),夹到 0;NaN 原样放行,
        // 让"派生量在收敛点非有限"显式可见,而不是伪造一个精确的 0。
        let var = match var < 0.0 {
            true => 0.0,
            false => var,
        };
        out.push((d, var.sqrt()));
    }
    out
}

/// 统计量、协方差与标准误的共享计算:实数与复数装配壳共用。
///
/// * `model` —— 收敛点的模型(派生标准误的传播要重新构造它);
///   `params` —— 已回填拟合值的参数表;`nvarys` —— 变参数个数。
/// * `real_residual` —— 实数残差(复数按实部、虚部交错)。
/// * `jac_for_cov` —— 最终雅可比与梯度因子,不可得时为 None。
///
/// 返回:统计量、协方差与逐参数标准误(固定参数与无协方差时为 None)。
fn finish<M: ModelParams>(
    model: &M,
    params: &Parameters,
    nvarys: usize,
    real_residual: &[f64],
    jac_for_cov: Option<(Vec<f64>, Vec<f64>)>,
) -> (Statistics, Option<Covariance>, Vec<Option<f64>>) {
    let stats = statistics(real_residual, nvarys);

    // 协方差与标准误:雅可比不可得或不可信时全部为 None。
    let covar = match &jac_for_cov {
        Some((jac, gradients)) => covariance(jac, stats.redchi, gradients),
        None => None,
    };
    let mut stderr: Vec<Option<f64>> = vec![None; params.len()];
    match &covar {
        Some(c) => {
            let se = c.stderr();
            let mut k = 0;
            for (i, p) in params.iter().enumerate() {
                if p.vary {
                    stderr[i] = Some(se[k]);
                    k += 1;
                }
            }
            // 派生量的标准误不是自由度的函数,而是变参数协方差的函数。
            for (i, se) in derive_stderr(model, params, c) {
                stderr[i] = Some(se);
            }
        }
        None => {}
    }
    (stats, covar, stderr)
}

/// Translate the solver's termination reason into a success flag and a message.
///
/// Every variant is handled explicitly. Collapsing them to "converged or not"
/// would throw away the one thing a user needs when a fit goes wrong: whether
/// it ran out of budget, hit a NaN, or genuinely could not improve.
fn describe(reason: &TerminationReason, nonfinite_at: Option<usize>) -> (bool, String) {
    match reason {
        TerminationReason::Converged { .. } => (true, "Fit succeeded.".to_string()),
        TerminationReason::ResidualsZero => (
            true,
            "Fit succeeded: the residuals are exactly zero.".to_string(),
        ),
        TerminationReason::Orthogonal => (
            true,
            "Fit succeeded: the residual is orthogonal to the Jacobian.".to_string(),
        ),
        TerminationReason::LostPatience => (
            false,
            format!(
                "Fit did not converge: the function evaluation budget of {} was exhausted.",
                PATIENCE
            ),
        ),
        TerminationReason::NoImprovementPossible(what) => (
            false,
            format!("Fit stopped: `{what}` cannot improve further at this precision."),
        ),
        TerminationReason::Numerical(what) => (
            false,
            format!("Fit stopped: a non-finite value appeared in `{what}`."),
        ),
        TerminationReason::NoResiduals => (
            false,
            "Fit stopped: the model produced no residuals.".to_string(),
        ),
        TerminationReason::NoParameters => (
            false,
            "Fit stopped: there were no parameters to vary.".to_string(),
        ),
        TerminationReason::WrongDimensions(what) => (
            false,
            format!("Fit stopped: `{what}` had an unexpected shape."),
        ),
        // The solver reports the same opaque reason for a failed residual and
        // a failed Jacobian. We always supply a Jacobian, so a failure here is
        // a non-finite residual, and `nonfinite_at` says which point.
        TerminationReason::User(what) => match nonfinite_at {
            Some(i) => (
                false,
                format!("Fit stopped: the model produced a non-finite residual at data point {i}."),
            ),
            None => (
                false,
                format!("Fit stopped: the `{what}` evaluation failed."),
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_termination_reason_is_described() {
        let reasons = [
            TerminationReason::Converged {
                ftol: true,
                xtol: false,
            },
            TerminationReason::ResidualsZero,
            TerminationReason::Orthogonal,
            TerminationReason::LostPatience,
            TerminationReason::NoImprovementPossible("xtol"),
            TerminationReason::Numerical("residuals"),
            TerminationReason::NoResiduals,
            TerminationReason::NoParameters,
            TerminationReason::WrongDimensions("jacobian"),
            TerminationReason::User("jacobian"),
        ];
        for reason in reasons {
            let (success, message) = describe(&reason, None);
            assert!(!message.is_empty(), "{reason:?} produced no message");
            // Only the three convergence criteria count as success.
            let expected = matches!(
                reason,
                TerminationReason::Converged { .. }
                    | TerminationReason::ResidualsZero
                    | TerminationReason::Orthogonal
            );
            assert_eq!(
                success, expected,
                "{reason:?} mapped to success = {success}"
            );
        }
    }

    /// A failed residual is reported by data point rather than with the
    /// solver's opaque `User("residuals")`.
    #[test]
    fn a_failed_residual_names_the_offending_point() {
        let (success, message) = describe(&TerminationReason::User("residuals"), Some(17));
        assert!(!success);
        assert!(message.contains("17"), "message was {message:?}");
    }
}
