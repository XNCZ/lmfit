//! Numerical layer: differentiation and covariance.
//!
//! The solver needs a Jacobian, but a model is a user's closure — there are no
//! analytic derivatives to be had. So they are differenced, exactly as
//! MINPACK's `fdjac2` does internally for scipy, and therefore exactly as
//! lmfit's `leastsq` obtains them.
//!
//! The differential part is plain array arithmetic that knows nothing about
//! models, parameters, or the solver: it takes a vector in, calls a closure,
//! and writes a matrix out. The covariance part consumes that matrix and a
//! reduced chi-square and returns parameter uncertainties — likewise ignorant
//! of any solver.

use nalgebra::DMatrix;

/// The forward-difference step to use for a parameter currently at `value`.
///
/// scipy `leastsq` 将 `epsfcn` 缺省为机器精度,MINPACK `fdjac2` 取其平方根作相对步长。
/// 此处直接以 `f64::EPSILON.sqrt()`(1.49012e-8)为步长基数,与 scipy/lmfit 路径对齐。
///
/// Mirrors MINPACK's `fdjac2`: the step scales with the parameter's magnitude,
/// so a parameter of order `1e6` is not perturbed below the resolution of its
/// own floating-point representation. A parameter at exactly zero has no scale
/// to borrow, so the bare `sqrt(f64::EPSILON)` is used.
pub fn fd_step(value: f64) -> f64 {
    let eps = f64::EPSILON.sqrt();
    let h = eps * value.abs();
    if h == 0.0 { eps } else { h }
}

/// Fill `out` with a forward-difference Jacobian of the residual vector.
///
/// `params` is the point to differentiate at and `base` the residuals already
/// evaluated there — the caller has almost always computed them, and reusing
/// them is the difference between `n` and `n + 1` evaluations per Jacobian.
///
/// `residuals_at` receives a candidate parameter vector and a buffer to fill,
/// returning `false` if the evaluation failed.
///
/// `out` is **column-major** with `base.len()` rows and `params.len()`
/// columns — the layout `nalgebra` stores matrices in and the one
/// `DMatrix::from_vec` expects. Filling it row-major would transpose the
/// Jacobian into something the solver happily accepts and then fits against,
/// converging to a wrong answer with no error anywhere.
///
/// Returns `false` as soon as a perturbed evaluation fails or produces a
/// non-finite difference. A `false` here means the Jacobian was not fully
/// written and must not be used.
pub fn forward_diff_jacobian<R>(
    params: &[f64],
    base: &[f64],
    mut residuals_at: R,
    out: &mut [f64],
) -> bool
where
    R: FnMut(&[f64], &mut [f64]) -> bool,
{
    let nrows = base.len();
    let ncols = params.len();
    debug_assert_eq!(out.len(), nrows * ncols);

    let mut probe = params.to_vec();
    let mut resid = vec![0.0; nrows];

    for j in 0..ncols {
        let h = fd_step(params[j]);
        probe[j] = params[j] + h;
        let ok = residuals_at(&probe, &mut resid);
        probe[j] = params[j];
        if !ok {
            return false;
        }

        let inv_h = 1.0 / h;
        for i in 0..nrows {
            let d = (resid[i] - base[i]) * inv_h;
            if !d.is_finite() {
                return false;
            }
            // Column-major: column j, row i.
            out[j * nrows + i] = d;
        }
    }
    true
}

/// 变参数协方差(外部空间),行主序平铺。
///
/// * `nvarys` —— 变参数个数,矩阵为 nvarys × nvarys。
/// * `matrix` —— 行主序平铺的协方差元素。
#[derive(Debug, Clone, PartialEq)]
pub struct Covariance {
    pub nvarys: usize,
    pub matrix: Vec<f64>,
}

impl Covariance {
    /// 各变参数的标准误:协方差对角元的平方根。
    ///
    /// 返回:长度 nvarys 的标准误向量。
    pub fn stderr(&self) -> Vec<f64> {
        (0..self.nvarys)
            .map(|i| self.matrix[i * self.nvarys + i].sqrt())
            .collect()
    }

    /// |相关系数| >= 0.1 的参数对,按 |corr| 降序。
    ///
    /// 返回:变参数下标对与相关系数的三元组序列。
    pub fn correl(&self) -> Vec<(usize, usize, f64)> {
        let mut pairs = Vec::new();
        for i in 0..self.nvarys {
            for j in (i + 1)..self.nvarys {
                let c = self.matrix[i * self.nvarys + j]
                    / (self.matrix[i * self.nvarys + i] * self.matrix[j * self.nvarys + j])
                        .sqrt();
                if c.abs() >= 0.1 {
                    pairs.push((i, j, c));
                }
            }
        }
        pairs.sort_by(|a, b| match b.2.abs().partial_cmp(&a.2.abs()) {
            Some(ord) => ord,
            None => std::cmp::Ordering::Equal,
        });
        pairs
    }
}

/// 由最终雅可比计算外部空间协方差。
///
/// * `jacobian` —— 列主序 nrows × ncols 平铺,列序与变参数序一致。
/// * `redchi` —— 约化卡方,缩放协方差(镜像 lmfit scale_covar 缺省行为)。
/// * `gradients` —— 各变参数的 scale_gradient 因子,长度 ncols。
///
/// 返回:奇异或对角含负值时为 None(转录 lmfit 的疑点协方差守卫)。
pub fn covariance(jacobian: &[f64], redchi: f64, gradients: &[f64]) -> Option<Covariance> {
    let ncols = gradients.len();
    let nrows = jacobian.len() / ncols;
    let j = DMatrix::from_vec(nrows, ncols, jacobian.to_vec());
    let jtj = &j.transpose() * &j;
    let inv = match jtj.try_inverse() {
        Some(m) => m,
        None => return None,
    };
    for i in 0..ncols {
        if inv[(i, i)] < 0.0 {
            return None;
        }
    }
    let mut matrix = Vec::with_capacity(ncols * ncols);
    for i in 0..ncols {
        for j in 0..ncols {
            matrix.push(inv[(i, j)] * redchi * gradients[i] * gradients[j]);
        }
    }
    Some(Covariance { nvarys: ncols, matrix })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The step must track the parameter's magnitude, and fall back to the
    /// bare `sqrt(f64::EPSILON)` at zero where there is no magnitude to track.
    #[test]
    fn step_scales_with_magnitude_and_falls_back_at_zero() {
        let eps = f64::EPSILON.sqrt(); // 1.49012e-8, scipy leastsq 实际步长
        assert!((fd_step(0.0) - eps).abs() < 1e-20);
        assert!((fd_step(2.0) - 2.0 * eps).abs() < 1e-20);
        assert!((fd_step(-3.0) - 3.0 * eps).abs() < 1e-20);
    }

    /// For the linear model `m(x) = slope*x + intercept` the residual
    /// `r_i = y_i - m(x_i)` has the constant Jacobian `[-x_i, -1]`, so the
    /// difference is exact up to floating-point noise rather than merely close.
    #[test]
    fn matches_the_analytic_jacobian_of_a_line() {
        let x = [0.0, 1.0, 2.5, -4.0];
        let params = vec![3.0, 0.5];

        let residuals_at = |p: &[f64], out: &mut [f64]| {
            for (i, &xi) in x.iter().enumerate() {
                out[i] = -(p[0] * xi + p[1]);
            }
            true
        };

        let mut base = vec![0.0; x.len()];
        residuals_at(&params, &mut base);

        let mut jac = vec![0.0; x.len() * params.len()];
        assert!(forward_diff_jacobian(
            &params,
            &base,
            residuals_at,
            &mut jac
        ));

        let nrows = x.len();
        for (i, &xi) in x.iter().enumerate() {
            // Column-major: parameter j, data point i.
            let d_slope = jac[i];
            let d_intercept = jac[nrows + i];
            // A forward difference on a linear function is exact, so the only
            // error is the division by h. 步长 1.49e-8 时差商舍入误差下限为
            // 机器精度 / h ≈ 1e-8 量级,1e-9 断言在合法数值下必然失败,故取 1e-6。
            assert!(
                (d_slope - (-xi)).abs() < 1e-6,
                "row {i}: {d_slope} vs {}",
                -xi
            );
            assert!(
                (d_intercept - (-1.0)).abs() < 1e-6,
                "row {i}: {d_intercept}"
            );
        }
    }

    /// One Jacobian costs exactly `n + 1` residual evaluations when the base
    /// residuals are supplied — the caller's already-computed values are
    /// reused rather than recomputed.
    #[test]
    fn uses_exactly_one_evaluation_per_parameter() {
        let params = vec![1.0, 2.0, 3.0, 4.0];
        let base = vec![0.0; 5];
        let mut calls = 0usize;

        let mut jac = vec![0.0; 5 * 4];
        assert!(forward_diff_jacobian(
            &params,
            &base,
            |_p: &[f64], out: &mut [f64]| {
                calls += 1;
                out.fill(0.0);
                true
            },
            &mut jac,
        ));

        assert_eq!(calls, params.len());
    }

    /// A non-finite difference aborts the whole Jacobian rather than leaving
    /// the caller to fit against a matrix with a NaN column in it.
    #[test]
    fn aborts_on_non_finite_values() {
        let params = vec![1.0, 2.0];
        let base = vec![1.0, 2.0];

        // Residuals blow up at the second parameter's perturbation.
        let mut jac = vec![0.0; 2 * 2];
        let ok = forward_diff_jacobian(
            &params,
            &base,
            |p: &[f64], out: &mut [f64]| {
                if p[1] != 2.0 {
                    out[0] = f64::NAN;
                    out[1] = 0.0;
                } else {
                    out.copy_from_slice(&base);
                }
                true
            },
            &mut jac,
        );
        assert!(!ok);
    }

    /// An evaluation that reports failure stops the sweep immediately.
    #[test]
    fn aborts_when_an_evaluation_fails() {
        let params = vec![1.0, 2.0, 3.0];
        let base = vec![1.0];
        let mut calls = 0usize;

        let mut jac = vec![0.0; 3];
        let ok = forward_diff_jacobian(
            &params,
            &base,
            |_p: &[f64], out: &mut [f64]| {
                calls += 1;
                out[0] = 0.0;
                calls < 2
            },
            &mut jac,
        );

        assert!(!ok);
        assert_eq!(calls, 2, "should stop at the first failure, not sweep on");
    }

    /// 手算黄金:J 列为 [-x, -1](线性模型残差对两参数的导数),
    /// x = [-1, 0, 2] 时 J^T J = [[5, 1], [1, 3]],det = 14,redchi = 2。
    /// cov = (J^T J)^-1 * redchi = [[3, -1], [-1, 5]] / 7,stderr = sqrt(diag)。
    #[test]
    fn covariance_matches_a_hand_computed_reference() {
        // 列主序,nrows = 3(数据点),ncols = 2(参数)。
        let jac = [1.0, 0.0, -2.0,     // 参数 0 列:∂r/∂a = -x
                   -1.0, -1.0, -1.0];  // 参数 1 列:∂r/∂b = -1
        let cov = match covariance(&jac, 2.0, &[1.0, 1.0]) {
            Some(c) => c,
            None => panic!("invertible matrix must yield a covariance"),
        };
        assert_eq!(cov.nvarys, 2);
        let expect = [3.0 / 7.0, -1.0 / 7.0, -1.0 / 7.0, 5.0 / 7.0];
        for (got, want) in cov.matrix.iter().zip(expect) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
        let se = cov.stderr();
        assert!((se[0] - (3.0_f64 / 7.0).sqrt()).abs() < 1e-12);
        assert!((se[1] - (5.0_f64 / 7.0).sqrt()).abs() < 1e-12);
    }

    /// 两列完全相同 -> J^T J 奇异 -> None。
    #[test]
    fn singular_jacobian_yields_none() {
        let jac = [1.0, 2.0, 1.0, 2.0];
        assert!(covariance(&jac, 1.0, &[1.0, 1.0]).is_none());
    }

    /// 相关系数:cov = [[4, 1.5], [1.5, 1]],corr = 1.5 / 2 = 0.75,越过 0.1 阈值。
    /// 低于阈值的配对不出现。
    #[test]
    fn correlations_respect_the_point_one_cutoff() {
        let cov = Covariance {
            nvarys: 3,
            matrix: vec![
                4.0, 1.5, 0.02,
                1.5, 1.0, 0.03,
                0.02, 0.03, 9.0,
            ],
        };
        let pairs = cov.correl();
        // corr(0,1) = 1.5/sqrt(4) = 0.75;corr(1,2) = 0.03/sqrt(9) = 0.01(不报);
        // corr(0,2) = 0.02/sqrt(36) ≈ 0.0033(不报)。
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 1, 0.75));
    }
}
