//! 观测数据:一个点、一组点,以及"带不带测量不确定度"的类型级区分。
//!
//! 这一层的存在是为了让"数据长什么样"和"模型算什么"分开。求解器不认识曲线,
//! 只认识一个 [`DataSet`]——它按点给出 `(x, y, 权重)`;数据是逐点绑定还是三个
//! 平行切片,由实现决定,求解器不知道也不关心。
//!
//! # 为什么权重是类型而不是 `Option`
//!
//! [`Dat`] 的第三个参数是 [`NoSigma`] 或 [`Sigma`],不是一个 `bool` 或
//! `Option<f64>`。这样做的收益有三条,没有一条是性能:
//!
//! * **不可能自相矛盾。** 一个 `bool` 常量可以声明"不加权"却带着真实的 σ;
//!   而 [`NoSigma::weigh`](Weight::weigh) 就是恒等函数,没有可以写错的配对。
//! * **不加权的点是零字节。** [`NoSigma`] 是零尺寸类型,`size_of::<Dat>()` 是
//!   16,`size_of::<Dat<Sigma>>()` 是 24——而不是逐点 `Option<f64>` 的 32。
//! * **一个数据集里混不了。** 带 σ 与不带 σ 的点不能放进同一个 `Vec`,
//!   这看起来是限制,实际是好事:混合权重下 `chisqr`、`redchi` 与协方差
//!   的口径说不清楚。
//!
//! # 两个消费者
//!
//! 权重挂在数据上而不是求解器的形参上,因为它有**两个**消费者:残差写入按
//! [`Weight::weigh`] 加权,解析雅可比组装按 [`Weight::weigh_deriv`] 缩放。
//! 两者同源是必须的——只把加权做进残差会让有限差分路径正确、解析路径静默走偏。

use crate::error::{Error, Result};
use num_complex::Complex64;

/// 该点不带测量不确定度。
///
/// 零尺寸类型。作为 [`Dat`] 的权重字段出现时不占任何字节。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoSigma;

/// 该点的一倍标准差。
///
/// 字段私有,只能经 [`Sigma::new`] 或 [`Dat::with_sigma`] 构造。构造本身不校验
/// ——`new` 是 `const fn`——校验发生在求解前的 [`DataSet::validate`]。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sigma(f64);

impl Sigma {
    /// 包装一个标准差。
    #[inline(always)]
    pub const fn new(sigma: f64) -> Self {
        Self(sigma)
    }
}

/// 一个点的权重在热循环里的行为。
///
/// 这是整个 crate **唯一**区分"带 σ / 不带 σ"的地方。三个方法都是
/// `#[inline(always)]`,于是不加权路径上既没有除法也没有分支——与不带权重的
/// 那条路径生成的机器码相同。
pub trait Weight: Copy {
    /// 一倍标准差。不加权恒为 `1.0`。
    fn sigma(self) -> f64;

    /// 把原始差 `y - f` 换算成求解器空间的残差。
    #[inline(always)]
    fn weigh(self, d: f64) -> f64 {
        d / self.sigma()
    }

    /// `d(weigh)/dd` —— 解析雅可比按它把 `∂f/∂θ` 缩放成 `∂r/∂θ`。
    ///
    /// 见模块文档「两个消费者」。
    #[inline(always)]
    fn weigh_deriv(self) -> f64 {
        1.0 / self.sigma()
    }
}

impl Weight for NoSigma {
    #[inline(always)]
    fn sigma(self) -> f64 {
        1.0
    }

    /// 恒等——不加权路径没有除法。
    #[inline(always)]
    fn weigh(self, d: f64) -> f64 {
        d
    }

    #[inline(always)]
    fn weigh_deriv(self) -> f64 {
        1.0
    }
}

impl Weight for Sigma {
    #[inline(always)]
    fn sigma(self) -> f64 {
        self.0
    }
}

/// 一个观测点:`x`、`y` 与权重绑定在一起。
///
/// | 写法 | 含义 | `size_of` |
/// | --- | --- | --- |
/// | `Dat` | 实数,不加权 | 16 |
/// | `Dat<Sigma>` | 实数,带 σ | 24 |
/// | `ComplexDat` | 复数,x 为 `f64` | 16 |
/// | `ComplexDat<Sigma>` | 复数,带 σ | 24 |
///
/// `X` 与 `Y` 是自变量与观测值的类型,实数路径都是 `f64`;复数路径的 `Y` 固定为
/// [`Complex64`],`X` 可实可复,与 [`ComplexCurve`] 的口径一致。
///
/// 权重字段私有,构造走 [`Dat::new`] 或 [`Dat::with_sigma`]。
///
/// [`ComplexCurve`]: crate::traits::ComplexCurve
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dat<W = NoSigma, X = f64, Y = f64> {
    /// 自变量。
    pub x: X,
    /// 观测值。
    pub y: Y,
    /// 私有:构造必须走 `new` / `with_sigma`。
    w: W,
}

/// 复数观测点。[`ComplexCurve`](crate::traits::ComplexCurve) 用的形状。
pub type ComplexDat<W = NoSigma, X = f64> = Dat<W, X, Complex64>;

impl<X, Y> Dat<NoSigma, X, Y> {
    /// 一个不带测量不确定度的观测点。
    pub const fn new(x: X, y: Y) -> Self {
        Self {
            x,
            y,
            w: NoSigma,
        }
    }
}

impl<X, Y> Dat<Sigma, X, Y> {
    /// 一个带一倍标准差 `sigma` 的观测点。
    pub const fn with_sigma(x: X, y: Y, sigma: f64) -> Self {
        Self {
            x,
            y,
            w: Sigma::new(sigma),
        }
    }
}

impl<W: Weight, X, Y> Dat<W, X, Y> {
    /// 一倍标准差。不加权时恒为 `1.0`。
    #[inline(always)]
    pub fn sigma(&self) -> f64 {
        self.w.sigma()
    }

    /// 该点的权重标记。
    #[inline(always)]
    pub fn weight(&self) -> W {
        self.w
    }
}

/// 求解器眼里的实数数据集。
///
/// 对 `S` 单态化,热循环里既没有动态派发,也没有"带不带权重"的运行期分支——
/// 那一区分由关联类型 [`DataSet::W`] 承担。
///
/// 四个实现覆盖两种布局:逐点绑定的 `&[Dat<_>]`,与三个平行切片的
/// `(&y, &x)` / `(&y, &x, &sigma)`(参数序沿用旧版 `fit(y, x)` 与
/// `fit_sigma(y, x, sigma)`)。
///
/// 要求 `Copy`,因为数据集是一个**视图**而不是容器:它进求解器,求解完还要
/// 留在手里装配结果。四个实现全是借用,天然满足。
pub trait DataSet: Copy {
    /// 该数据集逐点携带的权重类型:[`NoSigma`] 或 [`Sigma`]。
    type W: Weight;

    /// 数据点个数。
    fn len(&self) -> usize;

    /// 是否没有数据点。
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 第 `i` 点,按 `(x, y, 权重)` 给出。
    ///
    /// # Panics
    ///
    /// 实现可以假定 `i < len()`。
    fn point(&self, i: usize) -> (f64, f64, Self::W);

    /// 求解前的自检,由 `fit` 在装配之前调用。
    ///
    /// 逐点绑定的数据集(一次切片)长度天然自洽,缺省体恒 `Ok(())`;
    /// 平行切片的数据集在这里查长度是否一致、σ 是否都是有限的正数。
    ///
    /// # Errors
    ///
    /// [`DimensionMismatch`](crate::Error::DimensionMismatch)、
    /// [`SigmaMismatch`](crate::Error::SigmaMismatch)、
    /// [`InvalidSigma`](crate::Error::InvalidSigma)。
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// 复数数据集。[`ComplexCurve`](crate::traits::ComplexCurve) 的求解器视图。
///
/// `σ` 仍是实数、每个复频点一个:实部与虚部同权,与旧版
/// `ComplexCurve::fit_sigma` 的口径一致。
pub trait ComplexDataSet: Copy {
    /// 自变量元素类型。
    type X: Copy + Into<Complex64>;

    /// 该数据集逐点携带的权重类型。
    type W: Weight;

    /// 数据点个数。
    fn len(&self) -> usize;

    /// 是否没有数据点。
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 第 `i` 点,按 `(x, y, 权重)` 给出。
    ///
    /// # Panics
    ///
    /// 实现可以假定 `i < len()`。
    fn point(&self, i: usize) -> (Self::X, Complex64, Self::W);

    /// 求解前的自检,与 [`DataSet::validate`] 同义。
    ///
    /// # Errors
    ///
    /// 同 [`DataSet::validate`]。
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 逐点绑定。`AsRef` 而非具体切片,于是 `&[Dat<_>]` 与 `&Vec<Dat<_>>` 都能直接传。
// ---------------------------------------------------------------------------

impl<Wt: Weight> DataSet for &[Dat<Wt>] {
    type W = Wt;

    fn len(&self) -> usize {
        <[Dat<Wt>]>::len(self)
    }

    #[inline(always)]
    fn point(&self, i: usize) -> (f64, f64, Wt) {
        let p = &self[i];
        (p.x, p.y, p.w)
    }
}

/// `&Vec<Dat<_>>` 也直接可用,免得每个调用点写 `.as_slice()`。
impl<Wt: Weight> DataSet for &Vec<Dat<Wt>> {
    type W = Wt;

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    #[inline(always)]
    fn point(&self, i: usize) -> (f64, f64, Wt) {
        self.as_slice().point(i)
    }
}

impl<Wt: Weight, X: Copy + Into<Complex64>> ComplexDataSet for &[Dat<Wt, X, Complex64>] {
    type X = X;
    type W = Wt;

    fn len(&self) -> usize {
        <[Dat<Wt, X, Complex64>]>::len(self)
    }

    #[inline(always)]
    fn point(&self, i: usize) -> (X, Complex64, Wt) {
        let p = &self[i];
        (p.x, p.y, p.w)
    }
}

impl<Wt: Weight, X: Copy + Into<Complex64>> ComplexDataSet for &Vec<Dat<Wt, X, Complex64>> {
    type X = X;
    type W = Wt;

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    #[inline(always)]
    fn point(&self, i: usize) -> (X, Complex64, Wt) {
        self.as_slice().point(i)
    }
}

/// 一个只读的逐元素视图:`[T]`、`[T; N]`、`Vec<T>` 都实现它。
///
/// 它存在的理由有两条,都不是性能:
///
/// * **coherence。** 平行切片的 `impl` 里,元素类型必须能被编译器从容器类型
///   **唯一推出**。`AsRef<[X]>` 做不到——它的类型参数不是关联类型,于是
///   `impl<Y, XS> ComplexDataSet for (&Y, &XS)` 里的 `X` 会报 E0207。
///   `AsSlice::Item` 是关联类型,推得出。
/// * **省掉 `.as_slice()`。** `&Vec<T>` 与 `&[T]` 在这里是同一个东西,调用点
///   不必关心手里是哪种。
pub trait AsSlice {
    /// 元素类型。
    type Item;

    /// 借出底层切片。
    fn elements(&self) -> &[Self::Item];
}

impl<T> AsSlice for [T] {
    type Item = T;

    #[inline(always)]
    fn elements(&self) -> &[T] {
        self
    }
}

impl<T, const N: usize> AsSlice for [T; N] {
    type Item = T;

    #[inline(always)]
    fn elements(&self) -> &[T] {
        self
    }
}

impl<T> AsSlice for Vec<T> {
    type Item = T;

    #[inline(always)]
    fn elements(&self) -> &[T] {
        self
    }
}

// ---------------------------------------------------------------------------
// 平行切片。**参数序是 `(y, x)`,观测在前**——沿用旧版入口 `fit(y, x)` /
// `fit_sigma(y, x, sigma)`。写反不报错,只会安静地拟合一个转置的问题。
// 拿不准就用 `Dat`,那条路没有位置歧义。
// ---------------------------------------------------------------------------

impl<Y, X> DataSet for (&Y, &X)
where
    Y: ?Sized + AsSlice<Item = f64>,
    X: ?Sized + AsSlice<Item = f64>,
{
    type W = NoSigma;

    fn len(&self) -> usize {
        self.0.elements().len()
    }

    #[inline(always)]
    fn point(&self, i: usize) -> (f64, f64, NoSigma) {
        (self.1.elements()[i], self.0.elements()[i], NoSigma)
    }

    fn validate(&self) -> Result<()> {
        let (y, x) = (self.0.elements(), self.1.elements());
        match y.len() == x.len() {
            true => Ok(()),
            false => Err(Error::DimensionMismatch {
                x: x.len(),
                y: y.len(),
            }),
        }
    }
}

impl<Y, X, S> DataSet for (&Y, &X, &S)
where
    Y: ?Sized + AsSlice<Item = f64>,
    X: ?Sized + AsSlice<Item = f64>,
    S: ?Sized + AsSlice<Item = f64>,
{
    type W = Sigma;

    fn len(&self) -> usize {
        self.0.elements().len()
    }

    #[inline(always)]
    fn point(&self, i: usize) -> (f64, f64, Sigma) {
        (
            self.1.elements()[i],
            self.0.elements()[i],
            Sigma::new(self.2.elements()[i]),
        )
    }

    fn validate(&self) -> Result<()> {
        let (y, x, sigma) = (self.0.elements(), self.1.elements(), self.2.elements());
        if y.len() != x.len() {
            return Err(Error::DimensionMismatch {
                x: x.len(),
                y: y.len(),
            });
        }
        if y.len() != sigma.len() {
            return Err(Error::SigmaMismatch {
                npoints: y.len(),
                sigma: sigma.len(),
            });
        }
        check_sigma(sigma)
    }
}

impl<Y, XS> ComplexDataSet for (&Y, &XS)
where
    Y: ?Sized + AsSlice<Item = Complex64>,
    XS: ?Sized + AsSlice,
    XS::Item: Copy + Into<Complex64>,
{
    type X = XS::Item;
    type W = NoSigma;

    fn len(&self) -> usize {
        self.0.elements().len()
    }

    #[inline(always)]
    fn point(&self, i: usize) -> (XS::Item, Complex64, NoSigma) {
        (self.1.elements()[i], self.0.elements()[i], NoSigma)
    }

    fn validate(&self) -> Result<()> {
        let (y, x) = (self.0.elements(), self.1.elements());
        match y.len() == x.len() {
            true => Ok(()),
            false => Err(Error::DimensionMismatch {
                x: x.len(),
                y: y.len(),
            }),
        }
    }
}

impl<Y, XS, S> ComplexDataSet for (&Y, &XS, &S)
where
    Y: ?Sized + AsSlice<Item = Complex64>,
    XS: ?Sized + AsSlice,
    XS::Item: Copy + Into<Complex64>,
    S: ?Sized + AsSlice<Item = f64>,
{
    type X = XS::Item;
    type W = Sigma;

    fn len(&self) -> usize {
        self.0.elements().len()
    }

    #[inline(always)]
    fn point(&self, i: usize) -> (XS::Item, Complex64, Sigma) {
        (
            self.1.elements()[i],
            self.0.elements()[i],
            Sigma::new(self.2.elements()[i]),
        )
    }

    fn validate(&self) -> Result<()> {
        let (y, x, sigma) = (self.0.elements(), self.1.elements(), self.2.elements());
        if y.len() != x.len() {
            return Err(Error::DimensionMismatch {
                x: x.len(),
                y: y.len(),
            });
        }
        if y.len() != sigma.len() {
            return Err(Error::SigmaMismatch {
                npoints: y.len(),
                sigma: sigma.len(),
            });
        }
        check_sigma(sigma)
    }
}

/// 逐点 σ 的合法性:每项有限且为正。
fn check_sigma(sigma: &[f64]) -> Result<()> {
    for (index, &value) in sigma.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::InvalidSigma { index, value });
        }
    }
    Ok(())
}
