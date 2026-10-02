//! The fit report.
//!
//! Presentation only — nothing here computes anything, and nothing else in the
//! crate depends on it. It exists as its own module so that [`ModelResult`]
//! stays a plain data structure, and so that formatting choices can be changed
//! without going near the numbers.
//!
//! The layout follows lmfit's: a `[[Model]]` line, a block of `[[Fit
//! Statistics]]`, then one line per `[[Variables]]`, then `[[Correlations]]`
//! when a covariance was computed. 实数与复数结果各自实现
//! [`Display`](std::fmt::Display),但渲染只有 `render` 一处——格式不存在第二份
//! 可漂移的实现。

use std::fmt;

use crate::numerics::Covariance;
use crate::parameter::{Parameter, Parameters};
use crate::result::{ComplexResult, ModelResult};
use crate::traits::ModelParams;

/// Number formatting, modelled on lmfit's `gformat`.
///
/// Fixed notation for middling magnitudes and scientific outside them, with
/// trailing zeros trimmed so a round value prints as `5` rather than
/// `5.0000000`. This is not a character-for-character reproduction of lmfit's
/// exponent-window arithmetic — it follows the same intent.
pub(crate) fn gformat(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }

    let magnitude = value.abs();
    let raw = if (1.0e-4..1.0e5).contains(&magnitude) {
        format!("{value:.7}")
    } else {
        format!("{value:.6e}")
    };
    trim_trailing_zeros(&raw)
}

/// Strip the insignificant zeros from a formatted float, leaving any exponent
/// untouched.
fn trim_trailing_zeros(formatted: &str) -> String {
    let (mantissa, exponent) = match formatted.split_once('e') {
        Some((mantissa, exponent)) => (mantissa, Some(exponent)),
        None => (formatted, None),
    };

    let trimmed = if mantissa.contains('.') {
        mantissa.trim_end_matches('0').trim_end_matches('.')
    } else {
        mantissa
    };

    match exponent {
        Some(exponent) => format!("{trimmed}e{exponent}"),
        None => trimmed.to_string(),
    }
}

/// 报告所需的共享字段:实数与复数结果各自组装后共用同一渲染。
pub(crate) struct ReportParts<'a> {
    /// 模型名(`[[Model]]` 行)。
    pub model_name: &'a str,
    /// 参数表(含起始值与固定标记)。
    pub params: &'a Parameters,
    /// 逐参数标准误,固定参数为 None。
    pub stderr: &'a [Option<f64>],
    /// 变参数协方差,不可得时为 None。
    pub covar: Option<&'a Covariance>,
    /// 拟合是否达到收敛判据;报告仅在失败时展开原因。
    pub success: bool,
    /// 收敛与否的文字说明:失败时作为 `reason` 行进报告。
    pub message: &'a str,
    /// 残差求值次数。
    pub nfev: usize,
    /// 数据点数(复数口径为槽位数 2n)。
    pub ndata: usize,
    /// 变参数个数。
    pub nvarys: usize,
    /// 卡方。
    pub chisqr: f64,
    /// 约化卡方。
    pub redchi: f64,
    /// Akaike 信息准则。
    pub aic: f64,
    /// Bayesian 信息准则。
    pub bic: f64,
}

/// 按 `[[Model]]`、`[[Fit Statistics]]`、`[[Variables]]`、`[[Correlations]]`
/// 顺序渲染报告。
///
/// * `parts` —— 报告共享字段。
///
/// 返回:完整报告文本。
pub(crate) fn render(parts: ReportParts<'_>) -> String {
    let mut out = String::new();
    handle_model(&parts, &mut out);
    handle_statistics(&parts, &mut out);
    handle_variables(&parts, &mut out);
    handle_correlations(&parts, &mut out);
    out
}

fn handle_model(parts: &ReportParts<'_>, out: &mut String) {
    out.push_str("[[Model]]\n");
    out.push_str("    ");
    out.push_str(parts.model_name);
    out.push('\n');
}

fn handle_statistics(parts: &ReportParts<'_>, out: &mut String) {
    // The label column is padded to its widest entry so every `=` lines up,
    // which is the whole reason this block is readable.
    let mut rows: Vec<(&str, String)> = vec![
        ("fitting method", "leastsq".to_string()),
        (
            "state",
            if parts.success { "success" } else { "failure" }.to_string(),
        ),
        ("function evals", parts.nfev.to_string()),
        ("data points", parts.ndata.to_string()),
        ("variables", parts.nvarys.to_string()),
        ("chi-square", gformat(parts.chisqr)),
        ("reduced chi-square", gformat(parts.redchi)),
        ("Akaike info crit", gformat(parts.aic)),
        ("Bayesian info crit", gformat(parts.bic)),
    ];
    // 失败原因紧跟 state,且只在失败时出现——成功时那行会是噪音。
    if !parts.success {
        rows.insert(2, ("reason", parts.message.to_string()));
    }
    let width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0);

    out.push_str("[[Fit Statistics]]\n");
    for (label, value) in rows {
        out.push_str("    ");
        out.push_str(label);
        out.push_str(&" ".repeat(width - label.len()));
        out.push_str(" = ");
        out.push_str(&value);
        out.push('\n');
    }
}

fn handle_variables(parts: &ReportParts<'_>, out: &mut String) {
    let params: Vec<&Parameter> = parts.params.iter().collect();
    if params.is_empty() {
        return;
    }
    let width = params.iter().map(|p| p.name.len()).max().unwrap_or(0);

    // value 与 stderr 两列各自按实测最宽对齐,使 (init/(fixed 标记同列。
    let value_strs: Vec<String> = params.iter().map(|p| gformat(p.value)).collect();
    let se_strs: Vec<String> = parts
        .stderr
        .iter()
        .zip(params.iter())
        .map(|(se, p)| match (se, p.vary) {
            (Some(s), true) => format!("+/- {}", gformat(*s)),
            (None, true) | (_, false) => String::new(),
        })
        .collect();
    let value_width = value_strs.iter().map(|s| s.len()).max().unwrap_or(0);
    let se_width = se_strs.iter().map(|s| s.len()).max().unwrap_or(0);

    out.push_str("[[Variables]]\n");
    for (i, p) in params.iter().enumerate() {
        out.push_str("    ");
        out.push_str(&p.name);
        out.push(':');
        out.push_str(&" ".repeat(width - p.name.len()));
        out.push(' ');
        out.push_str(&" ".repeat(value_width - value_strs[i].len()));
        out.push_str(&value_strs[i]);
        out.push(' ');
        out.push_str(&se_strs[i]);
        out.push_str(&" ".repeat(se_width - se_strs[i].len()));
        out.push(' ');
        if p.vary {
            out.push_str("(init = ");
            out.push_str(&gformat(p.init));
            out.push(')');
        } else {
            out.push_str("(fixed)");
        }
        out.push('\n');
    }
}

fn handle_correlations(parts: &ReportParts<'_>, out: &mut String) {
    let covar = match parts.covar {
        Some(c) => c,
        None => return,
    };
    let names: Vec<&str> = parts
        .params
        .iter()
        .filter(|p| p.vary)
        .map(|p| p.name.as_str())
        .collect();
    let pairs = covar.correl();
    if pairs.is_empty() {
        return;
    }
    // `C(a, b)` 补到等宽:lmfit 的 `lspace = maxlen - len(name)` 同理,只是这里
    // 让它与报表其余两块一样把 `=` 对齐成列。
    let labels: Vec<String> = pairs
        .iter()
        .map(|pair| format!("C({}, {})", names[pair.0], names[pair.1]))
        .collect();
    let width = labels.iter().map(|label| label.len()).max().unwrap_or(0);

    out.push_str("[[Correlations]] (unreported correlations are < 0.100)\n");
    for (pair, label) in pairs.iter().zip(&labels) {
        out.push_str("    ");
        out.push_str(label);
        out.push_str(&" ".repeat(width - label.len()));
        out.push_str(" = ");
        out.push_str(&gformat(pair.2));
        out.push('\n');
    }
}

/// 渲染为 lmfit 风格的拟合报表,`println!("{result}")` 即可打印。
///
/// 报告以换行结尾:`println!` 之后会多一个空行,`print!` 恰好收在一行。
///
/// ```
/// # use lmfit::{Curve, Model};
/// # #[derive(Model)]
/// # struct Line {
/// #     #[param(value = 1.0)] slope: f64,
/// #     #[param(value = 0.0)] intercept: f64,
/// # }
/// # impl Curve for Line {
/// #     fn eval(&self, x: f64) -> f64 { self.slope * x + self.intercept }
/// # }
/// let x = vec![0.0, 1.0, 2.0, 3.0];
/// let y = vec![1.0, 3.0, 5.0, 7.0];
/// let result = Line::default().fit(&y, &x)?;
///
/// let report = result.to_string();
/// assert!(report.contains("[[Fit Statistics]]"));
/// assert!(report.contains("state              = success"));
/// assert!(report.contains("slope:"));
/// # Ok::<(), lmfit::Error>(())
/// ```
impl<M: ModelParams> fmt::Display for ModelResult<M> {
    /// 写出一份完整报表;渲染本体见 `render`。
    ///
    /// * `f` —— 目标 formatter。
    ///
    /// 返回:写入结果;渲染本身不会失败。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&render(ReportParts {
            model_name: M::MODEL_NAME,
            params: &self.params,
            stderr: &self.stderr,
            covar: self.covar.as_ref(),
            success: self.success,
            message: &self.message,
            nfev: self.nfev,
            ndata: self.ndata,
            nvarys: self.nvarys,
            chisqr: self.chisqr,
            redchi: self.redchi,
            aic: self.aic,
            bic: self.bic,
        }))
    }
}

/// 渲染复数拟合的报表,与实数结果同构:渲染共用,`ndata` 为槽位数 2n。
impl<M: ModelParams> fmt::Display for ComplexResult<M> {
    /// 写出一份完整报表;渲染本体见 `render`。
    ///
    /// * `f` —— 目标 formatter。
    ///
    /// 返回:写入结果;渲染本身不会失败。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&render(ReportParts {
            model_name: M::MODEL_NAME,
            params: &self.params,
            stderr: &self.stderr,
            covar: self.covar.as_ref(),
            success: self.success,
            message: &self.message,
            nfev: self.nfev,
            ndata: self.ndata,
            nvarys: self.nvarys,
            chisqr: self.chisqr,
            redchi: self.redchi,
            aic: self.aic,
            bic: self.bic,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_round_numbers_without_trailing_zeros() {
        assert_eq!(gformat(0.0), "0");
        assert_eq!(gformat(5.0), "5");
        assert_eq!(gformat(-1.0), "-1");
        assert_eq!(gformat(2.5), "2.5");
        assert_eq!(gformat(0.75), "0.75");
    }

    #[test]
    fn uses_scientific_notation_outside_the_fixed_window() {
        assert_eq!(gformat(1.0e-6), "1e-6");
        assert_eq!(gformat(1.5e7), "1.5e7");
        assert_eq!(gformat(-2.5e-9), "-2.5e-9");
    }

    /// Values inside the window keep fixed notation even when small.
    #[test]
    fn stays_fixed_inside_the_window() {
        assert_eq!(gformat(1.0e-4), "0.0001");
        assert_eq!(gformat(1234.5), "1234.5");
    }

    /// Non-finite values must render as words, not as `NaN`/`inf` leaking into
    /// a table that is otherwise numeric.
    #[test]
    fn renders_non_finite_values_as_words() {
        assert_eq!(gformat(f64::INFINITY), "inf");
        assert_eq!(gformat(f64::NEG_INFINITY), "-inf");
        assert_eq!(gformat(f64::NAN), "nan");
    }

    #[test]
    fn trims_zeros_without_disturbing_the_exponent() {
        assert_eq!(trim_trailing_zeros("1.5000000e-9"), "1.5e-9");
        assert_eq!(trim_trailing_zeros("1.0000000e5"), "1e5");
        assert_eq!(trim_trailing_zeros("5.0000000"), "5");
        assert_eq!(trim_trailing_zeros("5"), "5");
    }
}
