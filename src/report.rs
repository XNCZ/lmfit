//! The fit report.
//!
//! Presentation only — nothing here computes anything, and nothing else in the
//! crate depends on it. It exists as its own module so that [`ModelResult`]
//! stays a plain data structure, and so that formatting choices can be changed
//! without going near the numbers.
//!
//! The layout follows lmfit's: a `[[Model]]` line, a block of `[[Fit
//! Statistics]]`, then one line per `[[Variables]]`. Two of lmfit's blocks are
//! absent because the quantities behind them are not computed yet —
//! `+/- stderr` needs a covariance matrix, and `[[Correlations]]` needs the
//! same. Both are additive when they arrive.

use crate::parameter::Parameter;
use crate::result::ModelResult;
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

impl<M: ModelParams> ModelResult<M> {
    /// Render the fit as a report.
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
    /// let report = result.fit_report();
    /// assert!(report.contains("[[Fit Statistics]]"));
    /// assert!(report.contains("slope:"));
    /// # Ok::<(), lmfit::Error>(())
    /// ```
    pub fn fit_report(&self) -> String {
        let mut out = String::new();
        self.write_model(&mut out);
        self.write_statistics(&mut out);
        self.write_variables(&mut out);
        self.write_correlations(&mut out);
        out
    }

    fn write_model(&self, out: &mut String) {
        out.push_str("[[Model]]\n");
        out.push_str("    ");
        out.push_str(M::MODEL_NAME);
        out.push('\n');
    }

    fn write_statistics(&self, out: &mut String) {
        // The label column is padded to its widest entry so every `=` lines up,
        // which is the whole reason this block is readable.
        let rows: [(&str, String); 8] = [
            ("# fitting method", "leastsq".to_string()),
            ("# function evals", self.nfev.to_string()),
            ("# data points", self.ndata.to_string()),
            ("# variables", self.nvarys.to_string()),
            ("chi-square", gformat(self.chisqr)),
            ("reduced chi-square", gformat(self.redchi)),
            ("Akaike info crit", gformat(self.aic)),
            ("Bayesian info crit", gformat(self.bic)),
        ];
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

    fn write_variables(&self, out: &mut String) {
        let params: Vec<&Parameter> = self.params.iter().collect();
        if params.is_empty() {
            return;
        }
        let width = params.iter().map(|p| p.name.len()).max().unwrap_or(0);

        // value 与 stderr 两列各自按实测最宽对齐,使 (init/(fixed 标记同列。
        let value_strs: Vec<String> = params.iter().map(|p| gformat(p.value)).collect();
        let se_strs: Vec<String> = self
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

    fn write_correlations(&self, out: &mut String) {
        let covar = match &self.covar {
            Some(c) => c,
            None => return,
        };
        let names: Vec<&str> = self
            .params
            .iter()
            .filter(|p| p.vary)
            .map(|p| p.name.as_str())
            .collect();
        let pairs = covar.correl();
        if pairs.is_empty() {
            return;
        }
        out.push_str("[[Correlations]] (unreported correlations are < 0.100)\n");
        for (i, j, c) in pairs {
            out.push_str(&format!("    C({}, {}) = {}\n", names[i], names[j], gformat(c)));
        }
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
