//! The constant line shape.

use crate::{Curve, Model};

/// A flat background.
///
/// Pairs with a peak model to absorb an offset:
///
/// ```
/// use lmfit::{Curve, models::{Constant, Gaussian}};
///
/// # let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
/// # let y: Vec<f64> = x.iter()
/// #     .map(|&t| 5.0 / (std::f64::consts::TAU.sqrt() * 2.0)
/// #         * (-(t - 5.0f64).powi(2) / 8.0).exp() + 0.75)
/// #     .collect();
/// let model = Gaussian { amplitude: 4.0, center: 4.0, sigma: 1.5 } + Constant { c: 0.0 };
/// let result = model.fit(&y, &x)?;
///
/// assert!((result.model.b.c - 0.75).abs() < 1e-6);
/// # Ok::<(), lmfit::Error>(())
/// ```
#[derive(Model)]
pub struct Constant {
    /// The value returned everywhere.
    #[param(value = 0.0)]
    pub c: f64,
}

impl Curve for Constant {
    fn eval(&self, _x: f64) -> f64 {
        self.c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_its_value_at_every_point() {
        let c = Constant { c: -2.5 };
        for x in [-100.0, 0.0, 0.5, 1e6] {
            assert_eq!(c.eval(x), -2.5);
        }
    }

    #[test]
    fn parameter_name_matches_lmfit() {
        use crate::ModelParams;
        let names: Vec<String> = Constant::default()
            .specs()
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(names, vec!["c"]);
    }

    /// A lone constant has nothing to vary against a flat dataset, but the fit
    /// must still succeed and report the mean.
    #[test]
    fn fits_a_flat_dataset() {
        use crate::Curve as _;
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![2.0, 2.0, 2.0, 2.0];
        let result = Constant::default().fit(&y, &x).expect("fit ran");
        assert!(result.success, "{}", result.message);
        assert!((result.model.c - 2.0).abs() < 1e-8);
    }
}
