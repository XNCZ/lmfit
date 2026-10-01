//! The constant line shape.

/// A flat background.
///
/// Absorbs an offset when added to a peak:
///
/// ```
/// # use lmfit::{Curve, Model, lineshapes};
/// # #[derive(Model)]
/// # struct Peak {
/// #     #[param(value = 5.0)] amplitude: f64,
/// #     #[param(value = 5.0)] center: f64,
/// #     #[param(value = 2.0, min = 0.0)] sigma: f64,
/// #     #[param(value = 0.0)] background: f64,
/// # }
/// # impl Curve for Peak {
/// #     fn eval(&self, x: f64) -> f64 {
/// #         lineshapes::gaussian(x, self.amplitude, self.center, self.sigma)
/// #             + lineshapes::constant(x, self.background)
/// #     }
/// # }
/// # let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
/// # let y: Vec<f64> = x.iter()
/// #     .map(|&t| 5.0 / (std::f64::consts::TAU.sqrt() * 2.0)
/// #         * (-(t - 5.0f64).powi(2) / 8.0).exp() + 0.75)
/// #     .collect();
/// let result = Peak::default().fit(&y, &x)?;
///
/// assert!((result.model.background - 0.75).abs() < 1e-6);
/// # Ok::<(), lmfit::Error>(())
/// ```
///
/// The `x` argument is unused; it is there so every line shape shares one
/// signature, which is what lets them be combined as plain arithmetic.
pub fn constant(_x: f64, c: f64) -> f64 {
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_its_value_at_every_point() {
        for x in [-100.0, 0.0, 0.5, 1e6] {
            assert_eq!(constant(x, -2.5), -2.5);
        }
    }
}
