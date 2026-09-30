//! The Gaussian line shape.

use crate::{Curve, Model};

/// `sqrt(2π)`, lmfit's `lineshapes.s2pi`.
///
/// A literal because `f64::sqrt` is not a `const fn`; a unit test pins it
/// against `TAU.sqrt()` so a typo cannot survive.
const S2PI: f64 = 2.506_628_274_631_000_2;

/// A normalised Gaussian.
///
/// `amplitude` is the **area** under the curve, not its peak height — this is
/// lmfit's convention, and it is the opposite of the `amp * exp(..)` form most
/// people write from memory. The peak is
/// `amplitude / (sqrt(2π) · sigma)` tall, at `x = center`.
///
/// ```
/// use lmfit::models::Gaussian;
/// use lmfit::Curve;
///
/// let g = Gaussian { amplitude: 5.0, center: 5.0, sigma: 2.0 };
/// assert!((g.eval(5.0) - 5.0 / (std::f64::consts::TAU.sqrt() * 2.0)).abs() < 1e-12);
/// ```
///
/// Because the area is what is fitted, `amplitude` is not constrained to be
/// positive — a negative value is a negative-going peak, which is meaningful
/// for a difference of two signals.
#[derive(Model)]
pub struct Gaussian {
    /// Total area under the curve.
    #[param(value = 1.0)]
    pub amplitude: f64,
    /// Position of the peak.
    #[param(value = 0.0)]
    pub center: f64,
    /// Standard deviation. Bounded below because a negative width is not a
    /// narrower curve, it is a different (divergent) function.
    #[param(value = 1.0, min = 0.0)]
    pub sigma: f64,
}

impl Curve for Gaussian {
    fn eval(&self, x: f64) -> f64 {
        (self.amplitude / (S2PI * self.sigma))
            * (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s2pi_matches_the_computed_value() {
        assert!((S2PI - std::f64::consts::TAU.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn peak_height_is_area_over_sqrt_two_pi_sigma() {
        let g = Gaussian {
            amplitude: 5.0,
            center: 5.0,
            sigma: 2.0,
        };
        let expected = 5.0 / (std::f64::consts::TAU.sqrt() * 2.0);
        assert!((g.eval(5.0) - expected).abs() < 1e-12);
    }

    /// One sigma from the centre the curve is `exp(-1/2)` of its peak, and two
    /// sigmas out it is `exp(-2)` — the property that makes the parameter
    /// readable as a width.
    #[test]
    fn falls_off_by_the_expected_factors() {
        let g = Gaussian {
            amplitude: 1.0,
            center: 0.0,
            sigma: 2.0,
        };
        let peak = g.eval(0.0);
        assert!((g.eval(2.0) - peak * (-0.5f64).exp()).abs() < 1e-12);
        assert!((g.eval(4.0) - peak * (-2.0f64).exp()).abs() < 1e-12);
        assert!((g.eval(-2.0) - peak * (-0.5f64).exp()).abs() < 1e-12);
    }

    /// The area under the curve really is `amplitude`, which is the claim the
    /// parameter name makes.
    #[test]
    fn integrates_to_the_amplitude() {
        let g = Gaussian {
            amplitude: 3.0,
            center: 1.5,
            sigma: 0.8,
        };
        // Trapezoid over ±12 sigma.
        let (lo, hi) = (1.5 - 12.0 * 0.8, 1.5 + 12.0 * 0.8);
        let n = 20_000;
        let step = (hi - lo) / n as f64;
        let area: f64 = (0..=n)
            .map(|i| {
                let x = lo + i as f64 * step;
                let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
                weight * g.eval(x)
            })
            .sum::<f64>()
            * step;

        assert!((area - 3.0).abs() < 1e-6, "area = {area}");
    }

    #[test]
    fn parameter_names_match_lmfit() {
        use crate::ModelParams;
        let names: Vec<String> = Gaussian::default()
            .specs()
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(names, vec!["amplitude", "center", "sigma"]);
    }
}
