//! The Gaussian line shape.

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
/// Because the area is what is fitted, `amplitude` is not constrained to be
/// positive — a negative value is a negative-going peak, which is meaningful
/// for a difference of two signals.
pub fn gaussian(x: f64, amplitude: f64, center: f64, sigma: f64) -> f64 {
    (amplitude / (S2PI * sigma)) * (-(x - center).powi(2) / (2.0 * sigma.powi(2))).exp()
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
        let expected = 5.0 / (std::f64::consts::TAU.sqrt() * 2.0);
        assert!((gaussian(5.0, 5.0, 5.0, 2.0) - expected).abs() < 1e-12);
    }

    /// One sigma from the centre the curve is `exp(-1/2)` of its peak, and two
    /// sigmas out it is `exp(-2)` — the property that makes the parameter
    /// readable as a width.
    #[test]
    fn falls_off_by_the_expected_factors() {
        let peak = gaussian(0.0, 1.0, 0.0, 2.0);
        assert!((gaussian(2.0, 1.0, 0.0, 2.0) - peak * (-0.5f64).exp()).abs() < 1e-12);
        assert!((gaussian(4.0, 1.0, 0.0, 2.0) - peak * (-2.0f64).exp()).abs() < 1e-12);
        assert!((gaussian(-2.0, 1.0, 0.0, 2.0) - peak * (-0.5f64).exp()).abs() < 1e-12);
    }

    /// The area under the curve really is `amplitude`, which is the claim the
    /// parameter name makes.
    #[test]
    fn integrates_to_the_amplitude() {
        // Trapezoid over ±12 sigma.
        let (amp, cen, sig) = (3.0, 1.5, 0.8);
        let (lo, hi) = (cen - 12.0 * sig, cen + 12.0 * sig);
        let n = 20_000;
        let step = (hi - lo) / n as f64;
        let area: f64 = (0..=n)
            .map(|i| {
                let x = lo + i as f64 * step;
                let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
                weight * gaussian(x, amp, cen, sig)
            })
            .sum::<f64>()
            * step;

        assert!((area - 3.0).abs() < 1e-6, "area = {area}");
    }
}
