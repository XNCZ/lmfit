//! lmfit's bounded-parameter transform.
//!
//! The Levenberg-Marquardt solver has no notion of box constraints, so a
//! bounded parameter is mapped into an unconstrained "internal" space before
//! the fit and mapped back once it is done. A fit therefore runs entirely in
//! internal coordinates, and bounds are enforced by construction rather than
//! by rejection.
//!
//! These formulas are transcribed from lmfit's `Parameter.setup_bounds` (the
//! forward direction) and its `from_internal` lambda (the inverse), both in
//! `lmfit/parameter.py`.
//!
//! Two properties of the inverse are inherited on purpose, and are not bugs:
//!
//! * For the one-sided transforms it is an **even** function, so
//!   `to_internal(from_internal(b)) == b.abs()` rather than `b`. The solver is
//!   free to use either sign; both name the same parameter value.
//! * The two-sided transform is **periodic** with period `2π`, and the
//!   internal→external→internal round trip folds into `[-π/2, π/2]`.
//!
//! The direction that matters in practice — external → internal → external,
//! used to seed the solver from the user's initial guess — is exact in all
//! four cases.

use crate::error::{Error, Result};

/// lmfit's `tiny` constant, from `lmfit/lineshapes.py`.
///
/// lmfit snaps an internal value of magnitude below this to exactly zero,
/// which lands the parameter exactly on its bound.
const TINY: f64 = 1.0e-15;

/// How one parameter's bounds map onto the solver's unconstrained space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transform {
    /// Unbounded: the internal value *is* the parameter value.
    Free,
    /// Lower bound only.
    Lower(f64),
    /// Upper bound only.
    Upper(f64),
    /// Both bounds, as `(min, max)` with `min < max`.
    Both(f64, f64),
}

impl Transform {
    /// Build a transform from a parameter's bounds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MinEqualsMax`] when both bounds are present and equal
    /// (there would be nothing left to vary), and [`Error::InvertedBounds`]
    /// when `min > max`. A non-finite bound is likewise rejected.
    pub fn new(min: Option<f64>, max: Option<f64>) -> Result<Self> {
        match (min, max) {
            (None, None) => Ok(Self::Free),
            (Some(lo), None) => {
                check_finite(lo, "min")?;
                Ok(Self::Lower(lo))
            }
            (None, Some(hi)) => {
                check_finite(hi, "max")?;
                Ok(Self::Upper(hi))
            }
            (Some(lo), Some(hi)) => {
                check_finite(lo, "min")?;
                check_finite(hi, "max")?;
                if lo == hi {
                    return Err(Error::MinEqualsMax {
                        name: String::from("<unnamed>"),
                        min: lo,
                    });
                }
                if lo > hi {
                    return Err(Error::InvertedBounds {
                        name: String::from("<unnamed>"),
                        min: lo,
                        max: hi,
                    });
                }
                Ok(Self::Both(lo, hi))
            }
        }
    }

    /// External parameter value → internal, unconstrained value.
    ///
    /// The input is clamped into the bounds first, matching lmfit, which
    /// clamps before transforming. That keeps this function total: it never
    /// produces a NaN, even for a value the user placed outside the bounds.
    pub fn to_internal(&self, value: f64) -> f64 {
        let value = self.clamp(value);
        let b = match *self {
            Self::Free => value,
            Self::Lower(min) => ((value - min + 1.0).powi(2) - 1.0).sqrt(),
            Self::Upper(max) => ((max - value + 1.0).powi(2) - 1.0).sqrt(),
            Self::Both(min, max) => (2.0 * (value - min) / (max - min) - 1.0).asin(),
        };
        if b.abs() < TINY { 0.0 } else { b }
    }

    /// Internal, unconstrained value → external parameter value.
    ///
    /// The result is always within the bounds.
    pub fn from_internal(&self, b: f64) -> f64 {
        match *self {
            Self::Free => b,
            Self::Lower(min) => min - 1.0 + (b * b + 1.0).sqrt(),
            Self::Upper(max) => max + 1.0 - (b * b + 1.0).sqrt(),
            Self::Both(min, max) => min + (b.sin() + 1.0) * (max - min) / 2.0,
        }
    }

    /// Clamp a value into the bounds, leaving unbounded parameters untouched.
    pub fn clamp(&self, value: f64) -> f64 {
        match *self {
            Self::Free => value,
            Self::Lower(min) => value.max(min),
            Self::Upper(max) => value.min(max),
            Self::Both(min, max) => value.clamp(min, max),
        }
    }
}

/// Reject NaN and infinite bounds, which would poison every downstream
/// comparison (`NaN < x` is false, so ordering checks would silently pass).
fn check_finite(value: f64, which: &str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::NonFiniteValue {
            name: which.to_string(),
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The internal values used by the round-trip tests. Deliberately spans
    /// sign, magnitude across many orders, and exact zero.
    const INTERNAL_GRID: [f64; 9] = [-50.0, -1.0e6, -1.0, -0.25, 0.0, 0.25, 1.0, 1.0e6, 50.0];

    fn all_transforms() -> [Transform; 4] {
        [
            Transform::Free,
            Transform::Lower(0.0),
            Transform::Upper(10.0),
            Transform::Both(0.0, 10.0),
        ]
    }

    /// The direction that matters for seeding a fit: an initial guess must
    /// survive a trip through internal space unchanged.
    #[test]
    fn external_round_trip_is_exact() {
        let cases: [(Transform, &[f64]); 4] = [
            (Transform::Free, &[-1.0e6, -1.0, 0.0, 1.0, 1.0e6]),
            (Transform::Lower(0.0), &[0.0, 1.0e-9, 2.5, 1.0, 1.0e6]),
            (Transform::Upper(10.0), &[10.0, 9.999, 2.5, -1.0e6]),
            (
                Transform::Both(0.0, 10.0),
                &[0.0, 1.0e-9, 2.5, 5.0, 9.999, 10.0],
            ),
        ];
        for (t, values) in cases {
            for &v in values {
                let back = t.from_internal(t.to_internal(v));
                assert!(
                    (back - v).abs() <= 1e-9 * (1.0 + v.abs()),
                    "{t:?}: {v} -> {} -> {back}",
                    t.to_internal(v)
                );
            }
        }
    }

    /// The internal → external → internal trip is *not* the identity for
    /// signed `b`: the one-sided inverses are even functions, so a negative
    /// internal value folds onto its positive twin. Pin that down, because a
    /// future "fix" that made this the identity would change which point the
    /// solver explores.
    #[test]
    fn internal_round_trip_preserves_magnitude() {
        for t in [Transform::Lower(0.0), Transform::Upper(10.0)] {
            for b in INTERNAL_GRID {
                let back = t.to_internal(t.from_internal(b));
                assert!(
                    (back - b.abs()).abs() <= 1e-9 * (1.0 + b.abs()),
                    "{t:?}: {b} -> {} -> {back}, expected {}",
                    t.from_internal(b),
                    b.abs()
                );
            }
        }
    }

    /// The two-sided transform is periodic with period 2π, so the internal
    /// round trip folds into [-π/2, π/2] via `asin(sin(b))`.
    #[test]
    fn two_sided_internal_round_trip_folds_and_repeats() {
        let t = Transform::Both(0.0, 10.0);
        for b in INTERNAL_GRID {
            let back = t.to_internal(t.from_internal(b));
            let expected = b.sin().asin();
            assert!(
                (back - expected).abs() <= 1e-9 * (1.0 + expected.abs()),
                "{b} -> {back}, expected {expected}"
            );
            // Same value one full period away.
            let shifted = t.from_internal(b + 2.0 * std::f64::consts::PI);
            assert!((shifted - t.from_internal(b)).abs() < 1e-9);
        }
    }

    /// An internal value of zero lands exactly on the bound — or, for the
    /// two-sided case, exactly in the middle. This is the single cheapest
    /// check that the four formulas are wired to the right bounds.
    #[test]
    fn zero_maps_to_the_expected_edge() {
        assert_eq!(Transform::Free.from_internal(0.0), 0.0);
        assert_eq!(Transform::Lower(2.5).from_internal(0.0), 2.5);
        assert_eq!(Transform::Upper(-3.0).from_internal(0.0), -3.0);
        assert_eq!(Transform::Both(-4.0, 6.0).from_internal(0.0), 1.0);
    }

    /// Values transcribed from lmfit's own formulas, computed by hand.
    #[test]
    fn matches_lmfit_reference_values() {
        // sqrt((2.5 - 0 + 1)^2 - 1) == sqrt(11.25)
        assert!((Transform::Lower(0.0).to_internal(2.5) - 11.25_f64.sqrt()).abs() < 1e-12);
        // asin(2 * 7.5 / 10 - 1) == asin(0.5) == pi/6
        assert!(
            (Transform::Both(0.0, 10.0).to_internal(7.5) - std::f64::consts::FRAC_PI_6).abs()
                < 1e-12
        );
        // min - 1 + sqrt(0^2 + 1) == min
        assert_eq!(Transform::Lower(7.0).from_internal(0.0), 7.0);
        // max + 1 - sqrt(3^2 + 1) == 10 + 1 - sqrt(10)
        assert!(
            (Transform::Upper(10.0).from_internal(3.0) - (11.0 - 10.0_f64.sqrt())).abs() < 1e-12
        );
    }

    /// The snap's contract: an internal value comes out either exactly zero or
    /// at least [`TINY`] in magnitude — never in between. lmfit applies it to
    /// every case, unbounded parameters included.
    #[test]
    fn tiny_internal_values_are_snapped_to_zero() {
        let probes: [(Transform, &[f64]); 4] = [
            (Transform::Free, &[0.0, 1.0e-20, -1.0e-20]),
            (Transform::Lower(0.0), &[0.0, 1.0e-30, 1.0e-9]),
            (Transform::Upper(10.0), &[10.0, 10.0 - 1.0e-15, 5.0]),
            (
                Transform::Both(0.0, 10.0),
                &[5.0, 5.0 + 5.0e-16, 5.0 + 1.0e-12],
            ),
        ];
        for (t, values) in probes {
            for &v in values {
                let b = t.to_internal(v);
                assert!(
                    b == 0.0 || b.abs() >= TINY,
                    "{t:?}: to_internal({v}) = {b} fell into the snap gap"
                );
            }
        }
    }

    /// Near the two-sided midpoint the snap is load-bearing: the raw formula
    /// produces a tiny but non-zero value there. The middle assertion pins the
    /// probe itself, so that if float behaviour ever shifts the test fails
    /// loudly instead of quietly ceasing to exercise the snap.
    #[test]
    fn snap_is_load_bearing_near_the_two_sided_midpoint() {
        let t = Transform::Both(0.0, 10.0);
        assert_eq!(t.to_internal(5.0), 0.0);

        let offset: f64 = 5.0e-16;
        let raw = (2.0 * (5.0 + offset) / 10.0 - 1.0).asin();
        assert!(
            raw != 0.0 && raw.abs() < TINY,
            "probe no longer lands in the snap gap: raw = {raw}"
        );
        assert_eq!(t.to_internal(5.0 + offset), 0.0);

        // Far enough from the midpoint to clear the threshold: untouched.
        assert!(t.to_internal(5.0 + 1.0e-12).abs() > TINY);
    }

    /// Values outside the bounds are clamped before transforming, so they land
    /// exactly on the bound's internal image rather than producing a NaN.
    #[test]
    fn clamps_values_into_bounds() {
        // One-sided bounds sit at internal zero.
        assert_eq!(Transform::Lower(0.0).to_internal(-5.0), 0.0);
        assert_eq!(Transform::Upper(10.0).to_internal(99.0), 0.0);

        // The two-sided transform puts its bounds at the ends of asin's range.
        let both = Transform::Both(0.0, 10.0);
        assert_eq!(both.to_internal(-5.0), -std::f64::consts::FRAC_PI_2);
        assert_eq!(both.to_internal(99.0), std::f64::consts::FRAC_PI_2);

        // Free passes anything through, including values a bounded transform
        // would have clamped.
        assert_eq!(Transform::Free.to_internal(-5.0), -5.0);
    }

    #[test]
    fn rejects_degenerate_and_inverted_bounds() {
        assert!(matches!(
            Transform::new(Some(1.0), Some(1.0)),
            Err(Error::MinEqualsMax { .. })
        ));
        assert!(matches!(
            Transform::new(Some(5.0), Some(1.0)),
            Err(Error::InvertedBounds { .. })
        ));
        assert!(matches!(
            Transform::new(Some(f64::NAN), None),
            Err(Error::NonFiniteValue { .. })
        ));
        assert!(matches!(
            Transform::new(None, Some(f64::INFINITY)),
            Err(Error::NonFiniteValue { .. })
        ));
    }

    /// Bounded transforms must never let a value escape its bounds, whatever
    /// the solver hands back.
    #[test]
    fn from_internal_always_respects_bounds() {
        for t in all_transforms() {
            for b in INTERNAL_GRID {
                let v = t.from_internal(b);
                assert!(v.is_finite(), "{t:?} produced {v} for b = {b}");
                assert_eq!(t.clamp(v), v, "{t:?} escaped its bounds with {v}");
            }
        }
    }
}
