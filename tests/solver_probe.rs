//! Smoke test that the `levenberg-marquardt` crate integrates as expected.
//!
//! This exists mainly to pin down the trait plumbing — `LeastSquaresProblem`
//! with dynamic dimensions and owned storage — so that a change in the
//! solver's API surfaces here first, rather than in `solver.rs`.

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};

// The solver does *not* re-export nalgebra, so we depend on it directly. The
// version is pinned to the one the solver pulls in transitively; `cargo tree -d`
// must stay free of a second copy.
use nalgebra::{DMatrix, DVector, Dyn, Matrix, Vector, storage::Owned};

/// Fit `y = a * x` through two points, with one parameter `a`.
struct Scale {
    a: f64,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for Scale {
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, p: &Vector<f64, Dyn, Self::ParameterStorage>) {
        self.a = p[0];
    }

    fn params(&self) -> Vector<f64, Dyn, Self::ParameterStorage> {
        DVector::from_vec(vec![self.a])
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(DVector::from_vec(
            self.x
                .iter()
                .zip(&self.y)
                .map(|(x, y)| y - self.a * x)
                .collect(),
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, Dyn, Self::JacobianStorage>> {
        // d(residual_i)/da = -x_i
        Some(DMatrix::from_vec(2, 1, self.x.iter().map(|x| -x).collect()))
    }
}

#[test]
fn converges_on_a_trivial_linear_problem() {
    let problem = Scale {
        a: 0.5,
        x: vec![2.0, 4.0],
        y: vec![4.0, 8.0],
    };

    // The scalar type must be named explicitly: `new()` is defined on
    // `impl<F: RealField + Float>`, which is not enough to infer `f64`.
    //
    // The tolerance is set explicitly rather than relying on `new()`'s default.
    // Without the `minpack-compat` feature the default is `eps * 30`, whereas
    // scipy's `leastsq` — and therefore lmfit — uses MINPACK's 1.49012e-8 with
    // `gtol` disabled. `with_tol` sets `ftol = xtol = tol` and `gtol = 0`,
    // which is exactly that combination.
    let (fitted, report) = LevenbergMarquardt::<f64>::new()
        .with_tol(1.49012e-8)
        .with_patience(2000)
        .minimize(problem);

    assert!(
        report.termination.was_successful(),
        "expected success, got {:?}",
        report.termination
    );
    // y = 2x exactly, so the recovered slope must be 2.
    assert!(
        (fitted.a - 2.0).abs() < 1e-10,
        "expected a = 2, got {}",
        fitted.a
    );
}
