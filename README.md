# lmfit

Arbitrary curve fitting in Rust.

A model is a struct whose fields are its parameters, and the arithmetic is the
one method you write. Fitted values are the model's fields, not string lookups.

Models come in two kinds — real (`Curve`) and complex (`ComplexCurve`) — and the
Jacobian comes one of two ways: differenced automatically, one probe per
parameter, or supplied exactly through `partials_at`. Both choices are per
model, and the examples below show each of the model kinds and both Jacobian
routes.

```rust
use lmfit::{Curve, Model};

#[derive(Model)]
struct Gaussian {
    #[param(value = 5.0)]
    amp: f64,
    #[param(value = 5.0)]
    cen: f64,
    #[param(value = 2.0, min = 0.0)]
    wid: f64,
}

impl Curve for Gaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amp * (-(x - self.cen).powi(2) / self.wid).exp()
    }
}

fn main() -> Result<(), lmfit::Error> {
    let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| 5.0 * (-(t - 5.0f64).powi(2) / 2.0).exp())
        .collect();

    // The starting point is the struct's field values.
    let result = Gaussian { amp: 4.0, cen: 4.0, wid: 1.5 }.fit(&y, &x)?;

    assert!((result.model.amp - 5.0).abs() < 1e-6);
    assert!((result.model.cen - 5.0).abs() < 1e-6);
    assert!((result.model.wid - 2.0).abs() < 1e-6);

    println!("{result}");
    Ok(())
}
```

## Line shapes combine by arithmetic

Line shapes are plain functions, so a model is one struct whose `eval` does the
arithmetic — a peak plus a background is just a sum of two calls:

```rust
use lmfit::lineshapes::gaussian;
use lmfit::{Curve, Model};

#[derive(Model)]
struct PeakOnBackground {
    #[param(value = 4.0)]
    amplitude: f64,
    #[param(value = 4.0)]
    center: f64,
    #[param(value = 1.5, min = 0.0)]
    sigma: f64,
    #[param(value = 0.0)]
    background: f64,
}

impl Curve for PeakOnBackground {
    fn eval(&self, x: f64) -> f64 {
        gaussian(x, self.amplitude, self.center, self.sigma) + self.background
    }
}

fn main() -> Result<(), lmfit::Error> {
    let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| {
            5.0 / (std::f64::consts::TAU.sqrt() * 2.0)
                * (-(t - 5.0f64).powi(2) / 8.0).exp()
                + 0.75
        })
        .collect();

    let result = PeakOnBackground { amplitude: 4.0, center: 4.0, sigma: 1.5, background: 0.0 }
        .fit(&y, &x)?;

    println!("area       = {}", result.model.amplitude);
    println!("background = {}", result.model.background);
    Ok(())
}
```

## Bounds and fixed parameters

`#[param(min = .., max = ..)]` constrains a parameter and `vary = false` pins it.

```rust
use lmfit::Model;

#[derive(Model)]
struct Bounded {
    #[param(value = 5.0, min = 0.0, max = 10.0)]
    amplitude: f64,
    #[param(value = 1.0, vary = false)]
    offset: f64,
}
```

Bounds are enforced *by construction*: the fit runs in an unbounded internal
space and the transform cannot produce a value outside the range, so nothing is
clamped or rejected during iteration. This is lmfit's own scheme, transcribed —
including its `to_internal`/`from_internal` formulas and its snap of near-zero
internal values onto the bound.

## Results

`ModelResult<M>` carries the fitted model, the fitted parameters with their
bounds and starting points, the curve, the residuals, and the usual statistics.

| Field | Meaning |
| --- | --- |
| `model` | the fitted model — its fields are the parameter values |
| `params` | the fitted `Parameter`s, with bounds and starting values |
| `best_fit` | the model evaluated at `x` |
| `residual` | `y - best_fit` |
| `chisqr`, `redchi` | sum of squared residuals, and it divided by `nfree` |
| `aic`, `bic` | Akaike and Bayesian information criteria |
| `ndata`, `nvarys`, `nfree` | data points, varied parameters, degrees of freedom |
| `nfev` | residual evaluations performed |
| `success`, `message` | whether the solver converged, and how it ended |

```rust
use lmfit::{Curve, Model};

#[derive(Model)]
struct Gaussian {
    #[param(value = 5.0)]
    amp: f64,
    #[param(value = 5.0)]
    cen: f64,
    #[param(value = 2.0, min = 0.0)]
    wid: f64,
}

impl Curve for Gaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amp * (-(x - self.cen).powi(2) / self.wid).exp()
    }
}

fn main() -> Result<(), lmfit::Error> {
    let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| 5.0 * (-(t - 5.0f64).powi(2) / 2.0).exp())
        .collect();
    let result = Gaussian::default().fit(&y, &x)?;

    // Fitted values are the model's fields.
    println!("amplitude = {}", result.model.amplitude);
    println!("chi-square = {}, reduced = {}", result.chisqr, result.redchi);
    println!("converged: {} ({})", result.success, result.message);

    println!("{result}");
    Ok(())
}
```

## Line shapes

`lmfit::lineshapes` provides `gaussian` and `constant` as plain functions.
Note that `gaussian`'s `amplitude` is the **area** under the curve, not the
peak height — that is lmfit's convention and the opposite of the
`amp * exp(..)` form most people write from memory.

## Complex models

A complex-valued model implements `ComplexCurve` instead of `Curve`: `eval`
takes and returns `Complex64`, and the parameters stay real. Each data point
contributes its real and imaginary part as two least-squares slots, so a fit
over `n` complex points counts `2n` data points.

```rust
use lmfit::{Complex64, ComplexCurve, Model};

#[derive(Model)]
struct ComplexGaussian {
    #[param(value = 5.0)]
    amplitude: f64,
    #[param(value = 5.0)]
    center: f64,
    #[param(value = 2.0, min = 0.0)]
    sigma: f64,
    #[param(value = 0.3)]
    phase: f64,
}

impl ComplexCurve for ComplexGaussian {
    fn eval(&self, x: Complex64) -> Complex64 {
        let e = (-(x - self.center).powi(2) / (2.0 * self.sigma.powi(2))).exp();
        e * self.amplitude * Complex64::from_polar(1.0, self.phase)
    }
}

fn main() -> Result<(), lmfit::Error> {
    let x: Vec<f64> = (0..101).map(|i| i as f64 / 10.0).collect();
    let y: Vec<Complex64> = x
        .iter()
        .map(|&t| {
            let g = 5.0 * (-(t - 5.0f64).powi(2) / 8.0).exp();
            Complex64::from_polar(g, 0.3)
        })
        .collect();

    let result =
        ComplexGaussian { amplitude: 4.0, center: 4.0, sigma: 1.5, phase: 0.0 }.fit(&y, &x)?;

    assert!((result.model.amplitude - 5.0).abs() < 1e-6);
    assert!((result.model.phase - 0.3).abs() < 1e-6);

    println!("{result}");
    Ok(())
}
```

## Jacobians: automatic or analytic

By default the Jacobian is differenced for you — one probe per parameter, the
way MINPACK's `fdjac2` does it, so the examples above need no derivative code
at all. A model can instead hand the solver exact partial derivatives. Write
`partials_at` next to `eval`, returning the `{Model}Partials` bundle that
`#[derive(Model)]` generates (fields named after the model's own):

```rust
use lmfit::{Curve, Model, PartialValues};

impl Curve for Gaussian {
    fn eval(&self, x: f64) -> f64 {
        self.amp * (-(x - self.cen).powi(2) / self.wid).exp()
    }

    fn partials_at(&self, x: f64) -> Option<impl PartialValues<Scalar = f64>> {
        let d = x - self.cen;
        let e = (-d * d / self.wid).exp();
        Some(GaussianPartials {
            amp: e,                                             // ∂f/∂amp
            cen: 2.0 * self.amp * e * d / self.wid,             // ∂f/∂cen
            wid: self.amp * e * d * d / (self.wid * self.wid),  // ∂f/∂wid
        })
    }
}
```

Omitting the method keeps a fit on finite differences, unchanged. With it, the
solver spends no residual evaluations on Jacobian probes — the same fit needs
far fewer.

## Implementation

The solver is [Levenberg-Marquardt](https://crates.io/crates/levenberg-marquardt),
driven with scipy `leastsq`'s default tolerances and lmfit's function-evaluation
budget. Unless a model supplies `partials_at`, the Jacobian is differenced the
way MINPACK's `fdjac2` does it — the same path scipy's `leastsq` takes — so a
fit without analytic derivatives lands where the Python implementation lands,
and one with them skips the probes entirely.

The crate is layered so the pieces can be replaced independently: the derive
macro only ever produces a parameter *layout*, the solver only ever consumes
one, and neither knows about the other.

| Module | Role |
| --- | --- |
| `traits` | `ModelParams` (layout) and `Curve`/`ComplexCurve` (arithmetic) — the seam |
| `bounds` | lmfit's bounded-parameter transform |
| `solver` | the LM adapter; the only module that knows the solver exists |
| `numerics` | finite-difference Jacobian |
| `result` | `ModelResult` and the fit statistics |
| `render` | the `Display` fit report |
| `lineshapes` | ready-made line shapes as plain functions |

## Status

Not yet implemented, in rough order of how much they are missed:

- Parameter expressions (lmfit's `expr=`), and `guess()`-style heuristics for
  picking starting values.
- Solvers other than `leastsq`, and global or derivative-free methods.

## Minimum supported Rust version

1.87, which is what `nalgebra` 0.34 requires. The crate uses edition 2024.

## License

MIT — see [LICENSE](LICENSE).
