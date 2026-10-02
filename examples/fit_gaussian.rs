//! Fitting a Gaussian peak on a flat background.
//!
//! Run with `cargo run --example fit_gaussian`.

use lmfit::lineshapes::gaussian;
use lmfit::{Curve, Error, Model};

/// The model: one struct, and its `eval` is the arithmetic.
#[derive(Model)]
struct PeakOnBackground {
    #[param(value = 4.0)]
    amplitude: f64,
    #[param(value = 4.0)]
    center: f64,
    #[param(value = 1.0, min = 0.0)]
    sigma: f64,
    #[param(value = 0.0)]
    background: f64,
}

impl Curve for PeakOnBackground {
    fn eval(&self, x: f64) -> f64 {
        gaussian(x, self.amplitude, self.center, self.sigma) + self.background
    }
}

/// Deterministic pseudo-noise in `[-0.5, 0.5)`.
///
/// Not an RNG, so the example prints the same numbers on every run and in
/// every toolchain.
fn noise(i: usize) -> f64 {
    let h = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((h >> 33) as f64 / (1u64 << 31) as f64) - 0.5
}

fn main() -> Result<(), Error> {
    // The truth we are trying to recover.
    let (area, center, sigma, background) = (5.0, 4.5, 0.8, 0.25);

    let x: Vec<f64> = (0..201).map(|i| i as f64 * 10.0 / 200.0).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            gaussian(t, area, center, sigma) + background + 0.05 * noise(i)
        })
        .collect();

    // The struct's field values are the starting guesses.
    let model = PeakOnBackground {
        amplitude: 4.0,
        center: 4.0,
        sigma: 1.0,
        background: 0.0,
    };

    let result = model.fit(&y, &x)?;

    println!("{result}");

    println!();
    println!("true:  area={area}  centre={center}  sigma={sigma}  background={background}");
    println!(
        "fitted: area={:.4}  centre={:.4}  sigma={:.4}  background={:.4}",
        result.model.amplitude, result.model.center, result.model.sigma, result.model.background
    );

    // The fitted fields feed the same line shape back, so the components are
    // recoverable by calling it directly.
    let peak_only = gaussian(
        result.model.center,
        result.model.amplitude,
        result.model.center,
        result.model.sigma,
    );
    println!();
    println!("peak height at centre: {peak_only:.4}");
    println!("value from the model:  {:.4}", result.model.eval(result.model.center));

    Ok(())
}
