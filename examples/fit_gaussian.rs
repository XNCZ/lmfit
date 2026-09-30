//! Fitting a Gaussian peak on a flat background.
//!
//! Run with `cargo run --example fit_gaussian`.

use lmfit::models::{Constant, Gaussian};
use lmfit::{Curve, Error};

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
            let peak = area / (std::f64::consts::TAU.sqrt() * sigma)
                * (-(t - center).powi(2) / (2.0 * sigma.powi(2))).exp();
            peak + background + 0.05 * noise(i)
        })
        .collect();

    // A peak plus a flat offset — note that this is simply `+`.
    let model = Gaussian {
        amplitude: 4.0,
        center: 4.0,
        sigma: 1.0,
    } + Constant { c: 0.0 };

    let result = model.fit(&y, &x)?;

    println!("{}", result.fit_report());

    println!();
    println!("true:  area={area}  centre={center}  sigma={sigma}  background={background}");
    println!(
        "fitted: area={:.4}  centre={:.4}  sigma={:.4}  background={:.4}",
        result.model.a.amplitude, result.model.a.center, result.model.a.sigma, result.model.b.c
    );

    // The components can be evaluated separately, which is the point of
    // keeping a composite as two models rather than one merged function.
    let peak_only = result.model.a.eval(result.model.a.center);
    println!();
    println!("peak height at centre: {peak_only:.4}");
    println!(
        "value from the composite: {:.4}",
        result.model.eval(result.model.a.center)
    );

    Ok(())
}
