//! Composite models.
//!
//! `Gaussian::default() + Constant::default()` builds a [`Sum`] and, thanks to
//! the `Add` impl below, reads the way it does in lmfit. Any depth of nesting
//! works, because [`Sum`] is itself a [`Curve`].
//!
//! Two things have to be true of a composite, and they are handled by
//! different halves of the machinery:
//!
//! * Evaluation is trivial — a sum evaluates both children and adds them.
//! * Parameter layout is not — the two children's parameter tables must be
//!   concatenated into one index space, and each child's names prefix-qualified
//!   so that `Gaussian + Constant` does not present an ambiguous `amp`.
//!
//! Only the second is subtle, and it is what most of this module is about.

use std::ops::Add;

use crate::traits::{Curve, ModelParams, ParamSpec};

/// Two models added together: `eval` returns the sum of both.
#[derive(Debug, Clone, PartialEq)]
pub struct Sum<A, B> {
    /// Left-hand model.
    pub a: A,
    /// Right-hand model.
    pub b: B,
}

impl<A, B> Sum<A, B> {
    /// Combine two models.
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Curve, B: Curve> Curve for Sum<A, B> {
    fn eval(&self, x: f64) -> f64 {
        self.a.eval(x) + self.b.eval(x)
    }
}

impl<A: ModelParams, B: ModelParams> ModelParams for Sum<A, B> {
    /// Nested composites take the literal name `sum`, so `(A + B) + C` yields
    /// parameter names like `sum_alpha_a`. Distinct, if not pretty — an
    /// explicit prefix override is left for a later version.
    const MODEL_NAME: &'static str = "sum";

    fn specs(&self) -> Vec<ParamSpec> {
        let mut out = prefixed(self.a.specs(), A::MODEL_NAME);
        out.extend(prefixed(self.b.specs(), B::MODEL_NAME));
        out
    }

    fn describe(&self) -> String {
        format!("({} + {})", self.a.describe(), self.b.describe())
    }

    fn nparams(&self) -> usize {
        self.a.nparams() + self.b.nparams()
    }

    fn get(&self, index: usize) -> f64 {
        let split = self.a.nparams();
        if index < split {
            self.a.get(index)
        } else {
            self.b.get(index - split)
        }
    }

    fn set(&mut self, index: usize, value: f64) {
        let split = self.a.nparams();
        if index < split {
            self.a.set(index, value);
        } else {
            self.b.set(index - split, value);
        }
    }

    fn with_values(&self, values: &[f64]) -> Self {
        let split = self.a.nparams();
        Self {
            a: self.a.with_values(&values[..split]),
            b: self.b.with_values(&values[split..]),
        }
    }
}

/// Rewrite every spec's name to `{prefix}_{name}`.
fn prefixed(specs: Vec<ParamSpec>, prefix: &str) -> Vec<ParamSpec> {
    specs
        .into_iter()
        .map(|mut spec| {
            spec.name = format!("{prefix}_{}", spec.name);
            spec
        })
        .collect()
}

/// Combine two models with `+`.
///
/// # Why the leaf `Add` impls live elsewhere
///
/// The obvious spelling, `impl<A: Curve, B: Curve> Add<B> for A`, does not
/// compile: `A` is a bare type parameter rather than a local type, which
/// violates the orphan rule (`E0210`).
///
/// The fix is to put each impl where its `Self` type *is* local. `Sum` is
/// local to this crate, so the impl below covers it. A leaf model is local to
/// the crate that declares it, so `#[derive(Model)]` emits
/// `impl<B: Curve> Add<B> for ThatModel` — legal there by the same rule, which
/// permits `impl<T> ForeignTrait<T> for LocalType`.
///
/// The upshot is that `gaussian + constant + offset` works with real operator
/// syntax at any depth. The cost is that `Add` is generated per model rather
/// than written once, so a hand-written model must supply its own.
///
/// Nesting is left-leaning: `a + b + c` parses as `(a + b) + c`.
impl<A: Curve, B: Curve, C: Curve> Add<C> for Sum<A, B> {
    type Output = Sum<Sum<A, B>, C>;

    fn add(self, rhs: C) -> Self::Output {
        Sum::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stand-in for what `#[derive(Model)]` will generate in M4: a leaf model
    /// with a settable name so prefixing can be exercised directly.
    #[derive(Debug, Clone, PartialEq)]
    struct Leaf {
        name: &'static str,
        values: Vec<f64>,
        labels: &'static [&'static str],
    }

    impl Leaf {
        fn new(name: &'static str, labels: &'static [&'static str], values: &[f64]) -> Self {
            Self {
                name,
                values: values.to_vec(),
                labels,
            }
        }
    }

    impl ModelParams for Leaf {
        const MODEL_NAME: &'static str = "leaf";

        fn specs(&self) -> Vec<ParamSpec> {
            self.labels
                .iter()
                .zip(&self.values)
                .map(|(label, value)| ParamSpec {
                    name: (*label).to_string(),
                    value: *value,
                    min: None,
                    max: None,
                    vary: true,
                })
                .collect()
        }

        fn nparams(&self) -> usize {
            self.values.len()
        }

        fn get(&self, index: usize) -> f64 {
            self.values[index]
        }

        fn set(&mut self, index: usize, value: f64) {
            self.values[index] = value;
        }

        fn with_values(&self, values: &[f64]) -> Self {
            Self {
                name: self.name,
                values: values.to_vec(),
                labels: self.labels,
            }
        }
    }

    impl Curve for Leaf {
        fn eval(&self, x: f64) -> f64 {
            // Not meaningful — this leaf exists to test layout, not arithmetic.
            self.values.iter().sum::<f64>() + x
        }
    }

    /// Two leaves with distinct `MODEL_NAME`s, to test prefixing.
    #[derive(Debug, Clone, PartialEq)]
    struct Alpha(Leaf);
    #[derive(Debug, Clone, PartialEq)]
    struct Beta(Leaf);

    macro_rules! delegate {
        ($t:ty, $name:literal) => {
            impl ModelParams for $t {
                const MODEL_NAME: &'static str = $name;
                fn specs(&self) -> Vec<ParamSpec> {
                    self.0.specs()
                }
                fn nparams(&self) -> usize {
                    self.0.nparams()
                }
                fn get(&self, index: usize) -> f64 {
                    self.0.get(index)
                }
                fn set(&mut self, index: usize, value: f64) {
                    self.0.set(index, value);
                }
                fn with_values(&self, values: &[f64]) -> Self {
                    Self(self.0.with_values(values))
                }
            }
            impl Curve for $t {
                fn eval(&self, x: f64) -> f64 {
                    self.0.eval(x)
                }
            }
            // Exactly what `#[derive(Model)]` will emit for a leaf model: the
            // impl is legal here because `$t` is local to this module.
            impl<B: Curve> std::ops::Add<B> for $t {
                type Output = Sum<$t, B>;
                fn add(self, rhs: B) -> Self::Output {
                    Sum::new(self, rhs)
                }
            }
        };
    }

    delegate!(Alpha, "alpha");
    delegate!(Beta, "beta");

    fn alpha() -> Alpha {
        Alpha(Leaf::new("a", &["x", "y"], &[1.0, 2.0]))
    }

    fn beta() -> Beta {
        Beta(Leaf::new("b", &["z"], &[3.0]))
    }

    #[test]
    fn add_operator_builds_a_sum() {
        let model = alpha() + beta();
        assert_eq!(model.a, alpha());
        assert_eq!(model.b, beta());
        assert_eq!(model.eval(0.0), 1.0 + 2.0 + 3.0);
    }

    #[test]
    fn nparams_and_values_concatenate() {
        let model = alpha() + beta();
        assert_eq!(model.nparams(), 3);

        let values = vec![10.0, 20.0, 30.0];
        let rebuilt = model.with_values(&values);
        assert_eq!(rebuilt.nparams(), 3);
        assert_eq!(rebuilt.get(0), 10.0);
        assert_eq!(rebuilt.get(1), 20.0);
        assert_eq!(rebuilt.get(2), 30.0);
        // The split must land between the two children, not inside either.
        assert_eq!(rebuilt.a.0.values, vec![10.0, 20.0]);
        assert_eq!(rebuilt.b.0.values, vec![30.0]);
    }

    #[test]
    fn set_dispatches_by_offset() {
        let mut model = alpha() + beta();
        model.set(0, 100.0);
        model.set(2, 300.0);
        assert_eq!(model.a.0.values, vec![100.0, 2.0]);
        assert_eq!(model.b.0.values, vec![300.0]);
    }

    #[test]
    fn specs_are_prefix_qualified() {
        let model = alpha() + beta();
        let names: Vec<String> = model.specs().into_iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["alpha_x", "alpha_y", "beta_z"]);
    }

    #[test]
    fn nesting_flattens_into_one_index_space() {
        // (alpha + beta) + alpha-again would collide, so use a third name.
        let model = (alpha() + beta()) + Alpha(Leaf::new("c", &["w"], &[4.0]));

        assert_eq!(model.nparams(), 4);
        let rebuilt = model.with_values(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rebuilt.get(0), 1.0);
        assert_eq!(rebuilt.get(3), 4.0);
        assert_eq!(rebuilt.eval(0.0), 1.0 + 2.0 + 3.0 + 4.0);

        // The inner composite keeps its own prefix, and the outer one adds
        // another — distinct names are what matter, not their prettiness.
        let names: Vec<String> = model.specs().into_iter().map(|s| s.name).collect();
        assert_eq!(
            names,
            vec!["sum_alpha_x", "sum_alpha_y", "sum_beta_z", "alpha_w"]
        );
    }

    /// Combining two models of the same type produces colliding parameter
    /// names. That is reported rather than silently allowed, mirroring lmfit,
    /// which likewise asks for explicit prefixes in this situation.
    #[test]
    fn identical_models_collide_and_are_reported() {
        let model = alpha() + alpha();
        let err = model.parameters().unwrap_err();
        assert!(
            matches!(err, crate::Error::DuplicateParameter { ref name } if name == "alpha_x"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn distinct_models_produce_a_usable_parameter_set() {
        let model = alpha() + beta();
        let params = model.parameters().unwrap();
        assert_eq!(params.len(), 3);
        assert_eq!(params.get("alpha_x").unwrap().value, 1.0);
        assert_eq!(params.get("beta_z").unwrap().value, 3.0);
        assert_eq!(params.varied_indices(), vec![0, 1, 2]);
    }
}
