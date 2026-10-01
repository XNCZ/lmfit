//! Named fit parameters.
//!
//! A [`Parameter`] carries everything the solver needs to know about one
//! quantity: where it starts, how far it may move, and whether it moves at
//! all. [`Parameters`] is the ordered collection a fit reports over.
//!
//! Reading a fitted value is deliberately *not* done through this collection.
//! A model's parameters are its struct fields, so `result.model.amplitude` is
//! the idiomatic access path; this module exists for the fit report, for
//! runtime overrides, and as the seam the solver iterates over. Accordingly
//! there is no `Index` impl — looking up a parameter that does not exist is
//! an ordinary mistake, and it returns `None` rather than panicking.

use crate::bounds::Transform;
use crate::error::{Error, Result};

/// One quantity to be fitted.
///
/// Construct with [`Parameter::new`] and refine with the builder methods:
///
/// ```
/// # use lmfit::Parameter;
/// let p = Parameter::new("sigma", 2.0).min(0.0);
/// assert_eq!(p.value, 2.0);
/// assert_eq!(p.min, Some(0.0));
/// assert!(p.vary);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    /// Name as it appears in the fit report.
    pub name: String,
    /// Current value.
    pub value: f64,
    /// Lower bound, if any.
    pub min: Option<f64>,
    /// Upper bound, if any.
    pub max: Option<f64>,
    /// Whether the solver may vary this parameter. `false` pins it.
    pub vary: bool,
    /// The value the fit started from, preserved for the `(init = ...)` field
    /// of the fit report.
    pub init: f64,
}

impl Parameter {
    /// A free, varied parameter starting at `value`.
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            min: None,
            max: None,
            vary: true,
            init: value,
        }
    }

    /// Set the lower bound.
    #[must_use]
    pub fn min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }

    /// Set the upper bound.
    #[must_use]
    pub fn max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    /// Set both bounds at once.
    #[must_use]
    pub fn bounds(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    /// Pin this parameter: it keeps its value and the solver ignores it.
    #[must_use]
    pub fn fix(mut self) -> Self {
        self.vary = false;
        self
    }

    /// The transform mapping this parameter's bounds into solver space.
    ///
    /// # Errors
    ///
    /// Surfaces the bound validation from [`Transform::new`], with this
    /// parameter's name attached.
    pub fn transform(&self) -> Result<Transform> {
        Transform::new(self.min, self.max).map_err(|e| self.name_error(e))
    }

    /// Set the value, clamping it into the bounds.
    ///
    /// Clamping mirrors lmfit, which keeps every parameter inside its bounds
    /// at all times. A non-finite value is rejected rather than clamped,
    /// because silently turning a NaN into a bound would hide a bug upstream.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NonFiniteValue`] if `value` is NaN or infinite.
    pub fn set_value(&mut self, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::NonFiniteValue {
                name: self.name.clone(),
                value,
            });
        }
        let transform = self.transform()?;
        self.value = transform.clamp(value);
        Ok(())
    }

    /// Replace any placeholder name inside a bound-validation error with this
    /// parameter's actual name, so the message points at the right parameter.
    fn name_error(&self, err: Error) -> Error {
        match err {
            Error::MinEqualsMax { min, .. } => Error::MinEqualsMax {
                name: self.name.clone(),
                min,
            },
            Error::InvertedBounds { min, max, .. } => Error::InvertedBounds {
                name: self.name.clone(),
                min,
                max,
            },
            other => other,
        }
    }
}

/// An ordered collection of [`Parameter`]s, in the order they were added.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Parameters {
    items: Vec<Parameter>,
}

impl Parameters {
    /// An empty collection.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Append a parameter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DuplicateParameter`] if the name is already taken. A
    /// derived model cannot trip this — its field names are unique by
    /// construction — but a hand-written impl that emits the same name twice
    /// is a bug worth reporting rather than silently merging.
    pub fn push(&mut self, parameter: Parameter) -> Result<()> {
        if self.get(&parameter.name).is_some() {
            return Err(Error::DuplicateParameter {
                name: parameter.name,
            });
        }
        self.items.push(parameter);
        Ok(())
    }

    /// Look up a parameter by name.
    pub fn get(&self, name: &str) -> Option<&Parameter> {
        self.items.iter().find(|p| p.name == name)
    }

    /// Look up a parameter by name, mutably.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Parameter> {
        self.items.iter_mut().find(|p| p.name == name)
    }

    /// Set a parameter's value by name, clamping it into the bounds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownParameter`] if no parameter has that name — a
    /// misspelled name in a runtime override is a bug worth reporting, not
    /// something to ignore. Also surfaces [`Parameter::set_value`]'s errors.
    pub fn set_value(&mut self, name: &str, value: f64) -> Result<()> {
        match self.get_mut(name) {
            Some(p) => p.set_value(value),
            None => Err(Error::UnknownParameter {
                name: name.to_string(),
            }),
        }
    }

    /// Iterate over the parameters in insertion order.
    pub fn iter(&self) -> std::slice::Iter<'_, Parameter> {
        self.items.iter()
    }

    /// Iterate mutably over the parameters in insertion order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Parameter> {
        self.items.iter_mut()
    }

    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Indices of the parameters the solver is allowed to vary.
    pub fn no_fix_indices(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, p)| p.vary)
            .map(|(i, _)| i)
            .collect()
    }

    /// The current values, in insertion order.
    pub fn values(&self) -> Vec<f64> {
        self.items.iter().map(|p| p.value).collect()
    }
}

impl<'a> IntoIterator for &'a Parameters {
    type Item = &'a Parameter;
    type IntoIter = std::slice::Iter<'a, Parameter>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_sets_every_field() {
        let p = Parameter::new("amp", 5.0).bounds(0.0, 10.0);
        assert_eq!(p.name, "amp");
        assert_eq!(p.value, 5.0);
        assert_eq!(p.init, 5.0);
        assert_eq!(p.min, Some(0.0));
        assert_eq!(p.max, Some(10.0));
        assert!(p.vary);

        let fixed = Parameter::new("c", 1.0).fix();
        assert!(!fixed.vary);
    }

    #[test]
    fn rejects_duplicate_names() {
        let mut params = Parameters::new();
        params.push(Parameter::new("amp", 1.0)).unwrap();
        let err = params.push(Parameter::new("amp", 2.0)).unwrap_err();
        assert!(matches!(err, Error::DuplicateParameter { .. }));
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn set_value_clamps_into_bounds() {
        let mut p = Parameter::new("sigma", 2.0).min(0.0);
        p.set_value(-5.0).unwrap();
        assert_eq!(p.value, 0.0);
        p.set_value(3.0).unwrap();
        assert_eq!(p.value, 3.0);

        let mut q = Parameter::new("c", 5.0).max(10.0);
        q.set_value(99.0).unwrap();
        assert_eq!(q.value, 10.0);
    }

    #[test]
    fn set_value_rejects_non_finite() {
        let mut p = Parameter::new("amp", 1.0);
        assert!(matches!(
            p.set_value(f64::NAN),
            Err(Error::NonFiniteValue { .. })
        ));
        assert!(matches!(
            p.set_value(f64::INFINITY),
            Err(Error::NonFiniteValue { .. })
        ));
        // The failed writes must not have disturbed the value.
        assert_eq!(p.value, 1.0);
    }

    #[test]
    fn varied_indices_skips_fixed_parameters() {
        let mut params = Parameters::new();
        params.push(Parameter::new("a", 1.0)).unwrap();
        params.push(Parameter::new("b", 1.0).fix()).unwrap();
        params.push(Parameter::new("c", 1.0)).unwrap();
        assert_eq!(params.no_fix_indices(), vec![0, 2]);
    }

    /// A bound error must name the parameter it came from, so a fit over a
    /// dozen parameters points at the offending one.
    #[test]
    fn transform_errors_carry_the_parameter_name() {
        let p = Parameter::new("sigma", 1.0).bounds(2.0, 2.0);
        match p.transform().unwrap_err() {
            Error::MinEqualsMax { name, .. } => assert_eq!(name, "sigma"),
            other => panic!("expected MinEqualsMax, got {other:?}"),
        }

        let q = Parameter::new("width", 1.0).bounds(9.0, 1.0);
        match q.transform().unwrap_err() {
            Error::InvertedBounds { name, .. } => assert_eq!(name, "width"),
            other => panic!("expected InvertedBounds, got {other:?}"),
        }
    }

    #[test]
    fn set_value_by_name_reports_unknown_names() {
        let mut params = Parameters::new();
        params.push(Parameter::new("amp", 1.0)).unwrap();
        params.set_value("amp", 4.0).unwrap();
        assert_eq!(params.get("amp").unwrap().value, 4.0);
        assert!(params.set_value("nope", 1.0).is_err());
    }
}
