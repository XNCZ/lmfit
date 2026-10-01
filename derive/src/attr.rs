//! Parsing for the `#[param(...)]` and `#[model(...)]` attributes.

use syn::{Attribute, Expr, Result};

/// The contents of a field's `#[param(...)]` attribute.
#[derive(Default)]
pub struct ParamAttrs {
    /// Starting value. Required — without it the generated `Default` impl
    /// would have nothing to put in the field.
    pub value: Option<Expr>,
    /// Lower bound.
    pub min: Option<Expr>,
    /// Upper bound.
    pub max: Option<Expr>,
    /// Whether the solver may vary this field.
    pub vary: Option<Expr>,
}

/// Parse a field's `#[param(...)]`, rejecting anything unrecognised.
///
/// Unknown keys are an error rather than silently ignored: `#[param(mim = 0.0)]`
/// would otherwise quietly produce an unbounded parameter, and the resulting
/// fit would look perfectly healthy while exploring a region the user meant to
/// exclude.
pub fn parse_param(attr: &Attribute) -> Result<ParamAttrs> {
    let mut out = ParamAttrs::default();

    attr.parse_nested_meta(|meta| {
        let tgt = if meta.path.is_ident("value") {
            &mut out.value
        } else if meta.path.is_ident("min") {
            &mut out.min
        } else if meta.path.is_ident("max") {
            &mut out.max
        } else if meta.path.is_ident("vary") {
            &mut out.vary
        } else {
            return Err(
                meta.error("unknown `param` key; expected one of `value`, `min`, `max`, `vary`")
            );
        };

        if tgt.is_some() {
            return Err(meta.error("duplicate `param` key"));
        }
        *tgt = Some(meta.value()?.parse()?);
        Ok(())
    })?;

    Ok(out)
}

/// Parse the struct-level `#[model(name = "...")]` override, if present.
///
/// Returns `None` when no such attribute is present, and an error when one is
/// present but malformed.
pub fn parse_model_name(attrs: &[Attribute]) -> Result<Option<String>> {
    let mut model_name = None;

    for attr in attrs {
        if !attr.path().is_ident("model") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if !meta.path.is_ident("name") {
                return Err(meta.error("unknown `model` key; expected `name`"));
            }
            if model_name.is_some() {
                return Err(meta.error("duplicate `model(name = ..)`"));
            }
            let lit: syn::LitStr = meta.value()?.parse()?;
            model_name = Some(lit.value());
            Ok(())
        })?;
    }

    Ok(model_name)
}

/// Turn a type name into the name a model reports under.
///
/// `Gaussian` becomes `gaussian` and `BoundedGaussian` becomes
/// `bounded_gaussian`, so a fit report reads a model's name the way a person
/// would say it. Runs of capitals (as in `XPSModel`) get a separator before
/// each one; that is rarely what anyone wants, which is what the
/// `#[model(name = "...")]` override is for.
pub fn to_snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::to_snake_case;

    #[test]
    fn converts_type_names_to_prefixes() {
        assert_eq!(to_snake_case("Gaussian"), "gaussian");
        assert_eq!(to_snake_case("BoundedGaussian"), "bounded_gaussian");
        assert_eq!(to_snake_case("Line"), "line");
        assert_eq!(to_snake_case("Constant"), "constant");
    }
}
