//! Code generation for `#[derive(Model)]`.
//!
//! Given a struct whose fields are all `f64` and all carry `#[param(...)]`,
//! this emits three things:
//!
//! 1. `impl ModelParams` — the parameter layout the solver reads.
//! 2. `impl Default` — built from each field's `value`, so `Gaussian::default()`
//!    is the model at its starting guesses.
//! 3. Nothing else. Notably *not* `Curve`: the arithmetic is the user's to
//!    write, and that split is what keeps this macro out of their way.
//!
//! Generated paths go through `::lmfit::`, which resolves both in a
//! downstream crate and inside this crate itself thanks to the
//! `extern crate self as lmfit;` in the library root.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Result, spanned::Spanned};

use crate::attr::{parse_model_name, parse_param, to_snake_case};

/// One parsed field: the struct field plus its `#[param(...)]` settings.
struct Field {
    ident: syn::Ident,
    value: syn::Expr,
    min: Option<syn::Expr>,
    max: Option<syn::Expr>,
    vary: Option<syn::Expr>,
}

pub fn codegen(input: DeriveInput) -> Result<TokenStream> {
    let name = &input.ident;

    if !input.generics.params.is_empty() {
        return Err(syn::Error::new(
            input.generics.span(),
            "`#[derive(Model)]` does not support generic models",
        ));
    }

    let fields = collect_fields(&input)?;
    if fields.is_empty() {
        return Err(syn::Error::new(
            input.span(),
            "`#[derive(Model)]` needs at least one parameter field",
        ));
    }

    let model_name = match parse_model_name(&input.attrs)? {
        Some(explicit) => explicit,
        None => to_snake_case(&name.to_string()),
    };

    let nparams = fields.len();

    // specs(): one entry per field, in declaration order — which is also the
    // index order `get`, `set`, and `at_values` use.
    //
    // `value` reads the field's *current* value, not the `#[param(value = ..)]`
    // starting value: a spec describes where the model is now, and the fit
    // seeds itself from exactly that.
    let spec_entries = fields.iter().map(|f| {
        let field = &f.ident;
        let field_name = field.to_string();
        let min = option(f.min.as_ref());
        let max = option(f.max.as_ref());
        let vary = match &f.vary {
            Some(expr) => quote!(#expr),
            None => quote!(true),
        };
        quote! {
            ::lmfit::ParamSpec {
                name: ::std::string::String::from(#field_name),
                value: self.#field,
                min: #min,
                max: #max,
                vary: #vary,
            }
        }
    });

    let get_arms = fields.iter().enumerate().map(|(i, f)| {
        let field = &f.ident;
        quote!(#i => self.#field,)
    });

    let set_arms = fields.iter().enumerate().map(|(i, f)| {
        let field = &f.ident;
        quote!(#i => self.#field = value,)
    });

    let value_fields = fields.iter().enumerate().map(|(i, f)| {
        let field = &f.ident;
        quote!(#field: values[#i],)
    });

    let default_fields = fields.iter().map(|f| {
        let field = &f.ident;
        let value = &f.value;
        quote!(#field: #value,)
    });

    Ok(quote! {
        impl ::lmfit::ModelParams for #name {
            const MODEL_NAME: &'static str = #model_name;

            const NPARAMS: usize = #nparams;

            fn specs(&self) -> ::std::vec::Vec<::lmfit::ParamSpec> {
                ::std::vec![#(#spec_entries),*]
            }

            fn get(&self, index: usize) -> f64 {
                match index {
                    #(#get_arms)*
                    _ => ::std::panic!(
                        "parameter index {} is out of range for model `{}`, which has {} parameters",
                        index,
                        #model_name,
                        #nparams,
                    ),
                }
            }

            fn set(&mut self, index: usize, value: f64) {
                match index {
                    #(#set_arms)*
                    _ => ::std::panic!(
                        "parameter index {} is out of range for model `{}`, which has {} parameters",
                        index,
                        #model_name,
                        #nparams,
                    ),
                }
            }

            fn at_values(&self, values: &[f64]) -> Self {
                ::std::assert!(
                    values.len() == #nparams,
                    "model `{}` expects {} parameter values, got {}",
                    #model_name,
                    #nparams,
                    values.len(),
                );
                Self {
                    #(#value_fields)*
                }
            }
        }

        impl ::std::default::Default for #name {
            fn default() -> Self {
                Self {
                    #(#default_fields)*
                }
            }
        }
    })
}

/// Validate the input and collect one [`Field`] per struct field.
fn collect_fields(input: &DeriveInput) -> Result<Vec<Field>> {
    let data = match &input.data {
        Data::Struct(data) => data,
        _ => {
            return Err(syn::Error::new(
                input.span(),
                "`#[derive(Model)]` can only be applied to a struct",
            ));
        }
    };

    let named = match &data.fields {
        Fields::Named(named) => &named.named,
        Fields::Unnamed(fields) => {
            return Err(syn::Error::new(
                fields.span(),
                "`#[derive(Model)]` requires named fields, because parameter names come from field names",
            ));
        }
        Fields::Unit => {
            return Err(syn::Error::new(
                input.span(),
                "`#[derive(Model)]` requires a struct with named fields",
            ));
        }
    };

    let mut out = Vec::with_capacity(named.len());

    for field in named {
        let ident = field
            .ident
            .clone()
            .expect("named fields always have an identifier");

        let attr = field.attrs.iter().find(|a| a.path().is_ident("param"));
        let Some(attr) = attr else {
            return Err(syn::Error::new(
                field.span(),
                format!(
                    "field `{ident}` has no `#[param(...)]` attribute; every field of a \
                     `#[derive(Model)]` struct is a fitted parameter, so each needs \
                     `#[param(value = ...)]`"
                ),
            ));
        };
        let attrs = parse_param(attr)?;

        let Some(value) = attrs.value else {
            return Err(syn::Error::new(
                field.span(),
                format!("field `{ident}` needs a starting value: add `#[param(value = ...)]`"),
            ));
        };

        if !is_f64(&field.ty) {
            return Err(syn::Error::new(
                field.ty.span(),
                "model parameters must be `f64`",
            ));
        }

        out.push(Field {
            ident,
            value,
            min: attrs.min,
            max: attrs.max,
            vary: attrs.vary,
        });
    }

    Ok(out)
}

/// `Some(expr)` for a present bound, `None` for an absent one.
fn option(expr: Option<&syn::Expr>) -> TokenStream {
    match expr {
        Some(expr) => quote!(::std::option::Option::Some(#expr)),
        None => quote!(::std::option::Option::None),
    }
}

fn is_f64(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(path) => path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "f64"),
        _ => false,
    }
}
