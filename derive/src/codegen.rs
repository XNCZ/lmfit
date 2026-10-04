//! Code generation for `#[derive(Model)]`.
//!
//! Given a struct whose fields are all `f64` and all carry `#[param(...)]`,
//! this emits three things. 带 `#[param(derive)]` 的字段另算:它们不参与拟合,
//! 值由与字段同名的固有方法算出,`refresh_derive` 负责把结果写回字段。
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
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Result, spanned::Spanned};

use crate::attr::{parse_model_name, parse_param, to_snake_case};

/// 一个字段在模型里的角色。
enum Role {
    /// 普通参数:起点、界与可变性都来自 `#[param(...)]`。
    Plain {
        value: syn::Expr,
        min: Option<syn::Expr>,
        max: Option<syn::Expr>,
        vary: Option<syn::Expr>,
    },
    /// 派生量:不参与拟合,值由同名方法给出。
    Derive,
}

/// One parsed field: the struct field plus its role.
struct Field {
    ident: syn::Ident,
    role: Role,
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

    // 派生字段:声明序即 `refresh_derive` 的求值序,后者可引用先者。
    let derive_idents: Vec<&syn::Ident> = fields
        .iter()
        .filter_map(|f| match &f.role {
            Role::Derive => Some(&f.ident),
            Role::Plain { .. } => None,
        })
        .collect();
    let has_derive = !derive_idents.is_empty();

    // 偏导包与偏导臂只收普通字段:派生量不是自变量,没有 ∂f/∂θ。
    let plain_idents: Vec<&syn::Ident> = fields
        .iter()
        .filter_map(|f| match &f.role {
            Role::Plain { .. } => Some(&f.ident),
            Role::Derive => None,
        })
        .collect();
    let partial_arms: Vec<TokenStream> = fields
        .iter()
        .enumerate()
        .filter_map(|(i, f)| match &f.role {
            Role::Plain { .. } => {
                let field = &f.ident;
                Some(quote!(#i => self.#field,))
            }
            Role::Derive => None,
        })
        .collect();

    // 全部字段都是派生量时,偏导包里没有字段承载类型参数 `T`;用 `PhantomData`
    // 给它归属,否则生成的空结构体过不了 E0392。
    let partials_marker = match plain_idents.is_empty() {
        true => quote!(_marker: ::std::marker::PhantomData<T>,),
        false => quote!(),
    };

    // 偏导包的伴随类型:名字加 `Partials` 后缀,可见性镜射模型本身;
    // `#[allow(dead_code)]` 因为未覆写 `partials_at` 的私有模型不会构造它。
    let partials_name = format_ident!("{}Partials", name);
    let partials_doc = format!(
        "`#[derive(Model)]` 为 `{name}` 生成的偏导包:字段与模型同名,每个字段承载该参数的一阶偏导。派生字段没有偏导,故不在此包内。`T` 取 `f64`(实值模型)或 `Complex64`(复数模型)。"
    );
    let vis = &input.vis;

    // specs(): one entry per field, in declaration order — which is also the
    // index order `get`, `set`, and `at_values` use.
    //
    // `value` reads the field's *current* value, not the `#[param(value = ..)]`
    // starting value: a spec describes where the model is now, and the fit
    // seeds itself from exactly that.
    let spec_entries = fields.iter().map(|f| {
        let field = &f.ident;
        let field_name = field.to_string();
        let (min, max, vary, derive) = match &f.role {
            Role::Plain {
                min, max, vary, ..
            } => (
                option(min.as_ref()),
                option(max.as_ref()),
                match vary {
                    Some(expr) => quote!(#expr),
                    None => quote!(true),
                },
                quote!(false),
            ),
            Role::Derive => (
                quote!(::std::option::Option::None),
                quote!(::std::option::Option::None),
                quote!(false),
                quote!(true),
            ),
        };
        quote! {
            ::lmfit::ParamSpec {
                name: ::std::string::String::from(#field_name),
                value: self.#field,
                min: #min,
                max: #max,
                vary: #vary,
                derive: #derive,
            }
        }
    });

    let get_arms: Vec<TokenStream> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let field = &f.ident;
            quote!(#i => self.#field,)
        })
        .collect();

    let set_arms = fields.iter().enumerate().map(|(i, f)| {
        let field = &f.ident;
        quote!(#i => self.#field = value,)
    });

    let value_fields: Vec<TokenStream> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let field = &f.ident;
            match &f.role {
                Role::Plain { .. } => quote!(#field: values[#i],),
                Role::Derive => quote!(#field: ::std::f64::NAN,),
            }
        })
        .collect();

    let default_fields: Vec<TokenStream> = fields
        .iter()
        .map(|f| {
            let field = &f.ident;
            match &f.role {
                Role::Plain { value, .. } => quote!(#field: #value,),
                Role::Derive => quote!(#field: ::std::f64::NAN,),
            }
        })
        .collect();

    // 无派生字段时,生成的代码与不带本特性时逐字相同。
    // 全限定路径:调用方不必把 `ModelParams` 导入作用域,派生模型开箱即用。
    let set_refresh = match has_derive {
        true => quote!(::lmfit::ModelParams::refresh_derive(self);),
        false => quote!(),
    };
    let at_values_body = match has_derive {
        true => quote! {
            let mut out = Self {
                #(#value_fields)*
            };
            ::lmfit::ModelParams::refresh_derive(&mut out);
            out
        },
        false => quote! {
            Self {
                #(#value_fields)*
            }
        },
    };
    let default_body = match has_derive {
        true => quote! {
            let mut out = Self {
                #(#default_fields)*
            };
            ::lmfit::ModelParams::refresh_derive(&mut out);
            out
        },
        false => quote! {
            Self {
                #(#default_fields)*
            }
        },
    };
    // 派生字段的占位值取 NaN:公式若引用了后声明的派生字段,得 NaN 而不是
    // 静默的 0,报表与残差都会把它显出来。
    let refresh_impl = match has_derive {
        true => quote! {
            /// 由同名方法重算全部派生字段,声明序即求值序。
            fn refresh_derive(&mut self) {
                #(self.#derive_idents = Self::#derive_idents(self);)*
            }
        },
        false => quote!(),
    };

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
                #set_refresh
            }

            fn at_values(&self, values: &[f64]) -> Self {
                ::std::assert!(
                    values.len() == #nparams,
                    "model `{}` expects {} parameter values, got {}",
                    #model_name,
                    #nparams,
                    values.len(),
                );
                #at_values_body
            }

            #refresh_impl
        }

        impl ::std::default::Default for #name {
            fn default() -> Self {
                #default_body
            }
        }

        #[allow(dead_code)]
        #[doc = #partials_doc]
        #[derive(Debug, Clone)]
        #vis struct #partials_name<T = f64> {
            #(pub #plain_idents: T,)*
            #partials_marker
        }

        impl ::lmfit::PartialValues for #partials_name<f64> {
            type Scalar = f64;

            fn len(&self) -> usize {
                #nparams
            }

            fn get(&self, index: usize) -> f64 {
                match index {
                    #(#partial_arms)*
                    _ => ::std::panic!(
                        "parameter index {} of model `{}` has no partial derivative (out of range, or derived)",
                        index,
                        #model_name,
                    ),
                }
            }
        }

        impl ::lmfit::PartialValues for #partials_name<::lmfit::Complex64> {
            type Scalar = ::lmfit::Complex64;

            fn len(&self) -> usize {
                #nparams
            }

            fn get(&self, index: usize) -> ::lmfit::Complex64 {
                match index {
                    #(#partial_arms)*
                    _ => ::std::panic!(
                        "parameter index {} of model `{}` has no partial derivative (out of range, or derived)",
                        index,
                        #model_name,
                    ),
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

        if !is_f64(&field.ty) {
            return Err(syn::Error::new(
                field.ty.span(),
                "model parameters must be `f64`",
            ));
        }

        // 派生字段没有起点、没有界、不可变:任一条都与之矛盾,在编译期报出
        // 而不是静默丢弃。
        let role = match attrs.derive {
            true => {
                if attrs.value.is_some() {
                    return Err(syn::Error::new(
                        field.span(),
                        format!(
                            "field `{ident}` is derived: its value comes from the method \
                             `{ident}()`, so `value` must be dropped"
                        ),
                    ));
                }
                if attrs.min.is_some() || attrs.max.is_some() {
                    return Err(syn::Error::new(
                        field.span(),
                        format!("field `{ident}` is derived and cannot be bounded"),
                    ));
                }
                if attrs.vary.is_some() {
                    return Err(syn::Error::new(
                        field.span(),
                        format!("field `{ident}` is derived and is never varied"),
                    ));
                }
                Role::Derive
            }
            false => {
                let Some(value) = attrs.value else {
                    return Err(syn::Error::new(
                        field.span(),
                        format!(
                            "field `{ident}` needs a starting value: add `#[param(value = ...)]`"
                        ),
                    ));
                };
                Role::Plain {
                    value,
                    min: attrs.min,
                    max: attrs.max,
                    vary: attrs.vary,
                }
            }
        };

        out.push(Field { ident, role });
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
