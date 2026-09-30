# lmfit_derive

Derive macro for [`lmfit`](https://crates.io/crates/lmfit) curve models.

This crate is an implementation detail. Its output is re-exported from `lmfit`,
so depend on that crate and write `use lmfit::Model;` rather than naming this
one.

```toml
[dependencies]
lmfit = "0.1"
```

`#[derive(Model)]` turns a struct into a set of fit parameters — one per field,
in declaration order — and generates `Default` and `Add` so the model can be
built from its starting guesses and combined with `+`. It deliberately does
**not** generate the model's arithmetic: you write that yourself as
`impl Curve`.

See the [`lmfit` README](https://github.com/XNCZ/lmfit#readme) for the full
picture.

## License

MIT — see [LICENSE](../LICENSE).
