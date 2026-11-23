# ddsify

Batch-converts PNG images to DDS. It's a drop-in replacement for [todds](https://github.com/todds-encoder/todds), with a library API too.

- Opaque images become **BC1** (legacy `DXT1` header). Images with any alpha below 255 become **BC7** (`DX10` header, `BC7_UNORM`).
- Writes a full mipmap chain, with an optional vertical flip and resize to a multiple of 4.
- Writes the `.dds` next to the `.png`. Reruns can skip PNGs that are older than their DDS.
- Clean mode deletes only `.dds` files that have a sibling `.png`. Other DDS files are left alone.
- Runs in parallel on all cores (rayon). Builds on Windows, macOS and Linux (x86_64 and aarch64) with plain `cargo build`.
- Encoders: BC1 uses [rgbcx](https://crates.io/crates/rgbcx), a pure-Rust port of the encoder todds uses. BC7 uses [intel_tex_2](https://crates.io/crates/intel_tex_2) (Intel ISPC kernels, prebuilt for the targets above) with its "alpha fast" profile.

## CLI

```sh
ddsify -on -p <dir>                          # encode every PNG, skip up-to-date DDS
ddsify -cl -p <dir>                          # delete DDS files that have a sibling PNG
ddsify -on -vf -fs -ss Textures -t -p <dir>  # RimWorld mods (RimSort's todds preset)
```

todds's multi-letter short flags (`-cl -on -vf -fs -ss`) map to long flags (`--clean --overwrite-new --vflip --fix-size --dir-name`). Run `ddsify --help` for the full list.

Differences from todds:
- `-ss`/`--dir-name` matches a whole folder name (case-insensitive), not a substring of the path.
- `-fs` scales the image up to the next multiple of 4. todds pads it with transparent pixels, which shifts the image and turns opaque images into BC7.
- Mipmaps (and `-fs`) are filtered with premultiplied alpha, so the color of fully transparent pixels doesn't bleed into visible edges in lower mips. todds filters straight alpha.
- The formats are fixed: there is no `-f` or `-af`. There is also no `--regex`, `--dry-run`, `--threads`, quality setting, or TXT/output-dir input.

## Library

```toml
ddsify = { path = "../ddsify", default-features = false }  # skips clap
```

```rust
let opts = ddsify::Options::rimworld(); // or Options::default(), or set the fields
let report = ddsify::optimize(&dir, &opts, |p: ddsify::Progress| {
    let _ = tx.send(p); // p.done, p.total, p.path
})?;
// report.done, report.skipped, report.failed: Vec<(PathBuf, String)>, report.elapsed
ddsify::clean(&dir, &opts, |_| {})?;
```

The calls are synchronous and blocking, so run them on a worker thread. `Options::default()` matches the CLI with no flags. `Options::rimworld()` is RimSort's optimize preset. Errors on individual files go into `Report::failed`. Only a missing root or an unreadable directory returns `Err`.

On Windows, both calls opt the process out of power throttling (EcoQoS). Without that, a run whose window is not in the foreground ran about 10x slower in testing.

## License

GPL-3.0-only.
