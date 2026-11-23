//! Batch-convert PNG images to DDS (BC1 for opaque images, BC7 for images with alpha) with
//! full mip chains. A todds replacement. Synchronous; work runs on rayon's global thread pool.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ddsfile::{
    AlphaMode, D3D10ResourceDimension, D3DFormat, Dds, DxgiFormat, NewD3dParams, NewDxgiParams,
};
use image::imageops::{self, FilterType};
use image::{Rgba, Rgba32FImage, RgbaImage};
use intel_tex_2::{RgbaSurface, bc7};
use rayon::prelude::*;

/// When to replace an existing `.dds`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overwrite {
    /// Never replace an existing `.dds`.
    Never,
    /// Replace it only when the source PNG is newer (todds `-on`).
    IfNewer,
    /// Always replace it (todds `-o`).
    Always,
}

#[derive(Debug, Clone)]
pub struct Options {
    /// Only process PNGs that have an ancestor folder with this name (case-insensitive).
    /// `None` processes every PNG under the root.
    pub dir_name: Option<String>,
    pub overwrite: Overwrite,
    /// Flip images vertically before encoding (needed by loaders that expect bottom-up rows, e.g. RimWorld).
    pub vflip: bool,
    /// Resize images whose width or height is not a multiple of 4 up to the next multiple of 4.
    pub fix_size: bool,
}

impl Default for Options {
    /// Same as the CLI with no flags: every PNG, keep existing DDS files, no flip, no resize.
    fn default() -> Self {
        Options {
            dir_name: None,
            overwrite: Overwrite::Never,
            vflip: false,
            fix_size: false,
        }
    }
}

impl Options {
    /// RimWorld mods, matching RimSort's "optimized" todds preset: only `Textures` folders,
    /// re-encode when the PNG is newer, flip vertically, resize to a multiple of 4.
    pub fn rimworld() -> Self {
        Options {
            dir_name: Some("Textures".into()),
            overwrite: Overwrite::IfNewer,
            vflip: true,
            fix_size: true,
        }
    }
}

/// Sent once per finished file (whether it succeeded or failed), in `done` order.
/// The callback runs under a lock on a rayon worker: keep it quick (e.g. send on a channel)
/// and don't start rayon work from it, or it can deadlock.
#[derive(Debug, Clone)]
pub struct Progress {
    pub done: usize,
    pub total: usize,
    pub path: PathBuf,
}

#[derive(Debug, Default)]
pub struct Report {
    /// DDS files written (optimize) or deleted (clean).
    pub done: usize,
    /// PNGs skipped because their DDS already exists and is up to date (optimize only).
    pub skipped: usize,
    /// Source path and error message for each file that failed.
    pub failed: Vec<(PathBuf, String)>,
    pub elapsed: Duration,
}

/// Encode every matching PNG under `root` (or `root` itself if it is a PNG) to a DDS next to it.
pub fn optimize(
    root: &Path,
    opts: &Options,
    progress: impl Fn(Progress) + Sync,
) -> io::Result<Report> {
    let start = Instant::now();
    let mut report = Report::default();
    let mut targets = HashSet::new();
    let mut conflicts = HashSet::new();
    let mut todo = Vec::new();
    for png in find_pngs(root, opts)? {
        let dds = dds_path(&png);
        if !targets.insert(dds.clone()) {
            // a.png and a.PNG on a case-sensitive file system would both write a.dds.
            conflicts.insert(png.clone());
            todo.push(png);
        } else if needs_encode(&png, &dds, opts.overwrite) {
            todo.push(png);
        } else {
            report.skipped += 1;
        }
    }
    run(&todo, &mut report, progress, |png| {
        if conflicts.contains(png) {
            let dds = dds_path(png);
            return Err(format!(
                "{} is already the target of another PNG",
                dds.display()
            ));
        }
        encode_file(png, opts)
    });
    report.elapsed = start.elapsed();
    Ok(report)
}

/// Delete each `.dds` under `root` that sits next to a matching PNG of the same name (todds `-cl`).
/// A DDS without a sibling PNG is never touched.
pub fn clean(
    root: &Path,
    opts: &Options,
    progress: impl Fn(Progress) + Sync,
) -> io::Result<Report> {
    let start = Instant::now();
    let mut dds: Vec<PathBuf> = find_pngs(root, opts)?
        .iter()
        .map(|p| dds_path(p))
        .filter(|d| d.is_file())
        .collect();
    let mut seen = HashSet::new();
    dds.retain(|d| seen.insert(d.clone())); // a.png and a.PNG share one DDS
    let mut report = Report::default();
    run(&dds, &mut report, progress, |d| {
        fs::remove_file(d).map_err(|e| e.to_string())
    });
    report.elapsed = start.elapsed();
    Ok(report)
}

fn run(
    files: &[PathBuf],
    report: &mut Report,
    progress: impl Fn(Progress) + Sync,
    job: impl Fn(&Path) -> Result<(), String> + Sync,
) {
    keep_full_speed();
    let done = Mutex::new(0);
    let total = files.len();
    let results: Vec<_> = files
        .par_iter()
        .map(|f| {
            let r = job(f);
            // Count and report under one lock so callers never see `done` go backwards.
            let mut done = done.lock().unwrap();
            *done += 1;
            progress(Progress {
                done: *done,
                total,
                path: f.clone(),
            });
            drop(done);
            r.err().map(|e| (f.clone(), e))
        })
        .collect();
    report.failed.extend(results.into_iter().flatten());
    report.done = total - report.failed.len();
}

/// Windows 11 puts processes that are not in the foreground into EcoQoS (E-cores, low clocks),
/// which made a background run about 10x slower in testing. Opt the process out; it is left
/// opted out afterwards, which only matters while it uses CPU.
#[cfg(windows)]
fn keep_full_speed() {
    use windows_sys::Win32::System::Threading::*;
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask: 0,
    };
    // Best effort: failure only costs speed.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            (&raw const state).cast(),
            size_of_val(&state) as u32,
        )
    };
}

#[cfg(not(windows))]
fn keep_full_speed() {}

fn find_pngs(root: &Path, opts: &Options) -> io::Result<Vec<PathBuf>> {
    let root = std::path::absolute(root)?;
    if !root.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{} does not exist", root.display()),
        ));
    }
    let mut out = Vec::new();
    // Sorted so that when two PNGs map to one DDS, the same one wins every run.
    for entry in walkdir::WalkDir::new(&root).sort_by_file_name() {
        let entry = entry.map_err(io::Error::other)?;
        let path = entry.path();
        let is_png = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("png"));
        let in_dir = match &opts.dir_name {
            None => true,
            Some(name) => path.parent().is_some_and(|p| {
                p.components().any(|c| {
                    c.as_os_str()
                        .to_str()
                        .is_some_and(|s| s.eq_ignore_ascii_case(name))
                })
            }),
        };
        if entry.file_type().is_file() && is_png && in_dir {
            out.push(entry.into_path());
        }
    }
    Ok(out)
}

/// The DDS next to `png`: an existing `.DDS` is reused on case-sensitive file systems.
// shortcut: mixed-case extensions like `.Dds` are not matched; add a directory scan if one shows up.
fn dds_path(png: &Path) -> PathBuf {
    let (lower, upper) = (png.with_extension("dds"), png.with_extension("DDS"));
    if !lower.exists() && upper.is_file() {
        upper
    } else {
        lower
    }
}

fn needs_encode(png: &Path, dds: &Path, overwrite: Overwrite) -> bool {
    let modified = |p: &Path| fs::metadata(p).and_then(|m| m.modified());
    match (overwrite, modified(dds)) {
        (Overwrite::Always, _) | (_, Err(_)) => true,
        (Overwrite::Never, Ok(_)) => false,
        (Overwrite::IfNewer, Ok(dds_time)) => {
            modified(png).map_or(true, |png_time| png_time > dds_time)
        }
    }
}

fn encode_file(png: &Path, opts: &Options) -> Result<(), String> {
    let img = image::open(png).map_err(|e| e.to_string())?.to_rgba8();
    let mut bytes = Vec::new();
    encode(img, opts)
        .write(&mut bytes)
        .map_err(|e| e.to_string())?;
    // Write then rename so an interrupted run never leaves a truncated DDS for the game to load.
    // Unique per call and created exclusively, so other runs and user files are never clobbered.
    static NEXT_TMP: AtomicU64 = AtomicU64::new(0);
    let n = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
    // Independent of the stem, so a name near the file-system limit still fits.
    let tmp = png.with_file_name(format!(".ddsify-{}-{n}.tmp", std::process::id()));
    let mut file = fs::File::create_new(&tmp).map_err(|e| e.to_string())?;
    let written = file.write_all(&bytes);
    drop(file);
    written
        .and_then(|_| fs::rename(&tmp, dds_path(png)))
        .map_err(|e| {
            let _ = fs::remove_file(&tmp); // ours: create_new succeeded
            e.to_string()
        })
}

/// Encode an RGBA image to a DDS with a full mip chain: BC1 if fully opaque, else BC7.
fn encode(mut img: RgbaImage, opts: &Options) -> Dds {
    // Decide before resizing: Lanczos can round a lone 254 alpha up to 255.
    let alpha = img.pixels().any(|p| p.0[3] != 255);
    if opts.vflip {
        imageops::flip_vertical_in_place(&mut img);
    }
    // Resize in premultiplied alpha so the color of transparent pixels can't bleed into
    // visible ones (todds filters straight alpha, which halos edges in lower mips).
    let mut filtered = premultiply(&img);
    let (w, h) = img.dimensions();
    if opts.fix_size && (w % 4 != 0 || h % 4 != 0) {
        // todds pads with transparent pixels instead; scaling keeps the sprite filling its mesh.
        let (fw, fh) = (w.next_multiple_of(4), h.next_multiple_of(4));
        filtered = imageops::resize(&filtered, fw, fh, FilterType::Lanczos3);
        img = unpremultiply(&filtered);
    }
    let (w, h) = img.dimensions();
    let levels = 32 - w.max(h).leading_zeros();

    let mut data = Vec::new();
    let mut level = img;
    for _ in 0..levels {
        let (lw, lh) = level.dimensions();
        if alpha {
            let padded = pad_to_blocks(&level);
            let surface = RgbaSurface {
                data: &padded,
                width: padded.width(),
                height: padded.height(),
                stride: padded.width() * 4,
            };
            data.extend(bc7::compress_blocks(&bc7::alpha_fast_settings(), &surface));
        } else {
            // Not intel_tex_2's BC1: it (and squish) flatten blocks with anti-correlated colors,
            // e.g. red/green mask textures, to one average color. rgbcx is what todds uses.
            data.extend(
                rgbcx::encode(lw, lh, &level, rgbcx::Format::Bc1, rgbcx::Quality::Balanced)
                    .expect("RGBA buffer matches its dimensions"),
            );
        }
        if lw > 1 || lh > 1 {
            filtered = imageops::resize(
                &filtered,
                (lw / 2).max(1),
                (lh / 2).max(1),
                FilterType::Triangle,
            );
            level = unpremultiply(&filtered);
        }
    }

    let mut dds = if alpha {
        Dds::new_dxgi(NewDxgiParams {
            height: h,
            width: w,
            depth: None,
            format: DxgiFormat::BC7_UNorm,
            mipmap_levels: Some(levels),
            array_layers: None,
            caps2: None,
            is_cubemap: false,
            resource_dimension: D3D10ResourceDimension::Texture2D,
            alpha_mode: AlphaMode::Straight,
        })
    } else {
        Dds::new_d3d(NewD3dParams {
            height: h,
            width: w,
            depth: None,
            format: D3DFormat::DXT1,
            mipmap_levels: Some(levels),
            caps2: None,
        })
    }
    .expect("BC1/BC7 are supported by ddsfile");
    assert_eq!(dds.data.len(), data.len(), "mip chain size mismatch");
    dds.data = data;
    dds
}

fn premultiply(img: &RgbaImage) -> Rgba32FImage {
    Rgba32FImage::from_fn(img.width(), img.height(), |x, y| {
        let [r, g, b, a] = img.get_pixel(x, y).0.map(|c| f32::from(c) / 255.0);
        Rgba([r * a, g * a, b * a, a])
    })
}

fn unpremultiply(img: &Rgba32FImage) -> RgbaImage {
    RgbaImage::from_fn(img.width(), img.height(), |x, y| {
        let [r, g, b, a] = img.get_pixel(x, y).0;
        let a = a.clamp(0.0, 1.0);
        let straight = |c: f32| {
            if a > 0.0 {
                (c / a).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        Rgba([straight(r), straight(g), straight(b), a].map(|c| (c * 255.0).round() as u8))
    })
}

/// Grow the image to whole 4x4 blocks by repeating its last row and column.
fn pad_to_blocks(img: &RgbaImage) -> RgbaImage {
    let (w, h) = img.dimensions();
    let (pw, ph) = (w.next_multiple_of(4), h.next_multiple_of(4));
    if (pw, ph) == (w, h) {
        return img.clone();
    }
    RgbaImage::from_fn(pw, ph, |x, y| *img.get_pixel(x.min(w - 1), y.min(h - 1)))
}

#[cfg(test)]
mod tests;
