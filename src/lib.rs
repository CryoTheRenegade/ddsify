//! Batch-convert PNG images to DDS (BC1 for opaque images, BC7 for images with alpha) with
//! full mip chains. A todds replacement. Synchronous; work runs on rayon's global thread pool.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ddsfile::{
    AlphaMode, D3D10ResourceDimension, D3DFormat, Dds, DxgiFormat, NewD3dParams, NewDxgiParams,
};
use image::RgbaImage;
use image::imageops::{self, FilterType};
use intel_tex_2::{RgbaSurface, bc1, bc7};
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

/// Sent once per finished file (whether it succeeded or failed).
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
    let pngs = find_pngs(root, opts)?;
    let mut report = Report::default();
    let todo: Vec<PathBuf> = pngs
        .into_iter()
        .filter(|png| {
            let go = needs_encode(png, &png.with_extension("dds"), opts.overwrite);
            report.skipped += usize::from(!go);
            go
        })
        .collect();
    run(&todo, &mut report, progress, |png| encode_file(png, opts));
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
    let dds: Vec<PathBuf> = find_pngs(root, opts)?
        .into_iter()
        .map(|p| p.with_extension("dds"))
        .filter(|d| d.is_file())
        .collect();
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
    let done = AtomicUsize::new(0);
    let total = files.len();
    let results: Vec<_> = files
        .par_iter()
        .map(|f| {
            let r = job(f);
            progress(Progress {
                done: done.fetch_add(1, Ordering::Relaxed) + 1,
                total,
                path: f.clone(),
            });
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
    for entry in walkdir::WalkDir::new(&root) {
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
    let dds = png.with_extension("dds");
    let tmp = png.with_extension("dds.tmp");
    fs::write(&tmp, bytes)
        .and_then(|_| fs::rename(&tmp, &dds))
        .map_err(|e| {
            let _ = fs::remove_file(&tmp);
            e.to_string()
        })
}

/// Encode an RGBA image to a DDS with a full mip chain: BC1 if fully opaque, else BC7.
pub fn encode(mut img: RgbaImage, opts: &Options) -> Dds {
    if opts.vflip {
        imageops::flip_vertical_in_place(&mut img);
    }
    let (w, h) = img.dimensions();
    if opts.fix_size && (w % 4 != 0 || h % 4 != 0) {
        // todds pads with transparent pixels instead; scaling keeps the sprite filling its mesh.
        img = imageops::resize(
            &img,
            w.next_multiple_of(4),
            h.next_multiple_of(4),
            FilterType::Lanczos3,
        );
    }
    let (w, h) = img.dimensions();
    let alpha = img.pixels().any(|p| p.0[3] != 255);
    let levels = 32 - w.max(h).leading_zeros();

    let mut data = Vec::new();
    let mut level = img;
    for _ in 0..levels {
        let (lw, lh) = level.dimensions();
        let padded = pad_to_blocks(&level);
        let surface = RgbaSurface {
            data: &padded,
            width: padded.width(),
            height: padded.height(),
            stride: padded.width() * 4,
        };
        data.extend(if alpha {
            bc7::compress_blocks(&bc7::alpha_fast_settings(), &surface)
        } else {
            bc1::compress_blocks(&surface)
        });
        if lw > 1 || lh > 1 {
            level = imageops::resize(
                &level,
                (lw / 2).max(1),
                (lh / 2).max(1),
                FilterType::Triangle,
            );
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
