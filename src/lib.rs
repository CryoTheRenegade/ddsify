//! Batch-convert PNG images to DDS (BC1 for opaque images, BC7 for images with alpha) with
//! full mip chains. A todds replacement. Synchronous; work runs on rayon's global thread pool.

use std::collections::{HashMap, HashSet};
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
    let scan = scan(root, opts)?;
    let mut report = Report::default();
    // DDS path -> whether the first PNG that maps to it is encoded this run.
    let mut targets: HashMap<PathBuf, bool> = HashMap::new();
    // PNG -> Ok(DDS to write) or Err(DDS another PNG already writes).
    let mut jobs: HashMap<PathBuf, Result<PathBuf, PathBuf>> = HashMap::new();
    let mut todo = Vec::new();
    for png in &scan.pngs {
        let dds = scan.target(png);
        let job = match targets.get(&dds) {
            // a.png and a.PNG on a case-sensitive file system share one DDS and the first by
            // name wins. Report the other only when the DDS is written, not on every rerun.
            Some(true) => Err(dds),
            Some(false) => {
                report.skipped += 1;
                continue;
            }
            None => {
                let go = needs_encode(png, &dds, opts.overwrite);
                targets.insert(dds.clone(), go);
                if !go {
                    report.skipped += 1;
                    continue;
                }
                Ok(dds)
            }
        };
        jobs.insert(png.clone(), job);
        todo.push(png.clone());
    }
    run(&todo, &mut report, progress, |png| match &jobs[png] {
        Ok(dds) => encode_file(png, dds, opts),
        Err(dds) => Err(format!(
            "{} is already the target of another PNG",
            dds.display()
        )),
    });
    report.elapsed = start.elapsed();
    Ok(report)
}

/// Delete each DDS under `root` (any extension case) that sits next to a matching PNG of the
/// same name (todds `-cl`), plus staging files left by an interrupted run.
/// A DDS without a sibling PNG is never touched.
pub fn clean(
    root: &Path,
    opts: &Options,
    progress: impl Fn(Progress) + Sync,
) -> io::Result<Report> {
    let start = Instant::now();
    let scan = scan(root, opts)?;
    let mut seen = HashSet::new();
    let mut files = Vec::new();
    for png in &scan.pngs {
        let lower = png.with_extension("dds");
        let found = match scan.dds.get(&png.with_extension("")) {
            Some(found) => found.clone(),
            // Case-insensitive file systems: `A.dds` next to `a.png` is still its DDS.
            None if lower.is_file() => vec![lower],
            None => continue,
        };
        files.extend(found.into_iter().filter(|d| seen.insert(d.clone()))); // a.png + a.PNG
    }
    files.extend(scan.stale_tmp);
    let mut report = Report::default();
    run(&files, &mut report, progress, |f| {
        fs::remove_file(f).map_err(|e| e.to_string())
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

/// What one walk of `root` finds.
struct Scan {
    /// Matching PNGs, sorted by file name within each folder.
    pngs: Vec<PathBuf>,
    /// Existing DDS files keyed by their path without extension, so `a.dds`, `a.DDS` and
    /// `a.Dds` all belong to `a.png`.
    dds: HashMap<PathBuf, Vec<PathBuf>>,
    /// `.ddsify-*.tmp` staging files left by an interrupted run, in matching folders.
    stale_tmp: Vec<PathBuf>,
}

impl Scan {
    /// Where `png`'s DDS goes: an existing one (preferring `.dds`), else `<stem>.dds`.
    fn target(&self, png: &Path) -> PathBuf {
        let lower = png.with_extension("dds");
        match self.dds.get(&png.with_extension("")) {
            Some(found) if !found.contains(&lower) => found[0].clone(),
            _ => lower,
        }
    }
}

fn scan(root: &Path, opts: &Options) -> io::Result<Scan> {
    let root = std::path::absolute(root)?;
    if !root.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{} does not exist", root.display()),
        ));
    }
    let mut scan = Scan {
        pngs: Vec::new(),
        dds: HashMap::new(),
        stale_tmp: Vec::new(),
    };
    let has_ext = |path: &Path, want: &str| {
        path.extension()
            .is_some_and(|e| e.eq_ignore_ascii_case(want))
    };
    let in_dir = |path: &Path| match &opts.dir_name {
        None => true,
        Some(name) => path.parent().is_some_and(|p| {
            p.components().any(|c| {
                c.as_os_str()
                    .to_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case(name))
            })
        }),
    };
    // A file root is the only source, used as given (any case, symlinks followed); its folder
    // is walked one level deep only to find the DDS files beside it.
    let single = root.is_file();
    if single && has_ext(&root, "png") && in_dir(&root) {
        scan.pngs.push(root.clone());
    }
    let walk = match root.parent() {
        Some(dir) if single => walkdir::WalkDir::new(dir).max_depth(1),
        _ => walkdir::WalkDir::new(&root),
    };
    // Sorted so that when two PNGs map to one DDS, the same one wins every run.
    for entry in walk.sort_by_file_name() {
        let entry = entry.map_err(io::Error::other)?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if has_ext(path, "dds") {
            scan.dds
                .entry(path.with_extension(""))
                .or_default()
                .push(path.to_path_buf());
        } else if !single && in_dir(path) {
            if has_ext(path, "png") {
                scan.pngs.push(path.to_path_buf());
            } else if is_staging_name(&entry.file_name().to_string_lossy()) {
                scan.stale_tmp.push(path.to_path_buf());
            }
        }
    }
    Ok(scan)
}

/// Exactly the names `encode_file` stages to, `.ddsify-<pid>-<n>.tmp`, so a user's
/// `.ddsify-backup.tmp` is never taken for one.
fn is_staging_name(name: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    name.strip_prefix(".ddsify-")
        .and_then(|rest| rest.strip_suffix(".tmp"))
        .and_then(|rest| rest.split_once('-'))
        .is_some_and(|(pid, n)| digits(pid) && digits(n))
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

fn encode_file(png: &Path, dds: &Path, opts: &Options) -> Result<(), String> {
    let img = image::open(png).map_err(|e| e.to_string())?.to_rgba8();
    let mut bytes = Vec::new();
    encode(img, opts)
        .write(&mut bytes)
        .map_err(|e| e.to_string())?;
    // Write then rename so an interrupted run never leaves a truncated DDS for the game to load.
    // Unique per call and created exclusively, so other runs and user files are never clobbered.
    // Independent of the stem, so a name near the file-system limit still fits.
    static NEXT_TMP: AtomicU64 = AtomicU64::new(0);
    let (tmp, mut file) = loop {
        let n = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
        let tmp = png.with_file_name(format!(".ddsify-{}-{n}.tmp", std::process::id()));
        match fs::File::create_new(&tmp) {
            Ok(file) => break (tmp, file),
            // Left by a killed run whose process ID we now reuse; try the next name.
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.to_string()),
        }
    };
    let written = file.write_all(&bytes);
    drop(file);
    written.and_then(|_| fs::rename(&tmp, dds)).map_err(|e| {
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
