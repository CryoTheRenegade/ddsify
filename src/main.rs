use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use ddsify::{Options, Overwrite, Progress};

/// Convert PNG textures to DDS: BC1 for opaque images, BC7 for images with alpha, full mipmaps.
/// todds-style short flags (-cl, -on, -vf, -fs, -ss) are accepted too.
#[derive(Parser)]
#[command(version)]
struct Cli {
    /// Directory to process recursively, or a single PNG. DDS files are written next to each PNG.
    input: PathBuf,
    /// Delete DDS files that sit next to a matching PNG instead of encoding (todds -cl).
    #[arg(long)]
    clean: bool,
    /// Replace existing DDS files (todds -o).
    #[arg(short, long)]
    overwrite: bool,
    /// Replace existing DDS files only when the PNG is newer (todds -on).
    #[arg(long)]
    overwrite_new: bool,
    /// Flip images vertically before encoding, e.g. for RimWorld (todds -vf).
    #[arg(long)]
    vflip: bool,
    /// Resize images to the next multiple of 4 (todds -fs; todds pads instead of scaling).
    #[arg(long)]
    fix_size: bool,
    /// Only process PNGs inside a folder with this name, e.g. Textures (todds -ss matches a substring instead).
    #[arg(long, value_name = "NAME")]
    dir_name: Option<String>,
    /// Show progress.
    #[arg(short, long)]
    progress: bool,
    /// Show total execution time.
    #[arg(short, long)]
    time: bool,
}

/// todds uses multi-letter short flags, which clap cannot express; map them to the long names.
/// Arguments after `--` are left alone so a path can be named like a flag.
fn todds_args(args: impl Iterator<Item = OsString>) -> impl Iterator<Item = OsString> {
    let mut literal = false;
    args.map(move |a| {
        if literal {
            return a;
        }
        literal = a == "--";
        let long = match a.to_str() {
            Some("-cl") => "--clean",
            Some("-on") => "--overwrite-new",
            Some("-vf") => "--vflip",
            Some("-fs") => "--fix-size",
            Some("-ss") => "--dir-name",
            _ => return a,
        };
        long.into()
    })
}

fn main() -> ExitCode {
    let cli = Cli::parse_from(todds_args(std::env::args_os()));
    let opts = Options {
        dir_name: cli.dir_name,
        overwrite: if cli.overwrite {
            Overwrite::Always
        } else if cli.overwrite_new {
            Overwrite::IfNewer
        } else {
            Overwrite::Never
        },
        vflip: cli.vflip,
        fix_size: cli.fix_size,
    };
    let show = |p: Progress| {
        if cli.progress {
            eprint!("\r{}/{}", p.done, p.total);
        }
    };
    let result = if cli.clean {
        ddsify::clean(&cli.input, &opts, show)
    } else {
        ddsify::optimize(&cli.input, &opts, show)
    };
    let report = match result {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    if cli.progress && report.done + report.failed.len() > 0 {
        eprintln!();
    }
    for (path, err) in &report.failed {
        eprintln!("failed: {}: {err}", path.display());
    }
    let verb = if cli.clean { "deleted" } else { "encoded" };
    println!(
        "{} {verb}, {} up to date, {} failed",
        report.done,
        report.skipped,
        report.failed.len()
    );
    if cli.time {
        println!("time: {:.2?}", report.elapsed);
    }
    if report.failed.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todds_flags_are_rewritten_until_double_dash() {
        let args = ["ddsify", "-cl", "--", "-cl"]
            .map(OsString::from)
            .into_iter();
        let out: Vec<OsString> = todds_args(args).collect();
        assert_eq!(out, ["ddsify", "--clean", "--", "-cl"].map(OsString::from));
    }
}
