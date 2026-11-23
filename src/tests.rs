use super::*;
use image::Rgba;

/// Decode one mip level with bcdec_rs (independent of the intel_tex_2 encoder) into a w x h RGBA buffer.
fn decode_level(dds: &Dds, level: u32) -> RgbaImage {
    let bc7 = dds.get_dxgi_format() == Some(DxgiFormat::BC7_UNorm);
    let block = if bc7 { 16 } else { 8 };
    let dims = |l: u32| {
        (
            (dds.get_width() >> l).max(1),
            (dds.get_height() >> l).max(1),
        )
    };
    let offset: usize = (0..level)
        .map(dims)
        .map(|(w, h)| (w.div_ceil(4) * h.div_ceil(4)) as usize * block)
        .sum();
    let (w, h) = dims(level);
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    let mut out = vec![0u8; (bw * 4 * bh * 4 * 4) as usize];
    let pitch = (bw * 16) as usize;
    for by in 0..bh {
        for bx in 0..bw {
            let src = &dds.data[offset + ((by * bw + bx) as usize) * block..][..block];
            let dst = &mut out[(by * 4) as usize * pitch + (bx * 16) as usize..];
            if bc7 {
                bcdec_rs::bc7(src, dst, pitch)
            } else {
                bcdec_rs::bc1(src, dst, pitch)
            }
        }
    }
    let full = RgbaImage::from_raw(bw * 4, bh * 4, out).unwrap();
    imageops::crop_imm(&full, 0, 0, w, h).to_image()
}

fn roundtrip(img: RgbaImage, opts: &Options) -> Dds {
    let mut bytes = Vec::new();
    encode(img, opts).write(&mut bytes).unwrap();
    Dds::read(&bytes[..]).unwrap()
}

/// Block compression is lossy: allow small per-channel errors but catch gross ones (flip, channel order, wrong level).
fn assert_close(a: &RgbaImage, b: &RgbaImage) {
    assert_eq!(a.dimensions(), b.dimensions());
    let diffs: Vec<u8> = a
        .as_raw()
        .iter()
        .zip(b.as_raw())
        .map(|(x, y)| x.abs_diff(*y))
        .collect();
    let max = *diffs.iter().max().unwrap();
    let mean = diffs.iter().map(|&d| d as f64).sum::<f64>() / diffs.len() as f64;
    assert!(max <= 24 && mean <= 4.0, "max {max}, mean {mean:.2}");
}

fn gradient(w: u32, h: u32, alpha: bool) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        let a = if alpha {
            (x * 255 / w.max(2)) as u8
        } else {
            255
        };
        Rgba([(x * 255 / w) as u8, (y * 255 / h) as u8, 128, a])
    })
}

const NO_FLIP: Options = Options {
    dir_name: None,
    overwrite: Overwrite::IfNewer,
    vflip: false,
    fix_size: true,
};

#[test]
fn opaque_pot_is_bc1_with_full_mips() {
    let src = gradient(64, 64, false);
    let dds = roundtrip(src.clone(), &NO_FLIP);
    assert_eq!(dds.get_d3d_format(), Some(D3DFormat::DXT1));
    assert!(dds.header10.is_none());
    assert_eq!(
        (
            dds.get_width(),
            dds.get_height(),
            dds.get_num_mipmap_levels()
        ),
        (64, 64, 7)
    );
    assert_close(&decode_level(&dds, 0), &src);
    assert_eq!(decode_level(&dds, 6).dimensions(), (1, 1));
}

#[test]
fn alpha_is_bc7_dx10() {
    let src = gradient(32, 16, true);
    let dds = roundtrip(src.clone(), &NO_FLIP);
    assert_eq!(dds.get_dxgi_format(), Some(DxgiFormat::BC7_UNorm));
    assert_eq!(
        &dds.header.spf.fourcc.as_ref().unwrap().0.to_le_bytes(),
        b"DX10"
    );
    assert_eq!(dds.get_num_mipmap_levels(), 6); // 32x16 .. 1x1
    assert_close(&decode_level(&dds, 0), &src);
}

#[test]
fn single_transparent_pixel_selects_bc7() {
    let mut src = gradient(8, 8, false);
    src.put_pixel(3, 3, Rgba([0, 0, 0, 254]));
    assert_eq!(
        roundtrip(src, &NO_FLIP).get_dxgi_format(),
        Some(DxgiFormat::BC7_UNorm)
    );
}

#[test]
fn fix_size_resizes_to_multiple_of_4() {
    let src = gradient(30, 18, false);
    let dds = roundtrip(src.clone(), &NO_FLIP);
    assert_eq!(
        (
            dds.get_width(),
            dds.get_height(),
            dds.get_num_mipmap_levels()
        ),
        (32, 20, 6)
    );
    let expect = imageops::resize(&src, 32, 20, FilterType::Lanczos3);
    assert_close(&decode_level(&dds, 0), &expect);
    // Resized, not padded: an opaque image must stay opaque (BC1).
    assert_eq!(dds.get_d3d_format(), Some(D3DFormat::DXT1));
}

#[test]
fn without_fix_size_keeps_odd_size() {
    let src = gradient(30, 18, true);
    let dds = roundtrip(
        src.clone(),
        &Options {
            fix_size: false,
            ..NO_FLIP
        },
    );
    assert_eq!(
        (
            dds.get_width(),
            dds.get_height(),
            dds.get_num_mipmap_levels()
        ),
        (30, 18, 5)
    );
    assert_close(&decode_level(&dds, 0), &src);
}

#[test]
fn vflip_puts_bottom_row_first() {
    let src = RgbaImage::from_fn(16, 16, |_, y| {
        if y < 8 {
            Rgba([255, 0, 0, 255])
        } else {
            Rgba([0, 0, 255, 255])
        }
    });
    let flipped = decode_level(
        &roundtrip(
            src.clone(),
            &Options {
                vflip: true,
                ..NO_FLIP
            },
        ),
        0,
    );
    assert_eq!(flipped.get_pixel(5, 0), &Rgba([0, 0, 255, 255]));
    assert_eq!(flipped.get_pixel(5, 15), &Rgba([255, 0, 0, 255]));
    assert_eq!(
        decode_level(&roundtrip(src, &NO_FLIP), 0).get_pixel(5, 0),
        &Rgba([255, 0, 0, 255])
    );
}

fn write_png(path: &Path, img: &RgbaImage) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    img.save(path).unwrap();
}

#[test]
fn optimize_is_incremental_and_clean_spares_shipped_dds() {
    let dir = tempfile::tempdir().unwrap();
    let tex = dir.path().join("ModA/Textures/Things");
    let a = tex.join("a.png");
    write_png(&a, &gradient(16, 16, false));
    write_png(&tex.join("b.PNG"), &gradient(10, 6, true));
    write_png(
        &dir.path().join("ModA/About/Preview.png"),
        &gradient(8, 8, false),
    );
    let shipped = tex.join("shipped.dds");
    fs::write(&shipped, b"author dds").unwrap();

    let opts = Options::rimworld();
    let r = optimize(dir.path(), &opts, |_| {}).unwrap();
    assert_eq!((r.done, r.skipped, r.failed.len()), (2, 0, 0));
    assert!(tex.join("b.dds").is_file());
    assert!(
        !dir.path().join("ModA/About/Preview.dds").exists(),
        "only Textures folders"
    );
    let tmp_left = fs::read_dir(&tex)
        .unwrap()
        .flatten()
        .any(|e| e.path().to_string_lossy().ends_with(".tmp"));
    assert!(!tmp_left, "staging files are renamed away");

    let r = optimize(dir.path(), &opts, |_| {}).unwrap();
    assert_eq!((r.done, r.skipped), (0, 2));

    // A source newer than its DDS gets re-encoded.
    let later =
        fs::metadata(tex.join("a.dds")).unwrap().modified().unwrap() + Duration::from_secs(5);
    fs::File::options()
        .write(true)
        .open(&a)
        .unwrap()
        .set_modified(later)
        .unwrap();
    let seen = std::sync::Mutex::new(Vec::new());
    let r = optimize(dir.path(), &opts, |p| {
        seen.lock().unwrap().push((p.done, p.total))
    })
    .unwrap();
    assert_eq!((r.done, r.skipped), (1, 1));
    assert_eq!(seen.into_inner().unwrap(), vec![(1, 1)]);

    let r = clean(dir.path(), &opts, |_| {}).unwrap();
    assert_eq!(r.done, 2);
    assert!(!tex.join("a.dds").exists() && !tex.join("b.dds").exists());
    assert_eq!(fs::read(&shipped).unwrap(), b"author dds");
}

#[test]
fn bad_png_is_reported_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let tex = dir.path().join("Textures");
    write_png(&tex.join("ok.png"), &gradient(8, 8, false));
    fs::write(tex.join("broken.png"), b"not a png").unwrap();
    let r = optimize(dir.path(), &Options::rimworld(), |_| {}).unwrap();
    assert_eq!(r.done, 1);
    assert_eq!(r.failed.len(), 1);
    assert!(r.failed[0].0.ends_with("broken.png"));
    assert!(!tex.join("broken.dds").exists());
}

#[test]
fn missing_root_is_an_error() {
    assert!(
        optimize(
            Path::new("definitely/not/here"),
            &Options::rimworld(),
            |_| {}
        )
        .is_err()
    );
}

#[test]
fn bc1_keeps_hard_red_green_edges() {
    // RimWorld `_m` mask textures; intel_tex_2's BC1 flattened these blocks to one average color.
    let red = Rgba([255, 0, 0, 255]);
    let green = Rgba([0, 255, 0, 255]);
    let src = RgbaImage::from_fn(8, 8, |x, y| if x < 1 + y / 2 { red } else { green });
    let out = decode_level(&roundtrip(src.clone(), &NO_FLIP), 0);
    assert_eq!(out, src);
}

#[test]
fn fix_size_keeps_faint_alpha_as_bc7() {
    let mut src = gradient(7, 7, false);
    src.put_pixel(3, 3, Rgba([10, 20, 30, 254]));
    let dds = roundtrip(src, &NO_FLIP);
    assert_eq!(dds.get_dxgi_format(), Some(DxgiFormat::BC7_UNorm));
}

#[test]
fn transparent_color_does_not_bleed_into_mips() {
    // Opaque red next to fully transparent blue: the 1x1 mip must stay red, not turn purple.
    let src = RgbaImage::from_fn(8, 8, |x, _| {
        if x < 4 {
            Rgba([255, 0, 0, 255])
        } else {
            Rgba([0, 0, 255, 0])
        }
    });
    let dds = roundtrip(src, &NO_FLIP);
    let last = decode_level(&dds, dds.get_num_mipmap_levels() - 1);
    let [r, _, b, a] = last.get_pixel(0, 0).0;
    assert!(
        r > 240 && b < 16 && (100..156).contains(&a),
        "1x1 mip {:?}",
        [r, b, a]
    );
}

/// Needs a case-sensitive file system to create `a.png` next to `a.PNG`.
#[cfg(target_os = "linux")]
#[test]
fn case_variants_share_one_dds() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    write_png(&d.join("a.png"), &gradient(8, 8, false));
    write_png(&d.join("a.PNG"), &gradient(8, 8, false));
    write_png(&d.join("b.png"), &gradient(8, 8, false));
    fs::write(d.join("b.Dds"), b"old").unwrap();
    write_png(&d.join("c.png"), &gradient(8, 8, false));
    fs::write(d.join("c.dds"), b"old").unwrap();
    fs::write(d.join("c.DDS"), b"old").unwrap();

    let opts = Options {
        overwrite: Overwrite::Always,
        ..Options::default()
    };
    let r = optimize(d, &opts, |_| {}).unwrap();
    assert_eq!((r.done, r.failed.len()), (3, 1), "{:?}", r.failed);
    assert!(
        r.failed[0].0.ends_with("a.png"),
        "a.PNG sorts first and wins"
    );
    assert!(d.join("a.dds").is_file());
    assert_ne!(
        fs::read(d.join("b.Dds")).unwrap(),
        b"old",
        "existing b.Dds is replaced"
    );
    assert!(!d.join("b.dds").exists(), "no second b.dds");
    assert_ne!(
        fs::read(d.join("c.dds")).unwrap(),
        b"old",
        "c.dds is preferred"
    );

    // Nothing is written on a rerun, so the a.png/a.PNG conflict is not reported again.
    let r = optimize(d, &Options::default(), |_| {}).unwrap();
    assert_eq!(
        (r.done, r.skipped, r.failed.len()),
        (0, 4, 0),
        "{:?}",
        r.failed
    );

    let r = clean(d, &opts, |_| {}).unwrap();
    assert_eq!((r.done, r.failed.len()), (4, 0), "{:?}", r.failed);
    for f in ["a.dds", "b.Dds", "c.dds", "c.DDS"] {
        assert!(!d.join(f).exists(), "{f} deleted");
    }
}

#[test]
fn clean_removes_leftover_staging_files_only() {
    let dir = tempfile::tempdir().unwrap();
    let tex = dir.path().join("Textures");
    write_png(&tex.join("a.png"), &gradient(8, 8, false));
    fs::write(tex.join(".ddsify-1-2.tmp"), b"partial").unwrap();
    let keep = [
        tex.join("keep.tmp"),
        tex.join(".ddsify-backup.tmp"),
        dir.path().join(".ddsify-3-4.tmp"), // outside the Textures filter
    ];
    for f in &keep {
        fs::write(f, b"user").unwrap();
    }
    let r = clean(dir.path(), &Options::rimworld(), |_| {}).unwrap();
    assert_eq!((r.done, r.failed.len()), (1, 0));
    assert!(!tex.join(".ddsify-1-2.tmp").exists());
    for f in &keep {
        assert!(f.exists(), "{} kept", f.display());
    }
}

/// Case-insensitive file systems: `A.dds` beside `a.png` is that PNG's DDS.
#[cfg(windows)]
#[test]
fn clean_finds_dds_with_other_stem_case() {
    let dir = tempfile::tempdir().unwrap();
    write_png(&dir.path().join("a.png"), &gradient(8, 8, false));
    fs::write(dir.path().join("A.dds"), b"old").unwrap();
    let r = clean(dir.path(), &Options::default(), |_| {}).unwrap();
    assert_eq!((r.done, r.failed.len()), (1, 0));
    assert!(!dir.path().join("A.dds").exists());
}

/// Needs a case-sensitive file system to tell `a.DDS` from `a.dds`.
#[cfg(target_os = "linux")]
#[test]
fn single_png_root_reuses_uppercase_dds() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("a.png");
    write_png(&png, &gradient(8, 8, false));
    write_png(&dir.path().join("b.png"), &gradient(8, 8, false));
    fs::write(dir.path().join("a.DDS"), b"old").unwrap();
    let opts = Options {
        overwrite: Overwrite::Always,
        ..Options::default()
    };
    let r = optimize(&png, &opts, |_| {}).unwrap();
    assert_eq!((r.done, r.failed.len()), (1, 0));
    assert_ne!(fs::read(dir.path().join("a.DDS")).unwrap(), b"old");
    assert!(!dir.path().join("a.dds").exists() && !dir.path().join("b.dds").exists());
    assert_eq!(clean(&png, &opts, |_| {}).unwrap().done, 1);
    assert!(!dir.path().join("a.DDS").exists());
}

#[test]
fn single_non_png_root_touches_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let notes = dir.path().join("notes.txt");
    fs::write(&notes, b"user").unwrap();
    fs::write(dir.path().join(".ddsify-1-2.tmp"), b"partial").unwrap();
    let r = clean(&notes, &Options::default(), |_| {}).unwrap();
    assert_eq!((r.done, r.failed.len()), (0, 0));
    assert!(dir.path().join(".ddsify-1-2.tmp").exists());
}

/// Case-insensitive file systems: `A.PNG` names the file stored as `a.png`.
#[cfg(windows)]
#[test]
fn single_png_root_in_other_case_is_encoded() {
    let dir = tempfile::tempdir().unwrap();
    write_png(&dir.path().join("a.png"), &gradient(8, 8, false));
    let r = optimize(&dir.path().join("A.PNG"), &Options::default(), |_| {}).unwrap();
    assert_eq!((r.done, r.failed.len()), (1, 0));
    assert!(dir.path().join("a.dds").is_file());
}
