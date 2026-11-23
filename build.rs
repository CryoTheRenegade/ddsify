fn main() {
    // intel_tex_2's prebuilt kernels contain C++ objects but it only links the C++ runtime on
    // Windows (MSVC does it implicitly); without this, Linux fails with `__gxx_personality_v0`.
    // shortcut: macOS untested (assumed to need libc++ the same way); other Unix targets not handled.
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => println!("cargo:rustc-link-lib=stdc++"),
        Ok("macos") => println!("cargo:rustc-link-lib=c++"),
        _ => {}
    }
}
