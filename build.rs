use std::{env, io, process::Command};
use winres::WindowsResource;

fn main() -> io::Result<()> {
    let timestamp = chrono::Utc::now().format("%Y%m%d.%H%M%S").to_string();
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", timestamp);

    let git_hash = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let git_hash_short = if git_hash.len() >= 7 {
        git_hash[..7].to_string()
    } else {
        git_hash.clone()
    };
    println!("cargo:rustc-env=GIT_HASH={}", git_hash);
    println!("cargo:rustc-env=GIT_HASH_SHORT={}", git_hash_short);

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "unknown".to_string());
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=TARGET_PLATFORM={}-{}", arch, os);

    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=BUILD_PROFILE={}", profile);

    let version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());
    println!("cargo:rustc-env=BUILD_STRING={}.{}", version, timestamp);

    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads/");

    if env::var_os("CARGO_CFG_WINDOWS").is_some() {
        WindowsResource::new()
            .set_icon("./assets/icon.ico")
            .compile()?;
    }

    Ok(())
}
