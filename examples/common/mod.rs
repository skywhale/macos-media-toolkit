//! Helpers shared by the examples; not part of the library's API.

use anyhow::{Context, Result};
use macos_toolkit::BgraFrame;
use std::{fs, path::Path};

/// Write a frame as a binary PPM (P6), converting BGRA to RGB. Every image
/// viewer and `ffmpeg` read the format, and writing it needs no dependency.
pub fn write_ppm(path: impl AsRef<Path>, frame: &BgraFrame) -> Result<()> {
    let path = path.as_ref();

    let mut ppm = Vec::with_capacity(frame.bgra.len() / 4 * 3 + 32);
    ppm.extend_from_slice(format!("P6\n{} {}\n255\n", frame.width, frame.height).as_bytes());
    for pixel in frame.bgra.chunks_exact(4) {
        ppm.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
    }

    fs::write(path, &ppm).with_context(|| format!("writing {}", path.display()))?;
    println!("wrote {} ({} bytes)", path.display(), ppm.len());
    Ok(())
}
