//! Camera and screen capture plus hardware HEVC encode/decode for macOS.
//!
//! The `camera`, `screen` and `videotoolbox` modules wrap AVFoundation,
//! ScreenCaptureKit and VideoToolbox behind blocking, frames-in / frames-out
//! APIs requiring no async runtime, no GPU context and no caller-supplied
//! callbacks. [`hevc`] holds portable HEVC bitstream tools and compiles on every
//! platform.
//!
//! Frames cross the API boundary as [`BgraFrame`], tightly packed 32-bit BGRA:
//! the format Apple's capture and codec hardware produces and consumes
//! natively, so the library performs no colorspace conversion.
//!
//! # Permissions
//!
//! Camera capture requires the Camera privacy permission, screen capture
//! requires Screen Recording (System Settings → Privacy & Security). The OS
//! requests both on first use of the corresponding framework; opening a capture
//! fails until they are granted.

pub mod hevc;

#[cfg(target_os = "macos")]
pub mod camera;
#[cfg(target_os = "macos")]
mod permission;
#[cfg(target_os = "macos")]
pub mod screen;
#[cfg(target_os = "macos")]
mod slot;

#[cfg(target_os = "macos")]
pub use permission::Authorization;
#[cfg(target_os = "macos")]
pub mod videotoolbox;

/// A tightly packed 32-bit BGRA frame (no row padding).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BgraFrame {
    /// `width * height * 4` bytes of BGRA pixels, top row first.
    pub bgra: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}
