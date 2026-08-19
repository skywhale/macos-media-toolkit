//! Camera and screen capture plus hardware HEVC encode/decode for macOS.
//!
//! The crate wraps Apple's capture and codec frameworks — AVFoundation,
//! ScreenCaptureKit and VideoToolbox — behind small, blocking, frames-in /
//! frames-out APIs, with no runtime, no GPU API and no callbacks of its own.
//! It also carries [`hevc`], a portable HEVC bitstream module with no macOS
//! dependency at all.
//!
//! Frames cross the API boundary as [`BgraFrame`], a tightly packed 32-bit BGRA
//! buffer: the pixel format Apple's capture and codec hardware works in
//! natively, so no conversion pass is hidden inside the library.
//!
//! # Permissions
//!
//! Camera capture requires the Camera privacy permission, screen capture
//! requires Screen Recording (System Settings → Privacy & Security). Both are
//! requested by the OS on first use of the corresponding framework; without
//! them, opening a capture fails.

pub mod hevc;

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
