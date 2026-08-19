//! The privacy permissions the capture backends need.
//!
//! macOS starts a capture session whether or not the permission behind it has
//! been granted; the session then delivers no frames and reports no error.
//! [`Camera::open`](crate::camera::Camera::open) and
//! [`ScreenCapture::open`](crate::screen::ScreenCapture::open) therefore resolve
//! the permission before starting, so a missing grant surfaces as an error
//! rather than as a capture that never produces a frame.

/// Whether this process may capture from a camera or a display.
///
/// The two frameworks report with different fidelity — see
/// [`Camera::authorization`](crate::camera::Camera::authorization) and
/// [`ScreenCapture::authorization`](crate::screen::ScreenCapture::authorization).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Authorization {
    /// The user has not been asked yet; the backend's `request_access` asks them.
    NotDetermined,
    /// Capture is permitted.
    Authorized,
    /// The user declined. Only they can reverse it, in System Settings.
    Denied,
    /// Policy forbids capture and the user cannot override it.
    Restricted,
}
