//! Capture a display and save a frame.
//!
//! ```text
//! cargo run --example screen_capture       # main display
//! cargo run --example screen_capture -- 1  # second shareable display
//! ```
//!
//! Frames arrive only when the screen content changes, so this moves the mouse
//! or another window to produce one if the desktop is idle.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("screen capture requires macOS");
}

#[cfg(target_os = "macos")]
mod common;

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use macos_toolkit::{
        Authorization,
        screen::{ScreenCapture, ScreenCaptureConfig},
    };
    use std::time::Duration;

    env_logger::init();

    // A grant made at this prompt applies only to the next launch.
    let authorization = ScreenCapture::request_access();
    anyhow::ensure!(
        authorization == Authorization::Authorized,
        "screen recording access is {authorization:?}; grant it and run this again"
    );

    let display_index = std::env::args().nth(1).map(|a| a.parse()).transpose()?;

    let screen = ScreenCapture::open(&ScreenCaptureConfig {
        display_index,
        width: 1920,
        height: 1080,
        frame_rate: 30.0,
        shows_cursor: true,
        scales_to_fit: true,
    })?;
    println!("capturing {}", screen.display_description());

    // A static desktop delivers nothing, so poll until the content changes
    // rather than treating the first timeout as a failure.
    let mut frame = None;
    for _ in 0..50 {
        anyhow::ensure!(!screen.is_stopped(), "the OS stopped the stream");
        if let Some(f) = screen.take_frame(Duration::from_millis(100)) {
            frame = Some(f);
            break;
        }
        println!("no change on screen yet...");
    }

    let frame = frame.ok_or_else(|| anyhow::anyhow!("screen content never changed"))?;
    println!("captured {}x{}", frame.width, frame.height);

    common::write_ppm("screen.ppm", &frame)
}
