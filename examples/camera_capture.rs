//! Open a camera, report the format it settled on, and save a frame.
//!
//! ```text
//! cargo run --example camera_capture               # default device
//! cargo run --example camera_capture -- FaceTime   # by name substring
//! ```

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("camera capture requires macOS");
}

#[cfg(target_os = "macos")]
mod common;

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use macos_media_toolkit::{
        Authorization,
        camera::{Camera, CameraConfig},
    };
    use std::time::{Duration, Instant};

    env_logger::init();

    // The first run prompts; afterwards macOS returns the standing answer.
    let authorization = Camera::request_access(Duration::from_secs(60));
    anyhow::ensure!(
        authorization == Authorization::Authorized,
        "camera access is {authorization:?}"
    );

    println!("cameras: {:?}", Camera::device_names());

    let config = CameraConfig {
        device_name: std::env::args().nth(1),
        width: 1280,
        height: 720,
        frame_rate: 30.0,
    };
    let camera = Camera::open(&config)?;

    println!("opened '{}'", camera.device_name());
    if let Some(format) = camera.active_format() {
        println!(
            "active format: {}x{}, up to {:.0} fps",
            format.width, format.height, format.max_frame_rate
        );
    }

    // Three frame intervals of slack absorb the dispatch-queue jitter described
    // on `take_frame`; the first frame additionally waits for the session to
    // start delivering.
    const FRAMES: usize = 60;
    let timeout = Duration::from_secs_f64(3.0 / config.frame_rate);
    let mut last = None;
    let mut dropped = 0;
    let started = Instant::now();

    for _ in 0..FRAMES {
        match camera.take_frame(timeout) {
            Some(frame) => last = Some(frame),
            None => dropped += 1,
        }
    }

    let frame =
        last.ok_or_else(|| anyhow::anyhow!("no frame arrived in {:?}", started.elapsed()))?;
    println!(
        "captured {} frames in {:.2}s ({dropped} timed out), last frame {}x{}",
        FRAMES - dropped,
        started.elapsed().as_secs_f64(),
        frame.width,
        frame.height
    );

    common::write_ppm("camera.ppm", &frame)
}
