# macos-media-toolkit

Rust interops for a few of Apple's media frameworks: AVFoundation camera
capture, ScreenCaptureKit screen capture, VideoToolbox HEVC encode and decode,
and a portable HEVC bitstream module.

The selection is arbitrary. Each piece exists because a project of mine needed
it, so coverage is narrow and the API follows those needs rather than the
frameworks' full surface. Anything not listed below is absent because nothing
has demanded it yet.

What is here wraps the frameworks behind blocking, frames-in / frames-out APIs:
a capture is opened and drained, and the codec exchanges byte buffers. No async
runtime, GPU context, actor framework or caller-supplied callback is involved.
Frames cross the boundary as `BgraFrame`, tightly packed 32-bit BGRA — the
format Apple's capture and codec hardware produces and consumes natively, so the
library performs no colorspace conversion.

- `camera` — an AVFoundation capture session on a chosen device, with explicit
  format and frame-rate pinning.
- `screen` — a ScreenCaptureKit stream capturing one of the enumerable displays.
- `videotoolbox` — hardware HEVC encode and decode, speaking Annex B with
  in-band parameter sets on keyframes.
- `hevc` — Annex B framing, parameter-set extraction, and HEVC slice-header
  parsing and rewriting. Pure Rust: this module compiles on every platform, not
  just macOS.

## Status

0.1, and a personal project. The API is unstable and will keep changing as more
of it gets used.

Some of what looks over-elaborate here is load-bearing. AVFoundation ignores a
frame-rate request unless you also pick the capture format yourself;
VideoToolbox refuses frames that NVDEC would have decoded; a capture started
without permission returns no frames and no error. Each of those is explained
where it is handled.

## Camera

```rust
use macos_media_toolkit::{Authorization, camera::{Camera, CameraConfig}};
use std::time::Duration;

// AVFoundation delivers no frames at all without permission, so resolve it
// first; macOS asks once and remembers the answer.
assert_eq!(Camera::request_access(Duration::from_secs(60)), Authorization::Authorized);

let camera = Camera::open(&CameraConfig {
    device_name: None, // Or a substring of a name from `Camera::device_names()`.
    width: 1280,
    height: 720,
    frame_rate: 30.0,
})?;

loop {
    // Budget more than one frame interval: frames arrive with a few ms of
    // dispatch-queue scheduling jitter.
    let Some(frame) = camera.take_frame(Duration::from_millis(100)) else {
        continue; // No frame in time.
    };
    println!("{}x{}, {} bytes", frame.width, frame.height, frame.bgra.len());
}
```

Only the newest frame is kept. A frame you do not take before the next one
arrives is thrown away, so a slow reader skips frames rather than working
through a growing queue of stale ones.

## Screen

```rust
use macos_media_toolkit::{Authorization, screen::{ScreenCapture, ScreenCaptureConfig}};
use std::time::Duration;

// Screen Recording is granted per launch: a user who accepts this prompt
// authorizes the *next* run, not this one.
assert_eq!(ScreenCapture::request_access(), Authorization::Authorized);

let screen = ScreenCapture::open(&ScreenCaptureConfig {
    display_index: None, // Main display.
    width: 1920,
    height: 1080,
    frame_rate: 30.0,
    shows_cursor: true,
    scales_to_fit: true, // Letterbox rather than stretch.
})?;

let frame = screen.take_frame(Duration::from_millis(100));
```

ScreenCaptureKit delivers a frame only when the screen content changes, so
`take_frame` returns `None` on a static screen regardless of the timeout.
Consumers requiring a steady cadence must supply their own gap policy;
re-serving the last frame is the usual one. A stream the OS stops (display
disconnected, permission revoked) never recovers, so `is_stopped()` reports the
condition and the capture must be reopened.

## Encode and decode

```rust
use macos_media_toolkit::videotoolbox::{DecoderConfig, EncoderConfig, HevcDecoder, HevcEncoder};

let mut encoder = HevcEncoder::new(&EncoderConfig {
    width: 1280,
    height: 720,
    frame_rate: 30.0,
    average_bitrate_bps: 8_000_000,
    keyframe_interval: 60,
})?;
let mut decoder = HevcDecoder::new(&DecoderConfig::default());

let encoded = encoder.encode(&frame.bgra, frame.width, frame.height, false)?;
assert!(!encoded.annex_b.is_empty());

let decoded = decoder.decode(&encoded.annex_b)?;
assert_eq!(decoded.frame.width, frame.width);
```

Both directions are synchronous with a single frame in flight, so neither
requires a thread or a runtime. The encoder emits Annex B with in-band
VPS/SPS/PPS on keyframes; a new resolution passed to `encode` recreates the
session and makes that frame a keyframe. The decoder builds its session lazily
from the first keyframe's parameter sets and returns
`DecodeError::MissingKeyframe` until one arrives.

### VideoToolbox vs NVDEC loss behavior

The two hardware decoders disagree about missing references, and the difference
is fatal over a lossy link.

When a P-slice's reference picture set names a picture that is missing from the
decoded picture buffer — lost in transport, or skipped while the decoder waited
for a recovery keyframe — NVDEC conceals the missing reference and decodes an
artifacted frame. VideoToolbox rejects the slice outright with
`kVTVideoDecoderBadDataErr` (-12909). The consequence is a livelock: after each
recovery keyframe, the first P-slice predicts from an intermediate picture the
decoder never decoded, fails, and arms the next keyframe request — forever.

`HevcDecoder` therefore tracks the picture order counts presumed to be in the
hardware DPB and, before submitting an access unit, repairs slices naming
pictures it never decoded: a missing *used* reference is remapped to the newest
picture actually in the DPB, and keep-alive entries for pictures that were never
decoded are dropped. Every other header field and the whole slice payload are
preserved byte for byte. The repair is fail-open — a slice the parser does not
understand is submitted untouched, falling back to ordinary keyframe recovery —
and can be turned off with `DecoderConfig { conceal_missing_references: false }`.

The machinery behind this lives in the `hevc` module and is usable on its own on
any platform: `parse_sps`, `parse_pps`, `parse_slice` and `rewrite_rps`, plus
the Annex B framing helpers.

## Examples

```text
cargo run --example camera_capture             # save a camera frame as PPM
cargo run --example camera_capture -- FaceTime # select a device by name
cargo run --example screen_capture             # save a display frame as PPM
cargo run --example encode_decode              # encode and decode, writing encoded.h265
cargo run --example hevc_dump -- encoded.h265  # print a stream's NAL structure
```

`hevc_dump` uses only the `hevc` module and runs on any platform. The others
need macOS and the corresponding permission; set `RUST_LOG=info` to see the
library's diagnostics.

## Prior art

Much of this space is covered elsewhere, sometimes better. These are the notes
from deciding whether to write any of it, and they should point you at a
narrower crate when one fits.

- Screen capture is the crowded corner:
  [screencapturekit](https://github.com/doom-fish/screencapturekit-rs) is
  actively maintained and exposes the same ScreenCaptureKit controls. Prefer it
  unless you want the camera and codec pieces under one delivery model.
- Camera capture with working frame-rate control is not covered:
  [nokhwa](https://github.com/l1npengtul/nokhwa)'s AVFoundation backend
  documents its FPS adjustment as non-functional, and the rest of the field is
  dormant or raw bindings.
- VideoToolbox HEVC wrappers exist
  ([shiguredo_video_toolbox](https://github.com/shiguredo/video-toolbox-rs),
  [videotoolbox](https://github.com/doom-fish/videotoolbox-rs)) but none offer
  synchronous Annex B in/out with in-band parameter sets — the shape a network
  streaming pipeline wants.
- HEVC slice-header parsing with bit-exact *rewriting* is the one part I found
  no substitute for; [hevc_parser](https://github.com/quietvoid/hevc_parser)
  stops parsing before the slice-header RPS and is parse-only.

The Apple bindings underneath are [cidre](https://github.com/yury/cidre) —
actively developed and proven in shipping software, but a single-maintainer
project with a moving API; this crate pins a compatible version and absorbs
churn at upgrades.

## Requirements

- macOS 12.3 or later (ScreenCaptureKit); VideoToolbox HEVC encoding needs
  hardware support, which every Apple silicon Mac and every Intel Mac since
  Skylake has.
- The Camera privacy permission for `camera`, and Screen Recording for `screen`
  (System Settings → Privacy & Security). Both backends expose `authorization()`
  and `request_access()` returning the same `Authorization`, and both `open()`
  calls fail with an actionable error rather than returning a capture that never
  produces a frame. Two platform quirks are worth knowing: macOS can only prompt
  a process it can attribute the prompt to, so a bare binary run from a
  non-interactive shell may be refused without any prompt appearing; and a
  Screen Recording grant takes effect only at the next launch.
- The `hevc` module has none of the above requirements and builds on any
  platform.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
