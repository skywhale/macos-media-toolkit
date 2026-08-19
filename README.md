# macos-toolkit

Camera and screen capture plus hardware HEVC encode/decode for macOS
(AVFoundation, ScreenCaptureKit, VideoToolbox), with portable HEVC bitstream
tools.

The crate wraps Apple's media frameworks behind small, blocking, frames-in /
frames-out APIs. There is no runtime, no GPU API, no actor framework and no
callback of the crate's own: you open a capture and take frames from it, or you
hand the codec bytes and get bytes back. Frames cross the boundary as
`BgraFrame`, tightly packed 32-bit BGRA — the format Apple's capture and codec
hardware works in natively, so no conversion pass is hidden inside the library.

- `camera` — an AVFoundation capture session on a chosen device, with explicit
  format and frame-rate pinning.
- `screen` — a ScreenCaptureKit stream capturing one display.
- `videotoolbox` — hardware HEVC encode and decode, speaking Annex B with
  in-band parameter sets on keyframes.
- `hevc` — Annex B framing, parameter-set extraction, and HEVC slice-header
  parsing and rewriting. Pure Rust: this module compiles on every platform, not
  just macOS.

## Status

0.1, extracted from code running in production at [tonari](https://tonari.no).
The implementation is proven on real hardware; the API is not stable yet and
will change as more of it is used outside its original home.

## Camera

```rust
use macos_toolkit::camera::{Camera, CameraConfig};
use std::time::Duration;

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

Delivery is latest-frame, drop-old: a frame that is not taken before the next
one arrives is overwritten, so a slow consumer falls behind in latency, never in
backlog.

## Screen

```rust
use macos_toolkit::screen::{ScreenCapture, ScreenCaptureConfig};
use std::time::Duration;

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
`take_frame` on a static screen returns `None` however long you wait. Consumers
that need a steady cadence — an encoder, say — must decide what to do with the
gap; re-serving the last frame is the usual answer. If the OS stops the stream
(display disconnected, permission revoked) it never recovers: check
`is_stopped()` and open a new capture.

## Encode and decode

```rust
use macos_toolkit::videotoolbox::{DecoderConfig, EncoderConfig, HevcDecoder, HevcEncoder};

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

Both directions are synchronous with a single frame in flight, so neither needs
a thread or a runtime. The encoder emits Annex B with in-band VPS/SPS/PPS on
keyframes; passing `encode` a new resolution transparently recreates the session
and makes that frame a keyframe. The decoder builds its session lazily from the
first keyframe's parameter sets, and returns `DecodeError::MissingKeyframe`
until one arrives.

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

`HevcDecoder` therefore tracks the picture order counts it believes are in the
hardware DPB and, before submitting an access unit, repairs slices that name
pictures it never decoded: a missing *used* reference is remapped to the newest
picture actually in the DPB, and keep-alive entries for pictures that were never
decoded are dropped. Every other header field and the whole slice payload are
preserved byte for byte. The repair is fail-open — a slice the parser does not
understand is submitted untouched, falling back to ordinary keyframe recovery —
and can be turned off with `DecoderConfig { conceal_missing_references: false }`.

The machinery behind this lives in the `hevc` module and is usable on its own,
on any platform: `parse_sps`, `parse_pps`, `parse_slice` and `rewrite_rps`, plus
the Annex B framing helpers.

## Prior art

Parts of this space are covered elsewhere; this crate exists for the parts that
weren't, and for having one delivery model across all of them.

- Screen capture is the crowded corner:
  [screencapturekit](https://github.com/doom-fish/screencapturekit-rs) is
  actively maintained and exposes the same ScreenCaptureKit knobs. Use it if
  screen capture is all you need.
- Camera capture with working frame-rate control is not covered:
  [nokhwa](https://github.com/l1npengtul/nokhwa)'s AVFoundation backend
  documents its FPS adjustment as non-functional, and the rest of the field is
  dormant or raw bindings.
- VideoToolbox HEVC wrappers exist
  ([shiguredo_video_toolbox](https://github.com/shiguredo/video-toolbox-rs),
  [videotoolbox](https://github.com/doom-fish/videotoolbox-rs)) but none offer
  synchronous Annex B in/out with in-band parameter sets — the shape a network
  streaming pipeline wants.
- HEVC slice-header parsing with bit-exact *rewriting* exists nowhere else that
  we could find; [hevc_parser](https://github.com/quietvoid/hevc_parser) stops
  parsing before the slice-header RPS and is parse-only.

The Apple bindings underneath are [cidre](https://github.com/yury/cidre) —
actively developed and proven in shipping software, but a single-maintainer
project with a moving API; this crate pins a compatible version and absorbs
churn at upgrades.

## Requirements

- macOS 12.3 or later (ScreenCaptureKit); VideoToolbox HEVC encoding needs
  hardware support, which every Apple silicon Mac and every Intel Mac since
  Skylake has.
- The Camera privacy permission for `camera`, and Screen Recording for `screen`
  (System Settings → Privacy & Security). Without them, opening a capture fails.
- The `hevc` module has none of the above requirements and builds on any
  platform.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
