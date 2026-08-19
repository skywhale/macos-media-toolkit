//! ScreenCaptureKit screen capture.
//!
//! ScreenCaptureKit only delivers a new sample buffer when the screen content
//! actually changes, so a static screen produces no frames at all. Consumers
//! that need a steady cadence (an encoder, say) must decide what to do with the
//! gap — commonly, re-serving the last frame.
//!
//! Screen Recording permission (System Settings → Privacy & Security → Screen
//! Recording) is required; without it enumeration and stream start fail.

// `useless_transmute` is emitted inside cidre's `define_obj_type!` expansion.
#![expect(clippy::useless_transmute)]

use crate::{
    BgraFrame,
    slot::{FrameSlot, store_frame},
};
use anyhow::Result;
use cidre::{
    arc, cm, cv, define_obj_type, dispatch, ns, objc, sc,
    sc::stream::{Delegate as _, Output as _},
};
use log::*;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

/// Appended to capture errors that are most likely a missing-permission symptom.
const PERMISSION_HINT: &str = "Screen Recording permission may be missing (System Settings → \
                               Privacy & Security → Screen Recording).";

/// How to open a [`ScreenCapture`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenCaptureConfig {
    /// Zero-based index into the shareable-displays list; `None` captures the
    /// main display.
    pub display_index: Option<usize>,
    /// Width of the delivered frames, in pixels.
    pub width: u32,
    /// Height of the delivered frames, in pixels.
    pub height: u32,
    /// Upper bound on the delivery rate; content changes are still what triggers
    /// a frame.
    pub frame_rate: f64,
    /// Draw the mouse cursor into the captured frames.
    pub shows_cursor: bool,
    /// Letterbox instead of stretching when the display aspect differs from
    /// width:height.
    pub scales_to_fit: bool,
}

/// Wrapper letting us move a retained ScreenCaptureKit object out of the
/// enumeration callback (which runs on a GCD queue) back to the calling thread.
struct SendCell<T>(T);
// SAFETY: the wrapped value is a retained, immutable ScreenCaptureKit snapshot.
// It is produced on a GCD queue and handed to exactly one other thread through a
// channel; only one thread ever touches it at a time, so moving it across the
// thread boundary is sound.
unsafe impl<T> Send for SendCell<T> {}

/// Inner state owned by the Objective-C output object. Shares the latest-frame
/// slot with the consumer's thread.
struct StreamOutputHandlerInner {
    slot: Arc<FrameSlot>,
}

define_obj_type!(
    StreamOutputHandler + sc::stream::OutputImpl,
    StreamOutputHandlerInner,
    STREAM_OUTPUT_HANDLER
);

impl sc::stream::Output for StreamOutputHandler {}

#[objc::add_methods]
impl sc::stream::OutputImpl for StreamOutputHandler {
    // Runs on the stream's dispatch queue.
    extern "C" fn impl_stream_did_output_sample_buf(
        &mut self,
        _cmd: Option<&objc::Sel>,
        _stream: &sc::Stream,
        sample_buf: &mut cm::SampleBuf,
        kind: sc::OutputType,
    ) {
        if kind != sc::OutputType::Screen {
            return;
        }

        store_frame(&self.inner_mut().slot, sample_buf);
    }
}

/// Inner state owned by the Objective-C stream delegate.
struct StreamEventDelegateInner {
    stopped: Arc<AtomicBool>,
}

define_obj_type!(
    StreamEventDelegate + sc::stream::DelegateImpl,
    StreamEventDelegateInner,
    STREAM_EVENT_DELEGATE
);

impl sc::stream::Delegate for StreamEventDelegate {}

#[objc::add_methods]
impl sc::stream::DelegateImpl for StreamEventDelegate {
    extern "C" fn impl_stream_did_stop_with_err(
        &mut self,
        _cmd: Option<&objc::Sel>,
        _stream: &sc::Stream,
        error: &ns::Error,
    ) {
        error!("ScreenCaptureKit stream stopped with error: {error}");
        self.inner_mut().stopped.store(true, Ordering::Relaxed);
    }
}

/// A running ScreenCaptureKit stream delivering packed 32BGRA frames.
///
/// Keeps the stream, output handler, delegate and queue alive for its own
/// lifetime; the stream is stopped on drop.
pub struct ScreenCapture {
    display_desc: String,

    slot: Arc<FrameSlot>,
    stopped: Arc<AtomicBool>,

    // Objects that must outlive the stream.
    stream: arc::R<sc::Stream>,
    _output_handler: arc::R<StreamOutputHandler>,
    _delegate: arc::R<StreamEventDelegate>,
    _queue: arc::R<dispatch::Queue>,
}

impl ScreenCapture {
    /// Open a display capture stream and start delivering frames.
    pub fn open(config: &ScreenCaptureConfig) -> Result<ScreenCapture> {
        let ScreenCaptureConfig {
            width,
            height,
            frame_rate,
            shows_cursor,
            scales_to_fit,
            ..
        } = *config;

        // ScreenCaptureKit enumeration is async-only; bridge it to this thread
        // through a channel.
        let (tx, rx) = mpsc::channel();
        sc::ShareableContent::current_with_ch(move |content, err| {
            let result = match content {
                Some(c) => Ok(SendCell(c.retained())),
                None => Err(err
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "no shareable content".to_string())),
            };
            let _ = tx.send(result);
        });

        let content = rx
            .recv_timeout(Duration::from_secs(10))
            .map_err(|e| {
                anyhow::anyhow!("timed out querying shareable content ({e}). {PERMISSION_HINT}")
            })?
            .map_err(|e| anyhow::anyhow!("querying shareable content: {e}"))?
            .0;

        let displays = content.displays();
        let display_count = displays.len();
        let index = config.display_index.unwrap_or(0);
        let display = displays.get(index).map_err(|_| {
            anyhow::anyhow!(
                "display index {index} out of range; {display_count} shareable display(s) \
                 available"
            )
        })?;

        let display_desc = format!(
            "display {:?} ({}x{})",
            display.display_id(),
            display.width(),
            display.height()
        );

        let filter = sc::ContentFilter::with_display_excluding_windows(&display, &ns::Array::new());

        let mut cfg = sc::StreamCfg::new();
        cfg.set_width(width as usize);
        cfg.set_height(height as usize);
        cfg.set_pixel_format(cv::PixelFormat::_32_BGRA);
        cfg.set_minimum_frame_interval(cm::Time::new(1, frame_rate.round().max(1.0) as i32));
        cfg.set_shows_cursor(shows_cursor);
        cfg.set_scales_to_fit(scales_to_fit);

        let slot = Arc::new(FrameSlot::new());
        let stopped = Arc::new(AtomicBool::new(false));

        let output_handler = StreamOutputHandler::with(StreamOutputHandlerInner {
            slot: Arc::clone(&slot),
        });
        let delegate = StreamEventDelegate::with(StreamEventDelegateInner {
            stopped: Arc::clone(&stopped),
        });

        let queue = dispatch::Queue::serial_with_ar_pool();
        let stream = sc::Stream::with_delegate(&filter, &cfg, delegate.as_ref());
        stream
            .add_stream_output(
                output_handler.as_ref(),
                sc::OutputType::Screen,
                Some(&queue),
            )
            .map_err(|e| anyhow::anyhow!("adding screen stream output: {e:?}"))?;

        // Start synchronously, bridging the async completion handler.
        let (tx, rx) = mpsc::channel();
        stream.start_with_ch(move |err| {
            let _ = tx.send(err.map(|e| e.to_string()));
        });
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(None) => {}
            Ok(Some(err)) => {
                anyhow::bail!("starting ScreenCaptureKit stream: {err}. {PERMISSION_HINT}")
            }
            Err(e) => {
                anyhow::bail!("timed out starting ScreenCaptureKit stream ({e}). {PERMISSION_HINT}")
            }
        }

        info!(
            "Started ScreenCaptureKit capture of {display_desc} at \
             {width}x{height}@{frame_rate}fps"
        );

        Ok(Self {
            display_desc,
            slot,
            stopped,
            stream,
            _output_handler: output_handler,
            _delegate: delegate,
            _queue: queue,
        })
    }

    /// Block up to `timeout` for the next frame.
    ///
    /// ScreenCaptureKit delivers a frame only when the screen content changes,
    /// so a static screen yields `None` however long the timeout is. Delivery is
    /// latest-frame, drop-old: a frame that is not taken before the next one
    /// arrives is overwritten.
    pub fn take_frame(&self, timeout: Duration) -> Option<BgraFrame> {
        self.slot.take_blocking(timeout)
    }

    /// True once the OS stopped the stream (display disconnect, permission
    /// revoked); the capture cannot recover — recreate it.
    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }

    /// Human-readable description of the captured display, e.g.
    /// "display 1 (3456x2234)".
    pub fn display_description(&self) -> &str {
        &self.display_desc
    }
}

impl Drop for ScreenCapture {
    fn drop(&mut self) {
        // Fire-and-forget; we don't wait for the async stop to complete.
        self.stream.stop_with_ch(|_| {});
    }
}
