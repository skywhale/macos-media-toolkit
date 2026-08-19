//! AVFoundation camera capture.
//!
//! The delegate runs on a background dispatch queue and only copies each 32BGRA
//! frame into a latest-frame slot, which [`Camera::take_frame`] drains on the
//! caller's thread.

// `useless_transmute` is emitted inside cidre's `define_obj_type!` expansion.
#![expect(clippy::useless_transmute)]

use crate::{
    Authorization, BgraFrame,
    slot::{FrameSlot, store_frame},
};
use anyhow::{Context as _, Result, bail};
use cidre::{
    arc, av,
    av::capture::{VideoDataOutputSampleBufDelegate, VideoDataOutputSampleBufDelegateImpl},
    blocks, cm, cv, define_obj_type, dispatch, ns, objc,
    objc::Obj as _,
};
use log::*;
use std::{sync::Arc, sync::mpsc, time::Duration};

/// How to open a [`Camera`].
#[derive(Debug, Clone, PartialEq)]
pub struct CameraConfig {
    /// Substring of the device's localized name; `None` selects the system
    /// default device.
    pub device_name: Option<String>,
    /// Width of the delivered frames, in pixels.
    pub width: u32,
    /// Height of the delivered frames, in pixels.
    pub height: u32,
    /// Frame rate to pin the device to.
    pub frame_rate: f64,
}

/// A capture format of a camera device.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraFormat {
    /// Format width in pixels.
    pub width: u32,
    /// Format height in pixels.
    pub height: u32,
    /// The highest frame rate the format can deliver.
    pub max_frame_rate: f64,
}

/// Inner state owned by the Objective-C delegate object. Shares the latest-frame
/// slot with the consumer's thread.
struct CameraDelegateInner {
    slot: Arc<FrameSlot>,
}

define_obj_type!(
    CameraDelegate + VideoDataOutputSampleBufDelegateImpl,
    CameraDelegateInner,
    CAMERA_DELEGATE
);

impl VideoDataOutputSampleBufDelegate for CameraDelegate {}

#[objc::add_methods]
impl VideoDataOutputSampleBufDelegateImpl for CameraDelegate {
    // Runs on the delegate's dispatch queue.
    extern "C" fn impl_capture_output_did_output_sample_buf_from_connection(
        &mut self,
        _cmd: Option<&objc::Sel>,
        _output: &av::CaptureOutput,
        sample_buf: &cm::SampleBuf,
        _connection: &av::CaptureConnection,
    ) {
        store_frame(&self.inner_mut().slot, sample_buf);
    }
}

/// A running AVFoundation capture session delivering packed 32BGRA frames.
///
/// Keeps the session, delegate, output, input and device alive for its own
/// lifetime; the session is stopped on drop.
pub struct Camera {
    device_name: String,

    slot: Arc<FrameSlot>,

    // Objects that must outlive the session. `AVCaptureVideoDataOutput` holds its
    // delegate weakly, hence the strong reference here.
    session: arc::R<av::CaptureSession>,
    device: arc::R<av::CaptureDevice>,
    _input: arc::R<av::CaptureDeviceInput>,
    _output: arc::R<av::CaptureVideoDataOutput>,
    _delegate: arc::R<CameraDelegate>,
    _queue: arc::R<dispatch::Queue>,
}

impl Camera {
    /// Whether this process may capture from a camera. AVFoundation
    /// distinguishes all four states.
    pub fn authorization() -> Authorization {
        match av::CaptureDevice::authorization_status_for_media_type(av::MediaType::video()) {
            Ok(av::AuthorizationStatus::Authorized) => Authorization::Authorized,
            Ok(av::AuthorizationStatus::Denied) => Authorization::Denied,
            Ok(av::AuthorizationStatus::Restricted) => Authorization::Restricted,
            Ok(av::AuthorizationStatus::NotDetermined) => Authorization::NotDetermined,
            Err(e) => {
                warn!("could not query camera authorization status: {e:?}");
                Authorization::NotDetermined
            }
        }
    }

    /// Ask the user for camera access, blocking up to `timeout` for an answer,
    /// and report the resulting authorization.
    ///
    /// macOS asks once and remembers the answer, so subsequent calls return the
    /// standing decision without prompting. A bundled application must declare
    /// `NSCameraUsageDescription` in its Info.plist; the system terminates
    /// processes that request access without it. A bare binary that macOS cannot
    /// attribute a prompt to — one launched from a non-interactive shell, say —
    /// is refused outright and stays [`Authorization::NotDetermined`].
    pub fn request_access(timeout: Duration) -> Authorization {
        let status = Self::authorization();
        if status != Authorization::NotDetermined {
            return status;
        }

        let (tx, rx) = mpsc::channel();
        let mut block = blocks::SendBlock::<fn(bool)>::new1(move |_granted: bool| {
            let _ = tx.send(());
        });
        match av::CaptureDevice::request_access_for_media_type_ch(
            av::MediaType::video(),
            &mut block,
        ) {
            Ok(()) => {
                let _ = rx.recv_timeout(timeout);
            }
            Err(e) => warn!("could not request camera access: {e:?}"),
        }

        Self::authorization()
    }

    /// Localized names of the available video capture devices.
    pub fn device_names() -> Vec<String> {
        discovered_devices()
            .iter()
            .map(|d| d.localized_name().to_string())
            .collect()
    }

    /// Open a camera and start delivering frames.
    pub fn open(config: &CameraConfig) -> Result<Camera> {
        let CameraConfig {
            width,
            height,
            frame_rate,
            ..
        } = *config;

        match Self::authorization() {
            Authorization::Authorized => {}
            Authorization::NotDetermined => {
                bail!("camera access has not been granted; call Camera::request_access first")
            }
            Authorization::Denied => bail!(
                "camera access was denied; grant it in System Settings → Privacy & \
                 Security → Camera"
            ),
            Authorization::Restricted => {
                bail!("camera access is restricted by policy and cannot be granted")
            }
        }

        let mut device = find_device(config.device_name.as_deref())?;
        let device_name = device.localized_name().to_string();

        let input = av::CaptureDeviceInput::with_device(&device)
            .map_err(|e| anyhow::anyhow!("creating capture device input: {e:?}"))?;

        let mut output = av::CaptureVideoDataOutput::new();
        output.set_always_discard_late_video_frames(true);
        output.set_automatically_configures_output_buf_dims(false);
        let settings = ns::Dictionary::<ns::String, ns::Id>::with_keys_values(
            &[
                cv::pixel_buffer_keys::pixel_format().as_ns(),
                cv::pixel_buffer_keys::width().as_ns(),
                cv::pixel_buffer_keys::height().as_ns(),
            ],
            &[
                cv::PixelFormat::_32_BGRA.to_ns_number().as_id_ref(),
                ns::Number::with_i32(width as i32).as_id_ref(),
                ns::Number::with_i32(height as i32).as_id_ref(),
            ],
        );
        output.set_video_settings(Some(&settings)).map_err(|e| {
            anyhow::anyhow!("setting 32BGRA {width}x{height} video settings: {e:?}")
        })?;

        let slot = Arc::new(FrameSlot::new());
        let queue = dispatch::Queue::serial_with_ar_pool();
        let delegate = CameraDelegate::with(CameraDelegateInner {
            slot: Arc::clone(&slot),
        });
        output.set_sample_buf_delegate(Some(delegate.as_ref()), Some(&queue));

        let mut session = av::CaptureSession::new();
        let mut added_input = false;
        let mut added_output = false;
        session.configure(|s| {
            if s.can_add_input(&input) {
                s.add_input(&input);
                added_input = true;
            }
            if s.can_add_output(&output) {
                s.add_output(&output);
                added_output = true;
            }
        });
        anyhow::ensure!(
            added_input,
            "camera input could not be added to the capture session"
        );
        anyhow::ensure!(
            added_output,
            "video output could not be added to the capture session"
        );

        session.start_running();

        // The auto-negotiated format can top out below the configured rate, which
        // makes the frame-duration pin fail and leaves the camera silently
        // delivering its default rate; select a rate-capable format first. Every
        // step below is best-effort, and failure leaves the default rate in place.
        {
            let selected = select_format(&device, width, height, frame_rate);

            // Override the active format only when doing so gains the configured
            // rate: forcing a same-rate format can stop frame delivery entirely
            // (see the pixel-subtype preference in `select_format`).
            let (activate_format, pin_rate) = match &selected {
                Some((format, max_rate, true)) => (Some(format), frame_rate.min(*max_rate)),
                Some((_, max_rate, false)) => {
                    warn!(
                        "AVFoundation camera '{device_name}' has no {width}x{height} format \
                         supporting {frame_rate}fps; best available tops out at {max_rate}fps"
                    );
                    (None, *max_rate)
                }
                None => {
                    warn!(
                        "AVFoundation camera '{device_name}' exposes no format covering \
                         {width}x{height}; leaving the auto-negotiated format active"
                    );
                    (None, frame_rate)
                }
            };
            // Floor so the pinned rate never exceeds the format's maximum.
            let fps = pin_rate.floor().max(1.0) as i32;
            let frame_duration = cm::Time::new(1, fps);
            match device.config_lock() {
                Ok(mut lock) => {
                    if let Some(format) = activate_format {
                        lock.set_active_format(format);
                    }
                    if let Err(e) = lock.set_active_video_min_frame_duration(frame_duration) {
                        warn!("Could not set camera min frame duration: {e:?}");
                    }
                    if let Err(e) = lock.set_active_video_max_frame_duration(frame_duration) {
                        warn!("Could not set camera max frame duration: {e:?}");
                    }
                }
                Err(e) => warn!("Could not lock camera to set frame rate: {e:?}"),
            }
        }

        info!(
            "Started AVFoundation capture session for '{device_name}' at \
             {width}x{height}@{frame_rate}fps"
        );

        Ok(Self {
            device_name,
            slot,
            session,
            device,
            _input: input,
            _output: output,
            _delegate: delegate,
            _queue: queue,
        })
    }

    /// Block up to `timeout` for the next frame.
    ///
    /// Delivery is latest-frame, drop-old: a frame that is not taken before the
    /// next one arrives is overwritten. Frames arrive with a few milliseconds of
    /// dispatch-queue scheduling jitter, so a timeout of exactly one frame
    /// interval routinely expires just before the next frame lands — budget more
    /// than one interval.
    pub fn take_frame(&self, timeout: Duration) -> Option<BgraFrame> {
        self.slot.take_blocking(timeout)
    }

    /// The device's currently active format, for diagnostics.
    pub fn active_format(&self) -> Option<CameraFormat> {
        self.device.active_format().map(|f| {
            let dims = f.format_desc().dims();
            CameraFormat {
                width: dims.width as u32,
                height: dims.height as u32,
                max_frame_rate: format_max_frame_rate(&f),
            }
        })
    }

    /// The resolved localized device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        self.session.stop_running();
    }
}

/// The highest frame rate a device format can actually deliver, across all of
/// its supported frame-rate ranges (0.0 if it advertises none).
fn format_max_frame_rate(format: &av::CaptureDeviceFormat) -> f64 {
    format
        .video_supported_frame_rate_ranges()
        .iter()
        .map(|r| r.max_frame_rate())
        .fold(0.0_f64, f64::max)
}

/// Tolerance when comparing a format's maximum frame rate against the
/// configured rate (ranges are floats like 59.94).
const RATE_TOLERANCE_FPS: f64 = 0.5;

/// A device format, the highest frame rate it supports, and whether that reaches
/// the requested rate.
type FormatChoice = (arc::R<av::CaptureDeviceFormat>, f64, bool);

/// Lexicographic ranking key for a candidate format; smaller is better.
type FormatRank = (u8, i64, u8);

/// Pick the device format that best matches the requested size and frame rate.
///
/// Returns the chosen format, the highest frame rate it supports, and whether
/// that reaches the requested rate. Rate capability ranks first: a rate-capable
/// format wins even at larger dimensions, because the output scaler delivers the
/// configured size from any covering format. Ties break on the smallest covering
/// area, then on matching the active format's pixel subtype, which the running
/// session is already configured for. `None` when no format covers the
/// resolution.
fn select_format(
    device: &av::CaptureDevice,
    width: u32,
    height: u32,
    frame_rate: f64,
) -> Option<FormatChoice> {
    let req_w = width as i32;
    let req_h = height as i32;
    let active_subtype = device
        .active_format()
        .map(|f| f.format_desc().media_sub_type());

    let mut best: Option<(FormatRank, FormatChoice)> = None;
    for format in device.formats().iter() {
        let desc = format.format_desc();
        let dims = desc.dims();
        if dims.width < req_w || dims.height < req_h {
            continue;
        }
        let max_rate = format_max_frame_rate(format);
        let rate_ok = max_rate + RATE_TOLERANCE_FPS >= frame_rate;
        let subtype_ok = active_subtype == Some(desc.media_sub_type());

        let area = dims.width as i64 * dims.height as i64;
        let key = (!rate_ok as u8, area, !subtype_ok as u8);
        if best.as_ref().is_none_or(|(k, _)| key < *k) {
            best = Some((key, (format.retained(), max_rate, rate_ok)));
        }
    }

    best.map(|(_, choice)| choice)
}

/// The video capture devices a discovery session exposes: built-in and external
/// cameras.
fn discovered_devices() -> arc::R<ns::Array<av::CaptureDevice>> {
    let mut types = ns::ArrayMut::with_capacity(2);
    types.push(av::CaptureDeviceType::built_in_wide_angle_camera());
    types.push(av::CaptureDeviceType::external());
    let session = av::capture::DiscoverySession::with_device_types_media_and_pos(
        &types,
        Some(av::MediaType::video()),
        av::CaptureDevicePos::Unspecified,
    );
    session.devices()
}

/// Find the capture device by (localized) name, or the system default video device.
fn find_device(device_name: Option<&str>) -> Result<arc::R<av::CaptureDevice>> {
    let Some(name) = device_name else {
        return av::CaptureDevice::default_with_media(av::MediaType::video())
            .context("no default AVFoundation video device found");
    };

    let devices = discovered_devices();

    if let Some(device) = devices
        .iter()
        .find(|d| d.localized_name().to_string().contains(name))
    {
        return Ok(device.retained());
    }

    let available: Vec<String> = devices
        .iter()
        .map(|d| d.localized_name().to_string())
        .collect();
    anyhow::bail!(
        "AVFoundation camera matching '{name}' not found. Available cameras: {available:?}"
    )
}
