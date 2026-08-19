//! Hardware HEVC encoding via VideoToolbox (VTCompressionSession).
//!
//! Produces HEVC Annex B with 4-byte start codes and in-band VPS/SPS/PPS on
//! every keyframe, the framing NVIDIA's decoders expect. The encoder runs
//! synchronously: each frame is submitted with one frame in flight and drained
//! via `complete_all()`, so the C output callback fires before
//! [`HevcEncoder::encode`] returns.

use crate::{hevc, videotoolbox::err};
use anyhow::{Result, anyhow, bail};
use cidre::{
    arc, cf, cm, cv, os,
    vt::{
        self,
        compression_properties::{frame_keys, keys, profile_level},
    },
};
use std::{ffi::c_void, sync::Mutex};

/// How to create a [`HevcEncoder`] session.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// The frame rate the session's rate control should expect.
    pub frame_rate: f64,
    /// Target average bitrate, in bits per second.
    pub average_bitrate_bps: u32,
    /// Force a keyframe every N encoded frames (encoder-side counter), in
    /// addition to the session's MaxKeyFrameInterval property.
    pub keyframe_interval: usize,
}

/// One encoded access unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedFrame {
    /// HEVC Annex B bytes; keyframes carry VPS/SPS/PPS in band.
    pub annex_b: Vec<u8>,
    /// Whether this access unit is a keyframe a decoder can start from.
    pub is_keyframe: bool,
}

/// Shared with the C callback via the session ref-con. Encoding is synchronous
/// (one frame in flight) so the `Mutex` never contends; it just hands the result
/// back safely.
struct EncoderShared {
    output: Mutex<Option<Result<EncodedFrame>>>,
}

/// The VideoToolbox compression output callback. `ctx` is the `Box::into_raw`'d
/// [`EncoderShared`] passed as the session ref-con.
///
/// # Safety
/// Registered as the session's `OutputCallback<EncoderShared>`; VideoToolbox
/// invokes it with the exact ref-con pointer we supplied. The pointer outlives
/// the session (reclaimed only after the session is invalidated in teardown),
/// and encoding is synchronous with a single frame in flight, so there is no
/// concurrent or dangling access.
extern "C" fn output_callback(
    ctx: *mut EncoderShared,
    _src_frame_ref_con: *mut c_void,
    status: os::Status,
    _flags: vt::EncodeInfoFlags,
    sample_buf: Option<&cm::SampleBuf>,
) {
    // SAFETY: see the function's safety comment; `ctx` is a live `EncoderShared`.
    let shared = unsafe { &*ctx };
    let result = build_encoded_frame(status, sample_buf);
    *shared.output.lock().expect("encoder output mutex poisoned") = Some(result);
}

/// Repackage the callback's sample buffer into an Annex B access unit,
/// prepending VPS/SPS/PPS on keyframes.
fn build_encoded_frame(
    status: os::Status,
    sample_buf: Option<&cm::SampleBuf>,
) -> Result<EncodedFrame> {
    status
        .result()
        .map_err(|e| anyhow!("VideoToolbox encode failed: {e:?}"))?;
    let sample = sample_buf.ok_or_else(|| anyhow!("encode callback produced no sample buffer"))?;

    let is_keyframe = sample.is_key_frame();

    let block = sample
        .data_buf()
        .ok_or_else(|| anyhow!("encoded sample has no data buffer"))?;
    // The block buffer holds 4-byte length-prefixed NALs; make it contiguous
    // (cheaply, if it already is) before converting to Annex B.
    let contiguous = match block.try_contiguous_buf() {
        Some(contiguous) => contiguous,
        None => block
            .make_contiguous()
            .map_err(err("failed to make block buffer contiguous"))?,
    };
    let length_prefixed: &[u8] = contiguous.as_ref();

    // The slice data dominates the output; +128 covers any keyframe parameter sets.
    let mut annex_b = Vec::with_capacity(length_prefixed.len() + 128);
    if is_keyframe {
        let desc = sample
            .format_desc()
            .ok_or_else(|| anyhow!("keyframe sample has no format description"))?;
        let (count, _header_len) = desc
            .hevc_params_count_and_header_len()
            .map_err(|e| anyhow!("failed to read HEVC parameter set count: {e:?}"))?;
        for index in 0..count {
            let param_set = desc
                .hevc_param_set_at(index)
                .map_err(|e| anyhow!("failed to read HEVC parameter set {index}: {e:?}"))?;
            annex_b.extend_from_slice(&[0, 0, 0, 1]);
            annex_b.extend_from_slice(param_set);
        }
    }

    hevc::length_prefixed_to_annex_b(length_prefixed, &mut annex_b)?;

    Ok(EncodedFrame {
        annex_b,
        is_keyframe,
    })
}

/// A synchronous hardware HEVC encoder built on VideoToolbox.
pub struct HevcEncoder {
    session: arc::R<vt::CompressionSession>,
    /// `Box::into_raw`'d ref-con handed to the output callback. Owned by us;
    /// reclaimed on teardown (Drop / session recreation).
    shared: *mut EncoderShared,
    /// Frames seen since the last keyframe.
    seen_frames: usize,
    keyframe_interval: usize,
    frame_rate: f64,
    average_bitrate_bps: u32,
    /// Monotonic presentation timestamp counter (frame index).
    pts: i64,
    /// Resolution the current session was created at; a change recreates it.
    width: u32,
    height: u32,
}

impl HevcEncoder {
    /// Create a hardware HEVC encoder session.
    pub fn new(config: &EncoderConfig) -> Result<Self> {
        let (session, shared) = Self::create_session(
            config.width,
            config.height,
            config.frame_rate,
            config.average_bitrate_bps,
            config.keyframe_interval,
        )?;

        Ok(Self {
            session,
            shared,
            seen_frames: 0,
            keyframe_interval: config.keyframe_interval,
            frame_rate: config.frame_rate,
            average_bitrate_bps: config.average_bitrate_bps,
            pts: 0,
            width: config.width,
            height: config.height,
        })
    }

    /// Create and configure a hardware HEVC `VTCompressionSession` at the given
    /// resolution together with the `EncoderShared` ref-con wired to its output
    /// callback.
    fn create_session(
        width: u32,
        height: u32,
        frame_rate: f64,
        bitrate_bps: u32,
        keyframe_interval: usize,
    ) -> Result<(arc::R<vt::CompressionSession>, *mut EncoderShared)> {
        let shared = Box::into_raw(Box::new(EncoderShared {
            output: Mutex::new(None),
        }));

        let mut session = match vt::CompressionSession::new(
            width,
            height,
            cm::VideoCodec::HEVC,
            None, // encoder_spec
            None, // src_image_buf_attrs
            None, // compressed_data_allocator
            Some(output_callback),
            shared,
        ) {
            Ok(session) => session,
            Err(e) => {
                // SAFETY: `shared` was just created via `Box::into_raw` and the
                // session was never created, so nothing else references it.
                unsafe { drop(Box::from_raw(shared)) };
                return Err(anyhow!("failed to create VTCompressionSession: {e:?}"));
            }
        };

        let mut configure = || -> Result<()> {
            let fps = frame_rate_to_timescale(frame_rate);

            let bitrate = cf::Number::from_i32(bitrate_bps as i32);
            let expected_frame_rate = cf::Number::from_i32(fps);
            let max_key_frame_interval = cf::Number::from_i32(keyframe_interval as i32);

            let mut props = cf::DictionaryMut::with_capacity(6);
            // Real-time, no B-frames (no frame reordering), HEVC Main auto level.
            props.insert(keys::real_time(), cf::Boolean::value_true());
            props.insert(keys::allow_frame_reordering(), cf::Boolean::value_false());
            props.insert(keys::avarage_bit_rate(), &bitrate);
            props.insert(keys::expected_frame_rate(), &expected_frame_rate);
            props.insert(keys::max_key_frame_interval(), &max_key_frame_interval);
            props.insert(keys::profile_lvl(), profile_level::hevc::main_auto_lvl());

            session
                .set_props(&props)
                .map_err(err("failed to set compression properties"))?;
            session
                .prepare()
                .map_err(err("failed to prepare compression session"))?;
            Ok(())
        };

        if let Err(e) = configure() {
            session.invalidate();
            // SAFETY: the session was just invalidated, so its callback can no
            // longer fire; `shared` (from `Box::into_raw`) is otherwise unused.
            unsafe { drop(Box::from_raw(shared)) };
            return Err(e);
        }

        Ok((session, shared))
    }

    /// Encode one frame synchronously.
    ///
    /// `bgra` is packed 32BGRA, `width * height * 4` bytes. Passing a new
    /// resolution transparently recreates the session; the frame that does so is
    /// a keyframe. The output is HEVC Annex B, with in-band VPS/SPS/PPS on
    /// keyframes.
    pub fn encode(
        &mut self,
        bgra: &[u8],
        width: u32,
        height: u32,
        force_keyframe: bool,
    ) -> Result<EncodedFrame> {
        // A resolution change requires a fresh session (VTCompressionSession
        // dimensions are fixed at creation); the next frame is then a keyframe.
        let mut force_keyframe = force_keyframe;
        if (width, height) != (self.width, self.height) {
            self.recreate_session(width, height)?;
            force_keyframe = true;
        }

        // 1. Keyframe cadence.
        let force_keyframe = force_keyframe || self.seen_frames >= self.keyframe_interval;
        if force_keyframe {
            self.seen_frames = 0;
        }
        self.seen_frames += 1;

        let expected_len = width as usize * height as usize * 4;
        if bgra.len() < expected_len {
            bail!(
                "BGRA input too small: {} bytes, need {expected_len}",
                bgra.len()
            );
        }

        let pixel_buf = self.make_pixel_buffer(bgra, width, height)?;

        // 2. Encode synchronously: clear the output slot, submit one frame, then
        //    flush so the output callback fills the slot before we read it.
        // SAFETY: `self.shared` is a live `EncoderShared` (from `Box::into_raw`,
        // reclaimed only in teardown); the raw deref does not alias `self`.
        let shared = unsafe { &*self.shared };
        *shared.output.lock().expect("encoder output mutex poisoned") = None;

        let fps = frame_rate_to_timescale(self.frame_rate);
        let frame_props = force_keyframe.then(|| {
            cf::DictionaryOf::<cf::String, cf::Type>::with_keys_values(
                &[frame_keys::force_key_frame()],
                &[cf::Boolean::value_true().as_type_ref()],
            )
        });

        self.session
            .encode_frame(
                &pixel_buf,
                cm::Time::new(self.pts, fps),
                cm::Time::new(1, fps),
                frame_props.as_deref(),
                std::ptr::null_mut(),
                &mut None,
            )
            .map_err(|e| anyhow!("VTCompressionSessionEncodeFrame failed: {e:?}"))?;
        self.pts += 1;

        self.session
            .complete_all()
            .map_err(err("VTCompressionSessionCompleteFrames failed"))?;

        // 3. Take the callback's output (Annex B with in-band parameter sets on
        //    keyframes).
        shared
            .output
            .lock()
            .expect("encoder output mutex poisoned")
            .take()
            .ok_or_else(|| anyhow!("encoder produced no output for the submitted frame"))?
    }

    /// Copy tightly-packed BGRA into a new 32BGRA `cv::PixelBuf`, honoring the
    /// buffer's own `bytes_per_row` stride.
    fn make_pixel_buffer(
        &self,
        bgra: &[u8],
        width: u32,
        height: u32,
    ) -> Result<arc::R<cv::PixelBuf>> {
        let width = width as usize;
        let height = height as usize;
        let src_stride = width * 4;

        let mut pixel_buf = cv::PixelBuf::new(width, height, cv::PixelFormat::_32_BGRA, None)
            .map_err(|e| anyhow!("failed to create pixel buffer: {e:?}"))?;

        // SAFETY: we lock the base address for the duration of the copy and
        // unlock before returning; the copied rows stay within the locked
        // buffer (`height` rows of `src_stride` bytes at `bytes_per_row` stride).
        unsafe {
            pixel_buf
                .lock_base_addr(cv::pixel_buffer::LockFlags::DEFAULT)
                .result()
                .map_err(|e| anyhow!("failed to lock pixel buffer: {e:?}"))?;

            let bytes_per_row = pixel_buf.bytes_per_row();
            let dst = pixel_buf.base_address_mut() as *mut u8;
            for row in 0..height {
                let src = bgra[row * src_stride..].as_ptr();
                std::ptr::copy_nonoverlapping(src, dst.add(row * bytes_per_row), src_stride);
            }

            pixel_buf.unlock_lock_base_addr(cv::pixel_buffer::LockFlags::DEFAULT);
        }

        Ok(pixel_buf)
    }

    /// Tear down the current session and build a fresh one at the given
    /// resolution, resetting timing and forcing the next frame to be a keyframe.
    fn recreate_session(&mut self, width: u32, height: u32) -> Result<()> {
        // Build the new session first so a failure leaves the old one intact.
        let (session, shared) = Self::create_session(
            width,
            height,
            self.frame_rate,
            self.average_bitrate_bps,
            self.keyframe_interval,
        )?;

        self.session.invalidate();
        // SAFETY: the old session was just invalidated (its callback can no
        // longer fire), so its ref-con is safe to reclaim.
        unsafe { drop(Box::from_raw(self.shared)) };

        self.session = session;
        self.shared = shared;
        self.width = width;
        self.height = height;
        self.pts = 0;
        self.seen_frames = 0;
        Ok(())
    }

    /// Force a keyframe every `every` encoded frames from now on.
    pub fn set_keyframe_interval(&mut self, every: usize) {
        self.keyframe_interval = every;
        let interval = cf::Number::from_i32(every as i32);
        // Best-effort live update; the encoder-side counter enforces the
        // interval either way.
        if let Err(e) = self
            .session
            .set_prop(keys::max_key_frame_interval(), Some(interval.as_type_ref()))
        {
            log::warn!("failed to update MaxKeyFrameInterval: {e:?}");
        }
    }

    /// Retarget the running session's rate control.
    pub fn set_bitrate_and_frame_rate(&mut self, bitrate_bps: u32, frame_rate: f64) -> Result<()> {
        self.average_bitrate_bps = bitrate_bps;
        self.frame_rate = frame_rate;

        let bitrate = cf::Number::from_i32(bitrate_bps as i32);
        self.session
            .set_prop(keys::avarage_bit_rate(), Some(bitrate.as_type_ref()))
            .map_err(|e| anyhow!("failed to update AverageBitRate: {e:?}"))?;

        let expected_frame_rate = cf::Number::from_i32(frame_rate_to_timescale(frame_rate));
        self.session
            .set_prop(
                keys::expected_frame_rate(),
                Some(expected_frame_rate.as_type_ref()),
            )
            .map_err(|e| anyhow!("failed to update ExpectedFrameRate: {e:?}"))?;

        Ok(())
    }
}

impl Drop for HevcEncoder {
    fn drop(&mut self) {
        self.session.invalidate();
        // SAFETY: the session was just invalidated so its output callback can no
        // longer run; `self.shared` (from `Box::into_raw`) is reclaimed exactly
        // once here.
        unsafe { drop(Box::from_raw(self.shared)) };
    }
}

/// Timescale (and expected frame rate) for a floating-point frame rate, clamped
/// to a valid positive integer.
fn frame_rate_to_timescale(frame_rate: f64) -> i32 {
    frame_rate.round().max(1.0) as i32
}
