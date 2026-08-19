//! Hardware HEVC decoding via VideoToolbox (VTDecompressionSession).
//!
//! The session is built lazily from the first keyframe's in-band VPS/SPS/PPS,
//! which authoritatively describe the stream (resolution, profile, etc.). Each
//! access unit is repackaged from Annex B into the 4-byte length-prefixed
//! framing VideoToolbox expects, wrapped in a `cm::SampleBuf`, and decoded
//! synchronously: the C output callback delivers the decoded `cv::PixelBuf`
//! before [`HevcDecoder::decode`] returns, so there is only ever one frame in
//! flight.

use crate::{
    BgraFrame,
    hevc::{self, ParameterSets, RpsEntry},
    videotoolbox::err,
};
use anyhow::{Result, anyhow};
use cidre::{arc, cf, cm, cv, os, vt};
use std::{
    ffi::c_void,
    sync::Mutex,
    time::{Duration, Instant},
};

/// How to create a [`HevcDecoder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderConfig {
    /// Emulate NVDEC-style concealment when a slice references pictures missing
    /// from the decoded picture buffer (see the [`hevc`](crate::hevc) module
    /// docs). Disabling it leaves slice headers untouched, so such frames fail
    /// to decode and the decoder waits for the next keyframe.
    pub conceal_missing_references: bool,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            conceal_missing_references: true,
        }
    }
}

/// Why an access unit could not be turned into a frame.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// No keyframe seen yet (or resyncing after a failure); feed a keyframe.
    #[error("waiting for a keyframe")]
    MissingKeyframe,
    /// This frame failed to decode; the decoder now waits for a keyframe.
    #[error("frame decode failed: {0}")]
    Frame(String),
    /// Unrecoverable (e.g. session creation failed).
    #[error("fatal decoder error: {0}")]
    Fatal(String),
}

/// One decoded picture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    /// The picture, as tightly packed 32BGRA.
    pub frame: BgraFrame,
    /// Whether the access unit it came from was a keyframe.
    pub is_keyframe: bool,
}

/// Shared with the C callback via the session ref-con. Decoding is synchronous
/// (one frame in flight) so the `Mutex` never contends; it just hands the result
/// back safely.
struct DecoderShared {
    output: Mutex<Option<Result<BgraFrame, String>>>,
}

/// The VideoToolbox decompression output callback. `ctx` is the
/// `Box::into_raw`'d [`DecoderShared`] passed as the session ref-con.
///
/// # Safety
/// Registered as the session's `OutputCb<DecoderShared>`; VideoToolbox invokes
/// it with the exact ref-con pointer we supplied. The pointer outlives the
/// session (reclaimed only after the session is invalidated in teardown), and
/// decoding is synchronous with a single frame in flight, so there is no
/// concurrent or dangling access.
extern "C" fn output_callback(
    ctx: *mut DecoderShared,
    _src_frame_ref_con: *mut c_void,
    status: os::Status,
    _info_flags: vt::DecodeInfoFlags,
    image_buf: Option<&cv::ImageBuf>,
    _pts: cm::Time,
    _duration: cm::Time,
) {
    // SAFETY: see the function's safety comment; `ctx` is a live `DecoderShared`.
    let shared = unsafe { &*ctx };
    let result = copy_decoded_frame(status, image_buf);
    *shared.output.lock().expect("decoder output mutex poisoned") = Some(result);
}

/// Copy the decoded pixel buffer's BGRA rows into a tightly packed frame,
/// honoring the buffer's `bytes_per_row` stride.
fn copy_decoded_frame(
    status: os::Status,
    image_buf: Option<&cv::ImageBuf>,
) -> Result<BgraFrame, String> {
    status
        .result()
        .map_err(|e| format!("VideoToolbox decode failed: {e:?}"))?;
    let image_buf =
        image_buf.ok_or_else(|| "decode callback produced no image buffer".to_string())?;

    // Retain to obtain an owned, mutable handle (`lock_base_addr` needs `&mut`).
    let mut pixel_buf: arc::R<cv::PixelBuf> = image_buf.retained();

    crate::slot::copy_pixel_buf_bgra(&mut pixel_buf)
        .ok_or_else(|| "failed to lock decoded pixel buffer".to_string())
}

/// A live `VTDecompressionSession` together with the format description it was
/// built from and the `Box::into_raw`'d ref-con wired to its output callback.
struct DecoderSession {
    session: arc::R<vt::DecompressionSession>,
    /// The format description the session was built from; also referenced by
    /// every `cm::SampleBuf` we submit for decode.
    format_desc: arc::R<cm::VideoFormatDesc>,
    /// `Box::into_raw`'d ref-con handed to the output callback. Owned by us;
    /// reclaimed on teardown (Drop).
    shared: *mut DecoderShared,
}

impl DecoderSession {
    /// Build a decompression session from HEVC parameter sets, producing 32BGRA
    /// output.
    fn new(parameter_sets: &ParameterSets) -> Result<Self> {
        let pointers = [
            parameter_sets.vps.as_ptr(),
            parameter_sets.sps.as_ptr(),
            parameter_sets.pps.as_ptr(),
        ];
        let sizes = [
            parameter_sets.vps.len(),
            parameter_sets.sps.len(),
            parameter_sets.pps.len(),
        ];
        let format_desc = cm::VideoFormatDesc::with_hevc_param_sets(
            3, &pointers, &sizes, // Our repackaged NALs use 4-byte length prefixes.
            4, None,
        )
        .map_err(|e| anyhow!("failed to create HEVC format description: {e:?}"))?;

        let shared = Box::into_raw(Box::new(DecoderShared {
            output: Mutex::new(None),
        }));

        // Request 32BGRA output so decoded frames land in the packed byte order
        // callers get, without a colorspace conversion pass.
        let dst_attrs = cf::DictionaryOf::<cf::String, cf::Type>::with_keys_values(
            &[cv::pixel_buffer_keys::pixel_format()],
            &[cv::PixelFormat::_32_BGRA.to_cf_number().as_type_ref()],
        );

        let record = vt::DecompressionOutputCbRecord::new(shared, Some(output_callback));

        let session = match vt::DecompressionSession::new(
            &format_desc,
            None,
            Some(&dst_attrs),
            Some(&record),
        ) {
            Ok(session) => session,
            Err(e) => {
                // SAFETY: `shared` was just created via `Box::into_raw` and
                // the session was never created, so nothing else references it.
                unsafe { drop(Box::from_raw(shared)) };
                return Err(anyhow!("failed to create VTDecompressionSession: {e:?}"));
            }
        };

        Ok(Self {
            session,
            format_desc,
            shared,
        })
    }
}

impl Drop for DecoderSession {
    fn drop(&mut self) {
        self.session.invalidate();
        // SAFETY: the session was just invalidated so its output callback can no
        // longer run; `self.shared` (from `Box::into_raw`) is reclaimed exactly
        // once here.
        unsafe { drop(Box::from_raw(self.shared)) };
    }
}

/// A synchronous hardware HEVC decoder built on VideoToolbox.
pub struct HevcDecoder {
    /// Session + format description; `None` until the first parameter sets
    /// arrive (and after `reset()`).
    session: Option<DecoderSession>,
    /// Parameter sets the current session was created from; a change (e.g. the
    /// encoder restarted at a new resolution) forces session recreation.
    parameter_sets: Option<ParameterSets>,
    /// Scratch buffer for Annex B → length-prefixed repackaging.
    length_prefixed: Vec<u8>,
    /// Whether we still need a keyframe to (re)synchronize before inter-coded
    /// frames can be decoded. `true` initially, after `reset()`, and after any
    /// per-frame decode failure; cleared once a keyframe decodes successfully.
    /// See `decode` for why VideoToolbox needs this where NVDEC doesn't.
    waiting_for_keyframe: bool,
    /// Whether to repair slices referencing pictures missing from the DPB.
    conceal_missing_references: bool,
    /// Parsed SPS/PPS of the current session, used to parse slice headers for
    /// the RPS repair below. `None` (parse unsupported, or concealment off)
    /// disables the repair.
    header_info: Option<(hevc::SpsInfo, hevc::PpsInfo)>,
    /// POC lsbs assumed to be in VideoToolbox's DPB; used by the RPS repair (see `hevc`).
    decoded_poc_lsbs: Vec<u32>,
    /// Rate limiting for the RPS-repair warning: the first in a quiet period logs
    /// immediately, the rest fold into "and N more".
    repair_log_last: Option<Instant>,
    /// Repairs suppressed since the last emitted warning.
    repairs_suppressed: u32,
}

impl HevcDecoder {
    /// Create a decoder. The hardware session is created lazily, from the
    /// parameter sets of the first keyframe fed to [`HevcDecoder::decode`].
    pub fn new(config: &DecoderConfig) -> HevcDecoder {
        Self {
            session: None,
            parameter_sets: None,
            length_prefixed: Vec::new(),
            waiting_for_keyframe: true,
            conceal_missing_references: config.conceal_missing_references,
            header_info: None,
            decoded_poc_lsbs: Vec::new(),
            repair_log_last: None,
            repairs_suppressed: 0,
        }
    }

    /// Decode one HEVC Annex B access unit into a packed BGRA frame.
    pub fn decode(&mut self, annex_b: &[u8]) -> Result<DecodedFrame, DecodeError> {
        let nals = hevc::annex_b_nal_units(annex_b);

        // 1. If the access unit carries parameter sets and they differ from the
        //    current ones, (re)create the session. A keyframe always carries all
        //    three; a parameter-set change means the encoder restarted (e.g. at a
        //    new resolution), so the old session can no longer decode.
        if let Some(parameter_sets) = hevc::parameter_sets_from_nals(&nals)
            && self.parameter_sets.as_ref() != Some(&parameter_sets)
        {
            let session = DecoderSession::new(&parameter_sets)
                .map_err(|e| DecodeError::Fatal(format!("{e:#}")))?;
            // Parse the SPS/PPS for slice-header parsing (RPS repair). An
            // unsupported bitstream feature disables the repair but not
            // decoding itself.
            self.header_info = self
                .conceal_missing_references
                .then(|| {
                    hevc::parse_sps(&parameter_sets.sps)
                        .and_then(|sps| Ok((sps, hevc::parse_pps(&parameter_sets.pps)?)))
                        .inspect_err(|e| {
                            log::warn!("could not parse parameter sets, RPS repair disabled: {e:#}")
                        })
                        .ok()
                })
                .flatten();
            // Replace after successful creation so a failure leaves the old
            // session intact; dropping the old one invalidates it.
            self.session = Some(session);
            self.parameter_sets = Some(parameter_sets);
            self.decoded_poc_lsbs.clear();
        }

        // 2. Still no session: the sender emitted inter frames before any keyframe
        //    reached us. The caller must request a keyframe.
        let session = self.session.as_ref().ok_or(DecodeError::MissingKeyframe)?;

        // 2b. Out of sync and not a keyframe: don't submit — an inter frame can only
        //     reference pictures we don't have. Ask for a keyframe instead.
        let contains_keyframe = hevc::nals_contain_keyframe(&nals);
        if self.waiting_for_keyframe && !contains_keyframe {
            return Err(DecodeError::MissingKeyframe);
        }

        // 3. Repackage Annex B → 4-byte length prefixes, skipping VPS/SPS/PPS
        //    (they live in the format description). While repackaging, repair
        //    slices whose RPS references pictures we never decoded —
        //    VideoToolbox rejects these where NVDEC conceals; see the `hevc`
        //    module docs.
        self.length_prefixed.clear();
        // POC lsbs (current picture + retained references) to install as the
        // assumed DPB contents once this access unit decodes successfully.
        let mut dpb_after_decode: Option<Vec<u32>> = None;
        // Only first-slice segments are rewritten, so a multi-slice picture would end
        // up with disagreeing slices; skip the repair then and fail open (the
        // supported encoders emit single-slice pictures).
        let repair_enabled = nals
            .iter()
            .filter(|n| hevc::nal_unit_type(n).is_some_and(|t| t <= 21))
            .count()
            <= 1;
        for &nal in &nals {
            if hevc::is_parameter_set(nal) {
                continue;
            }
            let mut repaired: Option<Vec<u8>> = None;
            // Non-slice NALs (SEI, AUD, ...) and unsupported slice layouts
            // fail to parse and are passed through unmodified.
            if let Some((sps, pps)) = &self.header_info
                && let Ok(info) = hevc::parse_slice(nal, sps, pps)
            {
                let max = 1i64 << sps.poc_lsb_bits;
                let poc = i64::from(info.poc_lsb.unwrap_or(0));
                let ref_lsb = |delta: i32| ((poc + i64::from(delta)).rem_euclid(max)) as u32;
                let decoded = &self.decoded_poc_lsbs;

                // Partition the RPS: present entries stay; absent unused
                // entries are dropped; absent *used* entries need a
                // concealment target.
                let mut entries = Vec::with_capacity(info.rps.len());
                let mut missing_used = false;
                for e in &info.rps {
                    if decoded.contains(&ref_lsb(e.delta_poc)) {
                        entries.push(*e);
                    } else if e.used {
                        missing_used = true;
                    }
                }
                // Newest decoded picture (smallest POC distance behind the
                // current one) as the concealment reference.
                let conceal_delta = decoded
                    .iter()
                    .map(|&lsb| (poc - i64::from(lsb)).rem_euclid(max))
                    .filter(|&d| d > 0)
                    .min()
                    .map(|d| -d as i32);
                if missing_used && let Some(delta) = conceal_delta {
                    match entries.iter_mut().find(|e| e.delta_poc == delta) {
                        Some(e) => e.used = true,
                        None => entries.push(RpsEntry {
                            delta_poc: delta,
                            used: true,
                        }),
                    }
                }
                // Repair only when it can produce a decodable P-slice; an
                // unrepairable one is submitted as-is so the normal
                // failure path (keyframe request) takes over.
                let repairable = !missing_used || entries.iter().any(|e| e.used);
                if repair_enabled && entries != info.rps && repairable {
                    // Fail open: an un-rewritable slice passes through unmodified.
                    if let Ok(rewritten) = hevc::rewrite_rps(nal, sps, pps, &entries) {
                        let now = Instant::now();
                        let within_window = self
                            .repair_log_last
                            .is_some_and(|last| now.duration_since(last) < Duration::from_secs(1));
                        if within_window {
                            self.repairs_suppressed += 1;
                        } else {
                            let suppressed = self.repairs_suppressed;
                            log::warn!(
                                "repaired slice RPS (poc lsb {poc}): kept {} of {} references{}{}",
                                entries.len(),
                                info.rps.len(),
                                if missing_used {
                                    ", concealed a missing used reference"
                                } else {
                                    ""
                                },
                                if suppressed > 0 {
                                    format!(" (and {suppressed} more in the last second)")
                                } else {
                                    String::new()
                                },
                            );
                            self.repair_log_last = Some(now);
                            self.repairs_suppressed = 0;
                        }
                        repaired = Some(rewritten);
                    }
                }
                if dpb_after_decode.is_none() {
                    let rps = if repaired.is_some() {
                        &entries
                    } else {
                        &info.rps
                    };
                    let mut dpb: Vec<u32> = rps.iter().map(|e| ref_lsb(e.delta_poc)).collect();
                    dpb.push(poc as u32);
                    dpb_after_decode = Some(dpb);
                }
            }
            let bytes = repaired.as_deref().unwrap_or(nal);
            self.length_prefixed
                .extend_from_slice(&(bytes.len() as u32).to_be_bytes());
            self.length_prefixed.extend_from_slice(bytes);
        }

        // 4. Wrap the length-prefixed NALs in a SampleBuf and decode
        //    synchronously; the output callback fills the shared slot before
        //    `decode` returns (no ENABLE_ASYNCHRONOUS_DECOMPRESSION flag).
        // SAFETY: `session.shared` is a live `DecoderShared` (from `Box::into_raw`,
        // reclaimed only when the `DecoderSession` is dropped); the raw deref does
        // not alias `self`.
        let shared = unsafe { &*session.shared };
        *shared.output.lock().expect("decoder output mutex poisoned") = None;

        let sample_buf = build_sample_buf(&self.length_prefixed, &session.format_desc)
            .map_err(|e| DecodeError::Fatal(format!("{e:#}")))?;

        let submit_result = session
            .session
            .decode(&sample_buf, vt::DecodeFrameFlags::default());

        // 5. Take the callback's output. A missing slot or an error status after
        //    the session exists is a per-frame decode failure.
        let output = shared
            .output
            .lock()
            .expect("decoder output mutex poisoned")
            .take();

        let frame = match submit_result
            .map_err(|e| format!("submit failed: {e:?}"))
            .and_then(|()| {
                output.unwrap_or_else(|| Err("decode callback produced no result".to_string()))
            }) {
            Ok(frame) => frame,
            Err(msg) => {
                // Per-frame failures (e.g. kVTVideoDecoderBadDataErr on reference loss) are
                // recoverable: wait for a keyframe instead of tearing the decoder down.
                log::warn!("VideoToolbox per-frame decode failed, requesting a keyframe: {msg}");
                self.waiting_for_keyframe = true;
                return Err(DecodeError::Frame(msg));
            }
        };

        if contains_keyframe {
            self.waiting_for_keyframe = false;
        }

        // The decoder's DPB now holds this picture plus the references its
        // slice header retained; anything else was released per RPS semantics.
        if self.header_info.is_some() {
            self.decoded_poc_lsbs = dpb_after_decode.unwrap_or_default();
        }

        Ok(DecodedFrame {
            frame,
            is_keyframe: contains_keyframe,
        })
    }

    /// Drop the session and wait for a keyframe again.
    pub fn reset(&mut self) {
        self.session = None;
        self.parameter_sets = None;
        self.waiting_for_keyframe = true;
        self.header_info = None;
        self.decoded_poc_lsbs.clear();
    }
}

/// Wrap length-prefixed NAL data in a `cm::SampleBuf` referencing the given
/// format description.
fn build_sample_buf(
    data: &[u8],
    format_desc: &cm::VideoFormatDesc,
) -> Result<arc::R<cm::SampleBuf>> {
    let mut block = cm::BlockBuf::with_mem_block(data.len())
        .map_err(|e| anyhow!("failed to allocate block buffer: {e:?}"))?;
    block
        .as_mut_slice()
        .map_err(err("failed to access block buffer"))?
        .copy_from_slice(data);

    let timing = cm::SampleTimingInfo {
        duration: cm::Time::new(1, 30),
        pts: cm::Time::new(0, 30),
        dts: cm::Time::invalid(),
    };
    let sample_sizes = [data.len()];

    let mut sample_buf = None;
    // SAFETY: all pointers/references passed to `create_in` outlive the call:
    // `block` and `format_desc` are borrowed for the duration, `timing` and
    // `sample_sizes` are stack locals passed as pointers to one-element arrays
    // matching the `num_*_entries` counts of 1, and `sample_buf` receives the
    // created buffer.
    unsafe {
        cm::SampleBuf::create_in(
            None,
            Some(&block),
            true,
            None,
            std::ptr::null(),
            Some(format_desc),
            1,
            1,
            &timing,
            1,
            sample_sizes.as_ptr(),
            &mut sample_buf,
        )
        .map_err(|e| anyhow!("failed to create sample buffer: {e:?}"))?;
    }

    sample_buf.ok_or_else(|| anyhow!("sample buffer creation returned no buffer"))
}
