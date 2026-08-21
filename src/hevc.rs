//! Portable HEVC (H.265) bitstream tools: Annex B framing, parameter-set
//! extraction, and slice-header parsing and rewriting.
//!
//! Nothing here touches a macOS framework — the module is plain Rust and
//! compiles on every platform.
//!
//! Two jobs, both driven by decoder interop:
//!
//! # Framing
//!
//! VideoToolbox produces and consumes NAL units with 4-byte big-endian length
//! prefixes (hvcC-style, via `CMBlockBuffer`), while Annex B (`00 00 00 01`
//! start codes, in-band parameter sets) is what NVIDIA's codecs and most
//! low-latency wire formats use. [`annex_b_nal_units`],
//! [`length_prefixed_to_annex_b`] and [`parameter_sets_from_nals`] convert
//! between the two framings.
//!
//! # Reference concealment
//!
//! When a P-slice's short-term reference picture set (RPS) names a
//! `used_by_curr_pic` picture that is missing from the decoded picture buffer
//! — because the picture was lost in transport or was skipped while the
//! decoder waited for a recovery keyframe — NVDEC conceals the missing
//! reference and decodes a (possibly artifacted) frame, while VideoToolbox
//! rejects the slice outright with `kVTVideoDecoderBadDataErr` (-12909).
//!
//! Over a lossy link that rejection never resolves. The decoder asks for a
//! keyframe and decodes it, but the next P-slice still references a picture
//! from before that keyframe, which the decoder skipped while it was waiting.
//! That slice fails too and asks for another keyframe, and the cycle
//! repeats.
//!
//! [`rewrite_rps`] rewrites a slice header's RPS in place (slice payload
//! copied verbatim), which lets a decoder emulate NVDEC's concealment: remap a
//! missing *used* reference to the newest picture actually in the DPB, and
//! drop keep-alive entries (`used_by_curr_pic = 0`) for pictures that were
//! never decoded. VideoToolbox tolerates missing *unused* entries; dropping
//! them anyway keeps the RPS consistent with what the decoder holds.
//!
//! The parsers cover the profile of streams low-latency NVENC and VideoToolbox
//! HEVC encoders produce; anything outside that profile returns an error, in
//! which case callers must fall back to passing the slice through unmodified
//! (fail-open, surfacing the normal keyframe-recovery path).

mod bitstream;
mod headers;

pub use bitstream::*;
pub use headers::*;
