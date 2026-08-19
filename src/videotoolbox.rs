//! Hardware HEVC encode and decode backed by Apple VideoToolbox.
//!
//! The wire format is HEVC Annex B with in-band VPS/SPS/PPS on keyframes,
//! interoperable with NVIDIA's NVENC/NVDEC codecs. Both directions run
//! synchronously with a single frame in flight: [`HevcEncoder::encode`] and
//! [`HevcDecoder::decode`] return once VideoToolbox's C callback has delivered
//! the result, so neither needs a runtime or a background thread.
//!
//! The decoder additionally repairs slices whose reference picture set names
//! pictures it never decoded — see the [`hevc`](crate::hevc) module docs for
//! why that is needed to interoperate with NVDEC over a lossy link.

mod decoder;
mod encoder;

pub use decoder::{DecodeError, DecodedFrame, DecoderConfig, HevcDecoder};
pub use encoder::{EncodedFrame, EncoderConfig, HevcEncoder};

/// Build a `map_err` closure that wraps a debug-formatted error under `msg`.
pub(crate) fn err<E: std::fmt::Debug>(msg: &'static str) -> impl Fn(E) -> anyhow::Error {
    move |e| anyhow::anyhow!("{msg}: {e:?}")
}
