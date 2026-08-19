//! HEVC bitstream repackaging between Annex B and length-prefixed framing.
//!
//! See the [module docs](super) for why both framings show up.

use anyhow::{Result, bail};

/// HEVC VPS (video parameter set) NAL unit type (ITU-T H.265 Table 7-1).
pub const NAL_TYPE_VPS: u8 = 32;
/// HEVC SPS (sequence parameter set) NAL unit type (ITU-T H.265 Table 7-1).
pub const NAL_TYPE_SPS: u8 = 33;
/// HEVC PPS (picture parameter set) NAL unit type (ITU-T H.265 Table 7-1).
pub const NAL_TYPE_PPS: u8 = 34;

/// The NAL unit type from the first byte of an HEVC NAL unit header.
pub fn nal_unit_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|byte| (byte >> 1) & 0x3F)
}

/// Whether `nal_type` is an IRAP picture (BLA/IDR/CRA, types 16..=23) — a
/// keyframe the decoder can start from.
pub fn is_irap(nal_type: u8) -> bool {
    (16..=23).contains(&nal_type)
}

/// Whether `nal` is a VPS/SPS/PPS parameter set NAL unit.
pub fn is_parameter_set(nal: &[u8]) -> bool {
    matches!(
        nal_unit_type(nal),
        Some(NAL_TYPE_VPS | NAL_TYPE_SPS | NAL_TYPE_PPS)
    )
}

/// Whether a NAL list contains a keyframe (IRAP) slice.
pub fn nals_contain_keyframe(nals: &[&[u8]]) -> bool {
    nals.iter()
        .any(|&nal| nal_unit_type(nal).is_some_and(is_irap))
}

/// Split an Annex B stream into NAL unit payloads (start codes stripped).
/// Handles both 3-byte (`00 00 01`) and 4-byte (`00 00 00 01`) start codes.
pub fn annex_b_nal_units(data: &[u8]) -> Vec<&[u8]> {
    // Indices right after each start code.
    let mut payload_starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            payload_starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }

    let mut units = Vec::with_capacity(payload_starts.len());
    for (index, &start) in payload_starts.iter().enumerate() {
        let mut end = payload_starts
            .get(index + 1)
            .map_or(data.len(), |&next| next - 3);
        // A 4-byte start code leaves one extra zero before the 3-byte match.
        // The byte right before a `00 00 01` is ambiguous: it could be the lead
        // zero of a 4-byte start code, or the final `0x00` of the preceding NAL.
        // Attributing it to the start code matches the universal splitter
        // convention (ffmpeg et al.), and emulation prevention guarantees an
        // escaped payload never ends in `00 00`, so at most one byte is ever at
        // stake, and only against encoders that emit 3-byte start codes.
        if end > start && data[end - 1] == 0 {
            end -= 1;
        }
        if end > start {
            units.push(&data[start..end]);
        }
    }
    units
}

/// Convert 4-byte big-endian length-prefixed NAL units (as produced by
/// VideoToolbox) into Annex B with 4-byte start codes.
pub fn length_prefixed_to_annex_b(data: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let mut rest = data;
    while !rest.is_empty() {
        if rest.len() < 4 {
            bail!(
                "truncated NAL unit length prefix ({} trailing bytes)",
                rest.len()
            );
        }
        let length = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
        rest = &rest[4..];
        if rest.len() < length {
            bail!(
                "NAL unit shorter ({}) than its length prefix ({length})",
                rest.len()
            );
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&rest[..length]);
        rest = &rest[length..];
    }
    Ok(())
}

/// The three HEVC parameter set NAL units (without start codes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterSets {
    /// The video parameter set (VPS) NAL unit.
    pub vps: Vec<u8>,
    /// The sequence parameter set (SPS) NAL unit.
    pub sps: Vec<u8>,
    /// The picture parameter set (PPS) NAL unit.
    pub pps: Vec<u8>,
}

/// Extract VPS/SPS/PPS from a NAL list. Returns `None` unless all three are
/// present (keyframes carry all three in-band; other frames none).
pub fn parameter_sets_from_nals(nals: &[&[u8]]) -> Option<ParameterSets> {
    let find = |ty| {
        nals.iter()
            .rev()
            .find(|&&n| nal_unit_type(n) == Some(ty))
            .map(|n| n.to_vec())
    };
    Some(ParameterSets {
        vps: find(NAL_TYPE_VPS)?,
        sps: find(NAL_TYPE_SPS)?,
        pps: find(NAL_TYPE_PPS)?,
    })
}
