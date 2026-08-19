//! Minimal HEVC (H.265) header parsing and slice-header rewriting.
//!
//! See the [module docs](super) for the reference-concealment story these
//! parsers serve, and for the profile of streams they cover.

use anyhow::{Result, anyhow, ensure};

/// Reads bits from an RBSP (emulation-prevention bytes already removed).
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn u(&mut self, n: u32) -> Result<u32> {
        ensure!(n <= 32, "bit read too wide");
        let mut v = 0u32;
        for _ in 0..n {
            let byte = self
                .data
                .get(self.pos / 8)
                .ok_or_else(|| anyhow!("bitstream exhausted"))?;
            v = (v << 1) | u32::from((byte >> (7 - self.pos % 8)) & 1);
            self.pos += 1;
        }
        Ok(v)
    }

    fn flag(&mut self) -> Result<bool> {
        Ok(self.u(1)? == 1)
    }

    /// Exp-Golomb ue(v).
    fn ue(&mut self) -> Result<u32> {
        let mut zeros = 0u32;
        while self.u(1)? == 0 {
            zeros += 1;
            ensure!(zeros <= 31, "ue(v) prefix too long");
        }
        let suffix = self.u(zeros)?;
        Ok((1u32 << zeros) - 1 + suffix)
    }

    /// Exp-Golomb se(v).
    fn se(&mut self) -> Result<i32> {
        let k = self.ue()?;
        Ok(if k % 2 == 1 {
            (k as i32 + 1) / 2
        } else {
            -(k as i32 / 2)
        })
    }

    fn byte_aligned(&self) -> bool {
        self.pos.is_multiple_of(8)
    }
}

/// Writes bits, MSB-first, matching [`BitReader`]'s order.
#[derive(Default)]
struct BitWriter {
    data: Vec<u8>,
    /// Number of valid bits in the last byte of `data` (0 when byte-aligned).
    bit_len: usize,
}

impl BitWriter {
    fn put(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            let bit = (v >> i) & 1;
            if self.bit_len.is_multiple_of(8) {
                self.data.push(0);
            }
            let last = self.data.len() - 1;
            self.data[last] |= (bit as u8) << (7 - self.bit_len % 8);
            self.bit_len += 1;
        }
    }

    fn put_flag(&mut self, v: bool) {
        self.put(1, u32::from(v));
    }

    fn put_ue(&mut self, v: u32) {
        let v1 = (v as u64) + 1;
        let len = 64 - v1.leading_zeros();
        self.put(len - 1, 0);
        self.put(len, v1 as u32);
    }

    fn byte_aligned(&self) -> bool {
        self.bit_len.is_multiple_of(8)
    }
}

/// Remove HEVC emulation-prevention bytes (`00 00 03` -> `00 00`).
fn unescape_rbsp(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zeros = 0usize;
    for &b in data {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue; // Drop the emulation-prevention byte.
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        out.push(b);
    }
    out
}

/// Insert HEVC emulation-prevention bytes (`00 00 {00,01,02,03}` -> `00 00 03 ..`).
fn escape_rbsp(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 8);
    let mut zeros = 0usize;
    for &b in data {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        out.push(b);
    }
    out
}

/// One short-term RPS entry: POC delta relative to the current picture
/// (negative for "before"), and whether the current picture predicts from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RpsEntry {
    /// POC of the referenced picture minus the POC of the current picture.
    pub delta_poc: i32,
    /// `used_by_curr_pic_flag`: whether the current picture predicts from it.
    pub used: bool,
}

/// The fields of an SPS that slice-header parsing depends on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpsInfo {
    /// Number of bits in `slice_pic_order_cnt_lsb` (log2_max_pic_order_cnt_lsb).
    pub poc_lsb_bits: u32,
    /// `separate_colour_plane_flag` (4:4:4 streams coding planes separately).
    pub separate_colour_plane: bool,
    /// The SPS-level short-term RPS candidates a slice can select by index.
    pub short_term_rps: Vec<Vec<RpsEntry>>,
    /// `long_term_ref_pics_present_flag`.
    pub long_term_ref_pics_present: bool,
    /// `num_long_term_ref_pics_sps`.
    pub num_long_term_ref_pics_sps: u32,
    /// `sps_temporal_mvp_enabled_flag`.
    pub temporal_mvp: bool,
    /// `sample_adaptive_offset_enabled_flag`.
    pub sao: bool,
}

/// Skip profile_tier_level() for `max_sub_layers_minus1` sub-layers.
fn skip_profile_tier_level(r: &mut BitReader, max_sub_layers_minus1: u32) -> Result<()> {
    r.u(2)?; // profile_space
    r.u(1)?; // tier
    r.u(5)?; // profile_idc
    r.u(32)?; // compatibility flags
    r.u(4)?; // progressive/interlaced/non_packed/frame_only
    r.u(32)?; // reserved (43 bits) + inbld (1 bit), split for the 32-bit reader
    r.u(12)?;
    r.u(8)?; // level_idc
    let mut sub_layers = Vec::new();
    for _ in 0..max_sub_layers_minus1 {
        sub_layers.push((r.flag()?, r.flag()?));
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            r.u(2)?;
        }
    }
    for (sub_profile, sub_level) in sub_layers {
        if sub_profile {
            r.u(32)?;
            r.u(32)?;
            r.u(24)?; // 88 bits total
        }
        if sub_level {
            r.u(8)?;
        }
    }
    Ok(())
}

/// Parse one non-predicted st_ref_pic_set into absolute POC deltas.
fn parse_direct_st_rps(r: &mut BitReader) -> Result<Vec<RpsEntry>> {
    let num_negative = r.ue()?;
    let num_positive = r.ue()?;
    ensure!(
        num_negative <= 16 && num_positive <= 16,
        "implausible RPS size"
    );
    let mut entries = Vec::with_capacity((num_negative + num_positive) as usize);
    let mut poc = 0i32;
    for _ in 0..num_negative {
        let delta = r.ue()? + 1;
        poc -= delta as i32;
        entries.push(RpsEntry {
            delta_poc: poc,
            used: r.flag()?,
        });
    }
    let mut poc = 0i32;
    for _ in 0..num_positive {
        let delta = r.ue()? + 1;
        poc += delta as i32;
        entries.push(RpsEntry {
            delta_poc: poc,
            used: r.flag()?,
        });
    }
    Ok(entries)
}

/// Parse the SPS fields needed for slice-header parsing. Errors on any
/// feature outside the supported profile (callers must then skip slice
/// processing entirely).
pub fn parse_sps(sps_nal: &[u8]) -> Result<SpsInfo> {
    ensure!(sps_nal.len() > 2, "SPS NAL too short");
    let rbsp = unescape_rbsp(&sps_nal[2..]);
    let r = &mut BitReader::new(&rbsp);

    r.u(4)?; // sps_video_parameter_set_id
    let max_sub_layers_minus1 = r.u(3)?;
    r.u(1)?; // temporal_id_nesting
    skip_profile_tier_level(r, max_sub_layers_minus1)?;
    r.ue()?; // sps_seq_parameter_set_id
    let chroma_format_idc = r.ue()?;
    let separate_colour_plane = if chroma_format_idc == 3 {
        r.flag()?
    } else {
        false
    };
    r.ue()?; // pic_width_in_luma_samples
    r.ue()?; // pic_height_in_luma_samples
    if r.flag()? {
        // conformance window
        r.ue()?;
        r.ue()?;
        r.ue()?;
        r.ue()?;
    }
    r.ue()?; // bit_depth_luma_minus8
    r.ue()?; // bit_depth_chroma_minus8
    let poc_lsb_bits = r.ue()? + 4;
    ensure!(
        (4..=16).contains(&poc_lsb_bits),
        "bad log2_max_pic_order_cnt_lsb"
    );
    let sub_layer_ordering = r.flag()?;
    let start = if sub_layer_ordering {
        0
    } else {
        max_sub_layers_minus1
    };
    for _ in start..=max_sub_layers_minus1 {
        r.ue()?; // max_dec_pic_buffering_minus1
        r.ue()?; // num_reorder_pics
        r.ue()?; // max_latency_increase_plus1
    }
    r.ue()?; // log2_min_luma_coding_block_size_minus3
    r.ue()?; // log2_diff_max_min_luma_coding_block_size
    r.ue()?; // log2_min_luma_transform_block_size_minus2
    r.ue()?; // log2_diff_max_min_luma_transform_block_size
    r.ue()?; // max_transform_hierarchy_depth_inter
    r.ue()?; // max_transform_hierarchy_depth_intra
    if r.flag()? {
        // scaling_list_enabled
        ensure!(!r.flag()?, "SPS scaling list data not supported");
    }
    r.u(1)?; // amp_enabled
    let sao = r.flag()?;
    if r.flag()? {
        // pcm_enabled
        r.u(8)?; // sample bit depths
        r.ue()?;
        r.ue()?;
        r.u(1)?;
    }
    let num_st_rps = r.ue()?;
    ensure!(num_st_rps <= 64, "implausible num_short_term_ref_pic_sets");
    let mut short_term_rps = Vec::with_capacity(num_st_rps as usize);
    for i in 0..num_st_rps {
        if i > 0 {
            ensure!(
                !r.flag()?,
                "inter-RPS-predicted SPS ref pic set not supported"
            );
        }
        short_term_rps.push(parse_direct_st_rps(r)?);
    }
    let long_term_ref_pics_present = r.flag()?;
    let mut num_long_term_ref_pics_sps = 0;
    if long_term_ref_pics_present {
        num_long_term_ref_pics_sps = r.ue()?;
        for _ in 0..num_long_term_ref_pics_sps {
            r.u(poc_lsb_bits)?; // lt_ref_pic_poc_lsb_sps
            r.u(1)?; // used_by_curr_pic_lt_sps
        }
    }
    let temporal_mvp = r.flag()?;

    Ok(SpsInfo {
        poc_lsb_bits,
        separate_colour_plane,
        short_term_rps,
        long_term_ref_pics_present,
        num_long_term_ref_pics_sps,
        temporal_mvp,
        sao,
    })
}

/// The fields of a PPS that slice-header parsing depends on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PpsInfo {
    /// `dependent_slice_segments_enabled_flag`.
    pub dependent_slice_segments: bool,
    /// `output_flag_present_flag`.
    pub output_flag_present: bool,
    /// `num_extra_slice_header_bits`.
    pub num_extra_slice_header_bits: u32,
    /// `cabac_init_present_flag`.
    pub cabac_init_present: bool,
    /// `num_ref_idx_l0_default_active_minus1`.
    pub num_ref_idx_l0_default_minus1: u32,
    /// `lists_modification_present_flag`.
    pub lists_modification_present: bool,
    /// `weighted_pred_flag`.
    pub weighted_pred: bool,
    /// `tiles_enabled_flag`.
    pub tiles: bool,
    /// `entropy_coding_sync_enabled_flag` (wavefront parallel processing).
    pub entropy_coding_sync: bool,
    /// `pps_loop_filter_across_slices_enabled_flag`.
    pub loop_filter_across_slices: bool,
    /// `deblocking_filter_override_enabled_flag`.
    pub deblocking_filter_override_enabled: bool,
    /// `pps_deblocking_filter_disabled_flag`.
    pub pps_deblocking_disabled: bool,
    /// `pps_slice_chroma_qp_offsets_present_flag`.
    pub chroma_qp_offsets_present: bool,
    /// `slice_segment_header_extension_present_flag`.
    pub slice_header_extension: bool,
}

/// Parse the PPS fields needed for slice-header parsing.
pub fn parse_pps(pps_nal: &[u8]) -> Result<PpsInfo> {
    ensure!(pps_nal.len() > 2, "PPS NAL too short");
    let rbsp = unescape_rbsp(&pps_nal[2..]);
    let r = &mut BitReader::new(&rbsp);

    r.ue()?; // pps_pic_parameter_set_id
    r.ue()?; // pps_seq_parameter_set_id
    let dependent_slice_segments = r.flag()?;
    let output_flag_present = r.flag()?;
    let num_extra_slice_header_bits = r.u(3)?;
    r.u(1)?; // sign_data_hiding
    let cabac_init_present = r.flag()?;
    let num_ref_idx_l0_default_minus1 = r.ue()?;
    r.ue()?; // num_ref_idx_l1_default_active_minus1
    r.se()?; // init_qp_minus26
    r.u(1)?; // constrained_intra_pred
    r.u(1)?; // transform_skip_enabled
    if r.flag()? {
        // cu_qp_delta_enabled
        r.ue()?; // diff_cu_qp_delta_depth
    }
    r.se()?; // pps_cb_qp_offset
    r.se()?; // pps_cr_qp_offset
    let chroma_qp_offsets_present = r.flag()?;
    let weighted_pred = r.flag()?;
    r.u(1)?; // weighted_bipred_flag
    r.u(1)?; // transquant_bypass
    let tiles = r.flag()?;
    ensure!(!tiles, "tiled PPS not supported");
    let entropy_coding_sync = r.flag()?;
    let loop_filter_across_slices = r.flag()?;
    let mut deblocking_filter_override_enabled = false;
    let mut pps_deblocking_disabled = false;
    if r.flag()? {
        // deblocking_filter_control_present
        deblocking_filter_override_enabled = r.flag()?;
        pps_deblocking_disabled = r.flag()?;
        if !pps_deblocking_disabled {
            r.se()?; // beta_offset_div2
            r.se()?; // tc_offset_div2
        }
    }
    ensure!(!r.flag()?, "PPS scaling list data not supported");
    let lists_modification_present = r.flag()?;
    r.ue()?; // log2_parallel_merge_level_minus2
    let slice_header_extension = r.flag()?;

    Ok(PpsInfo {
        dependent_slice_segments,
        output_flag_present,
        num_extra_slice_header_bits,
        cabac_init_present,
        num_ref_idx_l0_default_minus1,
        lists_modification_present,
        weighted_pred,
        tiles,
        entropy_coding_sync,
        loop_filter_across_slices,
        deblocking_filter_override_enabled,
        pps_deblocking_disabled,
        chroma_qp_offsets_present,
        slice_header_extension,
    })
}

/// Summary of one parsed slice header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceInfo {
    /// `slice_pic_order_cnt_lsb`; `None` for IDR slices (their POC is 0).
    pub poc_lsb: Option<u32>,
    /// The slice's short-term RPS resolved to absolute POC deltas.
    pub rps: Vec<RpsEntry>,
}

/// What [`process_slice`] should do with the slice's short-term RPS.
enum RpsAction<'a> {
    /// Parse only; echo nothing.
    Inspect,
    /// Re-emit the header with this exact inline RPS (deltas relative to the
    /// current picture; negatives first, matching HEVC ordering rules).
    Replace(&'a [RpsEntry]),
}

/// Whether `nal_type` is an IDR slice (POC is inferred as 0, no RPS).
fn is_idr(nal_type: u8) -> bool {
    nal_type == 19 || nal_type == 20
}

/// Parse (and optionally rewrite) a slice segment header.
///
/// `nal` is the full NAL unit (2-byte header + escaped payload, no start
/// code). Returns the parsed [`SliceInfo`] and, when `action` is `Replace`,
/// the rewritten NAL with the slice payload copied verbatim (re-aligned and
/// re-escaped).
fn process_slice(
    nal: &[u8],
    sps: &SpsInfo,
    pps: &PpsInfo,
    action: RpsAction,
) -> Result<(SliceInfo, Option<Vec<u8>>)> {
    ensure!(nal.len() > 2, "slice NAL too short");
    let nal_type = (nal[0] >> 1) & 0x3F;
    ensure!(nal_type < 32, "not a slice NAL");
    let rbsp = unescape_rbsp(&nal[2..]);
    let r = &mut BitReader::new(&rbsp);
    let mut w = BitWriter::default();
    let rewrite = matches!(action, RpsAction::Replace(_));

    // Copy-through helpers: read a field and (when rewriting) echo it.
    macro_rules! copy_u {
        ($n:expr) => {{
            let v = r.u($n)?;
            if rewrite {
                w.put($n, v);
            }
            v
        }};
    }
    macro_rules! copy_ue {
        () => {{
            let v = r.ue()?;
            if rewrite {
                w.put_ue(v);
            }
            v
        }};
    }
    macro_rules! copy_se {
        () => {{
            // se(v) survives an ue-coded round trip: copy its ue code point.
            copy_ue!()
        }};
    }

    let first_slice = copy_u!(1) == 1;
    ensure!(
        first_slice,
        "dependent/non-first slice segments not supported"
    );
    if super::bitstream::is_irap(nal_type) {
        copy_u!(1); // no_output_of_prior_pics_flag
    }
    copy_ue!(); // slice_pic_parameter_set_id
    ensure!(
        !pps.dependent_slice_segments,
        "dependent slice segments not supported"
    );
    for _ in 0..pps.num_extra_slice_header_bits {
        copy_u!(1);
    }
    let slice_type = copy_ue!();
    ensure!(
        slice_type == 1 || slice_type == 2,
        "only P and I slices are supported"
    );
    let is_p = slice_type == 1;
    if pps.output_flag_present {
        copy_u!(1);
    }
    if sps.separate_colour_plane {
        copy_u!(2);
    }

    let mut poc_lsb = None;
    let mut rps: Vec<RpsEntry> = Vec::new();
    // NumPicTotalCurr drives whether the optional `ref_pic_list_modification`
    // syntax is present. It must be tracked separately for the SOURCE header
    // (whose bits we read) and the OUTPUT header (whose bits we write): a repair
    // that drops a missing used reference can cross the > 1 threshold, so the
    // source may carry the flag while the output must omit it (or vice versa).
    let mut src_num_pic_total_curr = 0u32;
    let mut out_num_pic_total_curr = 0u32;

    if !is_idr(nal_type) {
        poc_lsb = Some(copy_u!(sps.poc_lsb_bits));
        let num_sets = sps.short_term_rps.len() as u32;
        let sps_flag = r.flag()?;
        if sps_flag {
            let idx_bits = 32 - (num_sets.max(2) - 1).leading_zeros();
            let idx = if num_sets > 1 { r.u(idx_bits)? } else { 0 };
            rps = sps
                .short_term_rps
                .get(idx as usize)
                .ok_or_else(|| anyhow::anyhow!("slice references missing SPS RPS {idx}"))?
                .clone();
        } else {
            if num_sets > 0 {
                ensure!(
                    !r.flag()?,
                    "inter-RPS-predicted slice ref pic set not supported"
                );
            }
            rps = parse_direct_st_rps(r)?;
        }
        // The source header's optional fields hinge on the source RPS used count.
        src_num_pic_total_curr += rps.iter().filter(|e| e.used).count() as u32;
        // Re-emit the RPS (always as an inline, non-predicted set).
        if let RpsAction::Replace(new_rps) = action {
            w.put_flag(false); // short_term_ref_pic_set_sps_flag = 0
            if num_sets > 0 {
                w.put_flag(false); // inter_ref_pic_set_prediction_flag = 0
            }
            // HEVC orders RPS entries by increasing distance from the current
            // picture; sort so callers can pass entries in any order.
            let mut negatives: Vec<&RpsEntry> =
                new_rps.iter().filter(|e| e.delta_poc < 0).collect();
            negatives.sort_by_key(|e| -e.delta_poc);
            let mut positives: Vec<&RpsEntry> =
                new_rps.iter().filter(|e| e.delta_poc > 0).collect();
            positives.sort_by_key(|e| e.delta_poc);
            w.put_ue(negatives.len() as u32);
            w.put_ue(positives.len() as u32);
            let mut prev = 0i32;
            for e in &negatives {
                w.put_ue((prev - e.delta_poc - 1) as u32);
                w.put_flag(e.used);
                prev = e.delta_poc;
            }
            let mut prev = 0i32;
            for e in &positives {
                w.put_ue((e.delta_poc - prev - 1) as u32);
                w.put_flag(e.used);
                prev = e.delta_poc;
            }
            out_num_pic_total_curr += new_rps.iter().filter(|e| e.used).count() as u32;
        } else {
            out_num_pic_total_curr += rps.iter().filter(|e| e.used).count() as u32;
        }
        if sps.long_term_ref_pics_present {
            if sps.num_long_term_ref_pics_sps > 0 {
                ensure!(copy_ue!() == 0, "long-term SPS references not supported");
            }
            ensure!(copy_ue!() == 0, "long-term slice references not supported");
        }
        if sps.temporal_mvp {
            copy_u!(1); // slice_temporal_mvp_enabled_flag
        }
    }

    let mut sao_luma = false;
    let mut sao_chroma = false;
    if sps.sao {
        sao_luma = copy_u!(1) == 1;
        sao_chroma = copy_u!(1) == 1;
    }

    if is_p {
        let num_ref_l0 = if copy_u!(1) == 1 {
            copy_ue!() + 1
        } else {
            pps.num_ref_idx_l0_default_minus1 + 1
        };
        if pps.lists_modification_present {
            // Read per the source count and write per the output count: a repair
            // that crosses the > 1 threshold must still consume the source bit.
            let src_present = src_num_pic_total_curr > 1;
            let out_present = out_num_pic_total_curr > 1;
            // A repair only drops used references, so out-present without
            // src-present is impossible.
            ensure!(
                src_present || !out_present,
                "RPS repair added a used reference"
            );
            if src_present {
                ensure!(r.u(1)? == 0, "ref pic list modification not supported");
                if rewrite && out_present {
                    w.put(1, 0);
                }
            }
        }
        if pps.cabac_init_present {
            copy_u!(1);
        }
        // collocated_ref_idx is present only when the slice enables temporal
        // MVP and more than one reference is active. The low-latency encoders
        // covered here use a single active reference
        // (num_ref_idx_l0_active_minus1 = 0), so the field is never present;
        // reject anything else rather than mis-parse.
        ensure!(num_ref_l0 == 1, "multiple active references not supported");
        ensure!(!pps.weighted_pred, "weighted prediction not supported");
        copy_ue!(); // five_minus_max_num_merge_cand
    }
    copy_se!(); // slice_qp_delta
    if pps.chroma_qp_offsets_present {
        copy_se!();
        copy_se!();
    }
    let mut deblocking_disabled = pps.pps_deblocking_disabled;
    if pps.deblocking_filter_override_enabled && copy_u!(1) == 1 {
        deblocking_disabled = copy_u!(1) == 1;
        if !deblocking_disabled {
            copy_se!();
            copy_se!();
        }
    }
    if pps.loop_filter_across_slices && (sao_luma || sao_chroma || !deblocking_disabled) {
        copy_u!(1); // slice_loop_filter_across_slices_enabled_flag
    }
    if pps.tiles || pps.entropy_coding_sync {
        let num_entry_points = copy_ue!();
        if num_entry_points > 0 {
            let offset_len_minus1 = copy_ue!();
            ensure!(offset_len_minus1 < 32, "bad entry point offset length");
            for _ in 0..num_entry_points {
                copy_u!(offset_len_minus1 + 1);
            }
        }
    }
    ensure!(
        !pps.slice_header_extension,
        "slice header extensions not supported"
    );

    let info = SliceInfo { poc_lsb, rps };
    if !rewrite {
        return Ok((info, None));
    }

    // byte_alignment(): consume the source's alignment, emit our own, then
    // copy the (byte-aligned) slice payload verbatim.
    ensure!(r.u(1)? == 1, "missing alignment_bit_equal_to_one");
    while !r.byte_aligned() {
        ensure!(r.u(1)? == 0, "nonzero alignment bit");
    }
    w.put(1, 1);
    while !w.byte_aligned() {
        w.put(1, 0);
    }
    let payload_start = r.pos / 8;
    let mut out_rbsp = w.data;
    out_rbsp.extend_from_slice(&rbsp[payload_start..]);

    let mut out = Vec::with_capacity(nal.len() + 8);
    out.extend_from_slice(&nal[..2]);
    out.extend_from_slice(&escape_rbsp(&out_rbsp));
    Ok((info, Some(out)))
}

/// Parse a slice header without modifying it.
pub fn parse_slice(nal: &[u8], sps: &SpsInfo, pps: &PpsInfo) -> Result<SliceInfo> {
    process_slice(nal, sps, pps, RpsAction::Inspect).map(|(info, _)| info)
}

/// Rewrite a slice header's short-term RPS to exactly `entries` (in any
/// order; they are re-sorted per HEVC ordering rules), leaving every other
/// header field and the slice payload byte-for-byte intact. The RPS is always
/// re-emitted as an inline (non-SPS-indexed) set, which is semantically
/// equivalent.
pub fn rewrite_rps(
    nal: &[u8],
    sps: &SpsInfo,
    pps: &PpsInfo,
    entries: &[RpsEntry],
) -> Result<Vec<u8>> {
    let (_, rewritten) = process_slice(nal, sps, pps, RpsAction::Replace(entries))?;
    rewritten.ok_or_else(|| anyhow::anyhow!("rewrite produced no output"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SPS/PPS matching the profile of NVENC's ultra-low-latency HEVC streams
    /// as seen in production (9-bit POC lsb, no SPS-level RPS, no
    /// SAO/TMVP/WPP; PPS defaults).
    fn nvenc_like_sps() -> SpsInfo {
        SpsInfo {
            poc_lsb_bits: 9,
            ..SpsInfo::default()
        }
    }

    /// Shorthand for an [`RpsEntry`] in the tests below.
    fn e(delta_poc: i32, used: bool) -> RpsEntry {
        RpsEntry { delta_poc, used }
    }

    /// Build a TRAIL_R P-slice NAL with the structure NVENC emits: inline RPS
    /// entries `(delta_poc_minus1, used_by_curr_pic)`, one active reference,
    /// then an opaque payload standing in for the CABAC slice data. Under a PPS
    /// with `lists_modification_present`, a slice whose RPS has more than one
    /// used reference carries `ref_pic_list_modification_flag_l0` (emitted as
    /// 0, "no modification"), matching HEVC syntax order.
    fn build_nvenc_like_p_slice(
        poc: u32,
        entries: &[(u32, bool)],
        payload: &[u8],
        pps: &PpsInfo,
    ) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.put(1, 1); // first_slice_segment_in_pic_flag
        w.put_ue(0); // slice_pic_parameter_set_id
        w.put_ue(1); // slice_type = P
        w.put(9, poc); // slice_pic_order_cnt_lsb
        w.put_flag(false); // short_term_ref_pic_set_sps_flag (inline RPS follows)
        w.put_ue(entries.len() as u32); // num_negative_pics
        w.put_ue(0); // num_positive_pics
        for &(delta_minus1, used) in entries {
            w.put_ue(delta_minus1);
            w.put_flag(used);
        }
        w.put_flag(true); // num_ref_idx_active_override_flag
        w.put_ue(0); // num_ref_idx_l0_active_minus1
        // ref_pic_list_modification() is present when NumPicTotalCurr > 1.
        if pps.lists_modification_present && entries.iter().filter(|(_, used)| *used).count() > 1 {
            w.put_flag(false); // ref_pic_list_modification_flag_l0
        }
        w.put_ue(3); // five_minus_max_num_merge_cand
        w.put_ue(3); // slice_qp_delta (se code point)
        w.put(1, 1); // byte_alignment: alignment_bit_equal_to_one
        while !w.byte_aligned() {
            w.put(1, 0);
        }
        let mut rbsp = w.data;
        rbsp.extend_from_slice(payload);
        let mut nal = vec![0x02, 0x01]; // TRAIL_R
        nal.extend_from_slice(&escape_rbsp(&rbsp));
        nal
    }

    /// The RPS shape of every captured failing NVENC P-frame: the picture it
    /// predicts from plus unused DPB keep-alive entries.
    const NVENC_RPS: [(u32, bool); 4] = [(0, true), (0, false), (0, false), (0, false)];

    #[test]
    fn rewriting_with_unchanged_rps_is_byte_identical() {
        let (sps, pps) = (nvenc_like_sps(), PpsInfo::default());
        // Payload includes start-code-like bytes to exercise emulation prevention.
        let payload = [
            0x9E, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x7F, 0x00, 0x00, 0x00,
        ];
        let nal = build_nvenc_like_p_slice(44, &NVENC_RPS, &payload, &pps);
        let info = parse_slice(&nal, &sps, &pps).unwrap();
        let rewritten = rewrite_rps(&nal, &sps, &pps, &info.rps).unwrap();
        assert_eq!(rewritten, nal);
    }

    #[test]
    fn rewrites_rps_preserving_the_slice_payload() {
        let (sps, pps) = (nvenc_like_sps(), PpsInfo::default());
        let payload = [0x9E, 0x00, 0x00, 0x01, 0x42, 0x00, 0x00, 0x00, 0x55];
        let nal = build_nvenc_like_p_slice(44, &NVENC_RPS, &payload, &pps);

        // Emulate the decoder's concealment after a recovery keyframe: the
        // DPB holds only POC 41, so the used reference (POC 43) is remapped
        // to it and the never-decoded keep-alives are dropped.
        let repaired = rewrite_rps(&nal, &sps, &pps, &[e(-3, true)]).expect("rewrite succeeds");
        let info = parse_slice(&repaired, &sps, &pps).expect("repaired slice parses");
        assert_eq!(info.poc_lsb, Some(44));
        assert_eq!(info.rps, vec![e(-3, true)]);
        // The slice payload survives byte-for-byte (it is byte-aligned both
        // before and after the header rewrite).
        let tail = unescape_rbsp(&repaired[2..]);
        assert!(
            tail.ends_with(&payload),
            "payload was disturbed: {tail:02x?}"
        );

        // Entries passed in any order come out in canonical HEVC order.
        let reordered =
            rewrite_rps(&nal, &sps, &pps, &[e(-4, false), e(-1, true), e(-2, false)]).unwrap();
        let info = parse_slice(&reordered, &sps, &pps).unwrap();
        assert_eq!(info.rps, vec![e(-1, true), e(-2, false), e(-4, false)]);
    }

    /// Regression: under `lists_modification_present`, whether
    /// `ref_pic_list_modification_flag_l0` is present depends on NumPicTotalCurr.
    /// A repair that drops a missing used reference can cross the 2->1 boundary,
    /// so the SOURCE header carries the flag but the OUTPUT must omit it. The
    /// gate must consume the source bit regardless, or every later field parses
    /// one bit off and the payload is corrupted.
    #[test]
    fn rewrites_across_the_list_modification_flag_boundary() {
        let sps = nvenc_like_sps();
        let pps = PpsInfo {
            lists_modification_present: true,
            ..PpsInfo::default()
        };
        let payload = [0x9E, 0x00, 0x00, 0x01, 0x42, 0x00, 0x00, 0x00, 0x55];
        // Source RPS: two used references => NumPicTotalCurr = 2 => the source
        // header carries ref_pic_list_modification_flag_l0 (value 0).
        let nal = build_nvenc_like_p_slice(44, &[(0, true), (0, true)], &payload, &pps);
        let info = parse_slice(&nal, &sps, &pps).expect("source slice parses");
        assert_eq!(info.rps, vec![e(-1, true), e(-2, true)]);

        // Repair down to a single used reference: NumPicTotalCurr crosses 2 -> 1,
        // so the output header must omit the flag while the source bit is still
        // consumed. A desync here would corrupt the payload below.
        let repaired = rewrite_rps(&nal, &sps, &pps, &[e(-1, true)]).expect("rewrite succeeds");
        let info = parse_slice(&repaired, &sps, &pps).expect("repaired slice parses");
        assert_eq!(info.poc_lsb, Some(44));
        assert_eq!(info.rps, vec![e(-1, true)]);
        let tail = unescape_rbsp(&repaired[2..]);
        assert!(
            tail.ends_with(&payload),
            "payload was disturbed: {tail:02x?}"
        );

        // Identity: rewriting with the unchanged 2-used-entry RPS keeps the flag
        // and stays byte-identical.
        let identity =
            rewrite_rps(&nal, &sps, &pps, &[e(-1, true), e(-2, true)]).expect("rewrite succeeds");
        assert_eq!(
            identity, nal,
            "identity rewrite under lists_modification_present must be byte-exact"
        );
    }
}
