//! Print the NAL structure of an HEVC Annex B stream, with slice headers
//! resolved once the stream's parameter sets have been seen.
//!
//! ```text
//! cargo run --example hevc_dump -- stream.h265
//! ```
//!
//! Uses only the `hevc` module, so it builds and runs on any platform.

use anyhow::{Context, Result};
use macos_media_toolkit::hevc;
use std::io::{BufWriter, Write};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: hevc_dump <file.h265>")?;
    let data = std::fs::read(&path).with_context(|| format!("reading {path}"))?;

    let nals = hevc::annex_b_nal_units(&data);
    // A capture file holds enough NALs that a flush per line would dominate.
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    writeln!(out, "{} NAL units in {} bytes", nals.len(), data.len())?;

    let headers = match hevc::parameter_sets_from_nals(&nals) {
        Some(sets) => {
            let sps = hevc::parse_sps(&sets.sps).context("parsing SPS")?;
            let pps = hevc::parse_pps(&sets.pps).context("parsing PPS")?;
            writeln!(
                out,
                "parameter sets: vps {} B, sps {} B, pps {} B, {} POC lsb bits\n",
                sets.vps.len(),
                sets.sps.len(),
                sets.pps.len(),
                sps.poc_lsb_bits
            )?;
            Some((sps, pps))
        }
        None => {
            writeln!(
                out,
                "no complete VPS/SPS/PPS set; slice headers cannot be parsed\n"
            )?;
            None
        }
    };

    for (index, nal) in nals.iter().enumerate() {
        let Some(kind) = hevc::nal_unit_type(nal) else {
            continue;
        };
        // Types 0..=21 are the picture-carrying (VCL) NALs; the rest describe the
        // stream. See ITU-T H.265 Table 7-1.
        let label = match kind {
            _ if hevc::is_parameter_set(nal) => "parameter set",
            _ if hevc::is_irap(kind) => "keyframe",
            0..=21 => "slice",
            35 => "access unit delim",
            39 | 40 => "SEI",
            _ => "other",
        };
        write!(
            out,
            "[{index:4}] type {kind:2} {label:13} {:7} B",
            nal.len()
        )?;

        // Non-slice NALs and slice layouts outside the supported profile fail to
        // parse; the framing above is still reported for them.
        if let Some((sps, pps)) = &headers
            && let Ok(info) = hevc::parse_slice(nal, sps, pps)
        {
            match info.poc_lsb {
                Some(poc) => write!(out, "  poc_lsb {poc:<6}")?,
                None => write!(out, "  poc_lsb {:<6}", "-")?,
            }
            write!(out, " rps [")?;
            for (position, entry) in info.rps.iter().enumerate() {
                let separator = if position == 0 { "" } else { " " };
                let used = if entry.used { "*" } else { "" };
                write!(out, "{separator}{}{used}", entry.delta_poc)?;
            }
            write!(out, "]")?;
        }
        writeln!(out)?;
    }

    writeln!(
        out,
        "\n(rps entries are POC deltas; * marks used_by_curr_pic)"
    )?;
    Ok(())
}
