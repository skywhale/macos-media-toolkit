//! Encode synthetic frames to HEVC and decode them back, exercising the
//! keyframe cadence, a mid-stream resolution change, and the state a decoder
//! starting mid-stream sees. The stream is also written to `encoded.h265`, which
//! `hevc_dump` reads.
//!
//! ```text
//! cargo run --example encode_decode
//! cargo run --example hevc_dump -- encoded.h265
//! ```

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("VideoToolbox encode/decode requires macOS");
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use macos_media_toolkit::{
        BgraFrame,
        videotoolbox::{DecodeError, DecoderConfig, EncoderConfig, HevcDecoder, HevcEncoder},
    };

    /// A moving diagonal gradient, so successive frames differ and inter frames
    /// carry real residual.
    fn gradient(width: u32, height: u32, phase: u32) -> BgraFrame {
        let mut bgra = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let v = (x + y + phase) as u8;
                bgra.extend_from_slice(&[v, v.wrapping_mul(3), 0x40, 0xFF]);
            }
        }
        BgraFrame {
            bgra,
            width,
            height,
        }
    }

    env_logger::init();

    let mut stream = Vec::new();
    let mut encoder = HevcEncoder::new(&EncoderConfig {
        width: 640,
        height: 480,
        frame_rate: 30.0,
        average_bitrate_bps: 4_000_000,
        keyframe_interval: 8,
    })?;
    let mut decoder = HevcDecoder::new(&DecoderConfig::default());

    for i in 0..12u32 {
        let frame = gradient(640, 480, i * 4);
        let encoded = encoder.encode(&frame.bgra, frame.width, frame.height, i == 0)?;
        let decoded = decoder.decode(&encoded.annex_b)?;
        stream.extend_from_slice(&encoded.annex_b);

        println!(
            "frame {i:2}: {:6} B {:5}  ->  {}x{} {}",
            encoded.annex_b.len(),
            if encoded.is_keyframe { "key" } else { "inter" },
            decoded.frame.width,
            decoded.frame.height,
            if decoded.is_keyframe { "key" } else { "inter" },
        );
    }

    // A new resolution rebuilds the session, which makes that frame a keyframe;
    // its parameter sets rebuild the decoder's session in turn.
    let frame = gradient(320, 240, 0);
    let encoded = encoder.encode(&frame.bgra, frame.width, frame.height, false)?;
    let decoded = decoder.decode(&encoded.annex_b)?;
    println!(
        "resized: {}x{}, keyframe {}",
        decoded.frame.width, decoded.frame.height, decoded.is_keyframe
    );

    std::fs::write("encoded.h265", &stream)?;
    println!("wrote encoded.h265 ({} bytes)", stream.len());

    // A decoder joining mid-stream has no parameter sets and no reference
    // pictures, so it rejects inter frames until a keyframe arrives.
    let mut joining = HevcDecoder::new(&DecoderConfig::default());
    let inter = encoder.encode(&gradient(320, 240, 8).bgra, 320, 240, false)?;
    match joining.decode(&inter.annex_b) {
        Err(DecodeError::MissingKeyframe) => println!("mid-stream join: keyframe requested"),
        other => anyhow::bail!("expected MissingKeyframe, got {other:?}"),
    }

    let key = encoder.encode(&gradient(320, 240, 12).bgra, 320, 240, true)?;
    let recovered = joining.decode(&key.annex_b)?;
    println!(
        "mid-stream join: recovered on a keyframe, {}x{}",
        recovered.frame.width, recovered.frame.height
    );

    Ok(())
}
