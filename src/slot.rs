//! Frame plumbing for the macOS backends: copying Core Video pixel buffers
//! into tightly packed BGRA frames.

use crate::BgraFrame;
use cidre::cv;

/// Copy a `cv::PixelBuf`'s 32BGRA rows into a tightly packed [`BgraFrame`],
/// honoring the buffer's `bytes_per_row` stride (the result has no row
/// padding). `None` if the buffer can't be locked or exposes no base address.
pub(crate) fn copy_pixel_buf_bgra(pb: &mut cv::PixelBuf) -> Option<BgraFrame> {
    let flags = cv::pixel_buffer::LockFlags::READ_ONLY;

    // SAFETY: standard CoreVideo lock/copy/unlock. The base address is valid only
    // while the buffer is locked; every row is copied out before unlocking, and
    // reads stay within `height` rows of `width * 4` bytes at `bytes_per_row`
    // stride.
    unsafe {
        if pb.lock_base_addr(flags).result().is_err() {
            return None;
        }
        let base = pb.base_address() as *const u8;
        if base.is_null() {
            pb.unlock_lock_base_addr(flags);
            return None;
        }
        let width = pb.width();
        let height = pb.height();
        let bytes_per_row = pb.bytes_per_row();
        let row_bytes = width * 4;
        let mut bgra = Vec::with_capacity(row_bytes * height);
        for row in 0..height {
            let src = base.add(row * bytes_per_row);
            bgra.extend_from_slice(std::slice::from_raw_parts(src, row_bytes));
        }
        pb.unlock_lock_base_addr(flags);
        Some(BgraFrame {
            bgra,
            width: width as u32,
            height: height as u32,
        })
    }
}
