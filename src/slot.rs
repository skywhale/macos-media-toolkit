//! Frame plumbing shared by the macOS backends: copying Core Video pixel
//! buffers into tightly packed BGRA frames, and handing them from the
//! framework's dispatch queue to the consumer's thread.

use crate::BgraFrame;
use cidre::{cm, cv};
use std::{
    sync::{Condvar, Mutex},
    time::Duration,
};

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

/// Latest-frame slot: the producer (dispatch queue) overwrites, the consumer
/// blocks on the [`Condvar`].
pub(crate) struct FrameSlot {
    frame: Mutex<Option<BgraFrame>>,
    available: Condvar,
}

impl FrameSlot {
    pub(crate) fn new() -> Self {
        Self {
            frame: Mutex::new(None),
            available: Condvar::new(),
        }
    }

    /// Store the newest frame (dropping any unconsumed one) and wake a blocked
    /// consumer. Called on the delegate's dispatch queue only.
    pub(crate) fn store(&self, frame: BgraFrame) {
        if let Ok(mut slot) = self.frame.lock() {
            *slot = Some(frame);
            self.available.notify_one();
        }
    }

    /// Block up to `timeout` for a frame, returning (and clearing) it as soon as
    /// one is available, or `None` on timeout.
    pub(crate) fn take_blocking(&self, timeout: Duration) -> Option<BgraFrame> {
        let slot = self.frame.lock().expect("capture frame slot poisoned");
        let (mut slot, _) = self
            .available
            .wait_timeout_while(slot, timeout, |s| s.is_none())
            .expect("capture frame slot poisoned");
        slot.take()
    }
}

/// Copy the sample buffer's image into the slot as tightly packed 32BGRA. Idle/status-only
/// buffers carry no image and are skipped.
pub(crate) fn store_frame(slot: &FrameSlot, sample_buf: &cm::SampleBuf) {
    let Some(image_buf) = sample_buf.image_buf() else {
        return;
    };
    let mut pb = image_buf.retained();

    let Some(frame) = copy_pixel_buf_bgra(&mut pb) else {
        return;
    };

    slot.store(frame);
}
