//! Aligned buffer allocation for SIMD kernels and GPU staging.
//!
//! Standard `Vec` does not guarantee alignment beyond the type's natural alignment.
//! This module provides `AlignedVec<T>` which can be allocated with arbitrary
//! alignment (e.g. 64-byte for AVX-512, 256-byte for GPU DMA, 512-byte for
//! Linux `O_DIRECT`).

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::{Deref, DerefMut};

/// Common alignment constants.
pub const ALIGN_AVX2: usize = 32;
pub const ALIGN_AVX512: usize = 64;
pub const ALIGN_CACHE_LINE: usize = 64;
pub const ALIGN_GPU_STAGING: usize = 256;
pub const ALIGN_ODIRECT: usize = 512;

/// A contiguous buffer aligned to a specified boundary.
///
/// # Type Parameters
/// * `T` — Element type (must be `Copy` for safe zeroed allocation)
///
/// # Safety
/// The buffer is allocated with `alloc_zeroed` and deallocated on `Drop`.
/// It is safe to transmute to SIMD register types when alignment is satisfied.
pub struct AlignedVec<T> {
    ptr: *mut T,
    len: usize,
    layout: Layout,
}

impl<T: Copy> AlignedVec<T> {
    /// Allocate a zeroed buffer with `len` elements and at least `align` bytes alignment.
    ///
    /// # Panics
    /// Panics if the allocation fails or if `align` is not a power of two.
    pub fn new_zeroed(len: usize, align: usize) -> Self {
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        let size = std::mem::size_of::<T>().saturating_mul(len);
        // Round up to alignment boundary so the entire buffer is aligned
        let padded = size.checked_next_multiple_of(align).unwrap_or(size);
        let layout =
            Layout::from_size_align(padded, align).expect("invalid layout for aligned allocation");
        let ptr = unsafe { alloc_zeroed(layout) } as *mut T;
        assert!(!ptr.is_null(), "aligned allocation failed");
        Self { ptr, len, layout }
    }

    /// Length in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if length is zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Pointer to the start of the buffer (aligned).
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Mutable pointer to the start of the buffer (aligned).
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                dealloc(self.ptr as *mut u8, self.layout);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

// SAFETY: AlignedVec owns its allocation and the pointer is stable
unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_vec_f32_64byte() {
        let v = AlignedVec::<f32>::new_zeroed(100, ALIGN_AVX512);
        assert_eq!(v.len(), 100);
        assert!(!v.is_empty());
        assert!(v.as_ptr() as usize % ALIGN_AVX512 == 0);
        // All zeroed
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn aligned_vec_u8_512byte() {
        let v = AlignedVec::<u8>::new_zeroed(4096, ALIGN_ODIRECT);
        assert_eq!(v.len(), 4096);
        assert!(v.as_ptr() as usize % ALIGN_ODIRECT == 0);
    }

    #[test]
    fn aligned_vec_empty() {
        let v = AlignedVec::<f32>::new_zeroed(0, ALIGN_AVX2);
        assert!(v.is_empty());
    }

    #[test]
    fn aligned_vec_deref_mut() {
        let mut v = AlignedVec::<f32>::new_zeroed(8, ALIGN_AVX2);
        for (i, x) in v.iter_mut().enumerate() {
            *x = i as f32;
        }
        assert_eq!(v[3], 3.0);
    }
}
