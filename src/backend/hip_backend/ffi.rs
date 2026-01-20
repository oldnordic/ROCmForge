//! HIP FFI bindings
//!
//! FFI declarations below are bound to ROCm HIP API.
//! All functions are actively used through wrapper methods in HipBackend.
//! The dead_code allowance is needed because FFI symbols appear unused
//! to the compiler (they're only called through unsafe blocks).

use std::ffi::c_void;

#[link(name = "amdhip64")]
#[allow(dead_code)]
extern "C" {
    pub fn hipInit(flags: u32) -> i32;
    pub fn hipGetDeviceCount(count: *mut i32) -> i32;
    pub fn hipGetDeviceProperties(props: *mut super::device::HipDeviceProp, deviceId: i32) -> i32;
    pub fn hipSetDevice(deviceId: i32) -> i32;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn hipFree(ptr: *mut c_void) -> i32;
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub fn hipMemcpy2D(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
    ) -> i32;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn hipMemcpyHtoD(dst: *mut c_void, src: *const c_void, count: usize) -> i32;
    pub fn hipMemcpyDtoH(dst: *mut c_void, src: *const c_void, count: usize) -> i32;
    pub fn hipStreamCreate(stream: *mut *mut c_void) -> i32;
    pub fn hipStreamDestroy(stream: *mut c_void) -> i32;
    pub fn hipStreamSynchronize(stream: *mut c_void) -> i32;
    pub fn hipEventCreate(event: *mut *mut c_void) -> i32;
    pub fn hipEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    pub fn hipEventDestroy(event: *mut c_void) -> i32;
    pub fn hipEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    pub fn hipEventSynchronize(event: *mut c_void) -> i32;
    pub fn hipEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    pub fn hipModuleLoad(module: *mut *mut c_void, path: *const i8) -> i32;
    pub fn hipModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> i32;
    pub fn hipModuleUnload(module: *mut c_void) -> i32;
    pub fn hipModuleGetFunction(func: *mut *mut c_void, module: *mut c_void, name: *const i8) -> i32;
    pub fn hipModuleLaunchKernel(
        func: *mut c_void,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        stream: *mut c_void,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    pub fn hipGetLastError() -> i32;
    pub fn hipGetErrorString(error: i32) -> *const i8;
    pub fn hipDeviceSynchronize() -> i32;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    pub fn hipMemset(dst: *mut c_void, value: i32, count: usize) -> i32;
}

/// HIP memory copy kinds
pub const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

/// HIP success code
pub const HIP_SUCCESS: i32 = 0;

// HIP event creation flags
/// Default event creation flag (no special flags)
#[allow(dead_code)] // FFI constant - part of HIP API, available for future use
pub const HIP_EVENT_DEFAULT: u32 = 0x0;

/// Event flag to disable timing data collection
pub const HIP_EVENT_DISABLE_TIMING: u32 = 0x1;

/// Event flag to enable timing data collection
#[allow(dead_code)] // FFI constant - part of HIP API, available for future use
pub const HIP_EVENT_RECORD_TIMING: u32 = 0x2;
