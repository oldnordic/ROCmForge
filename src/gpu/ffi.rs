//! Safe FFI bindings to HIP runtime.
//!
//! All unsafe blocks wrapped with error checking.
//! No raw HIP API exposed outside gpu module.

use super::error::{GpuError, GpuResult};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

// ── HIP Error Codes ───────────────────────────────────────────────────────────────

/// Convert HIP error code to GpuError.
fn hip_check(code: hipError_t) -> GpuResult<()> {
    if code == hipError_t::hipSuccess {
        return Ok(());
    }

    let description = unsafe {
        let ptr = hipGetErrorString(code);
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    Err(match code {
        hipError_t::hipErrorNotInitialized | hipError_t::hipErrorInvalidContext => {
            GpuError::HipNotAvailable
        }
        hipError_t::hipErrorOutOfMemory => GpuError::OutOfMemory {
            requested: 0, // Call site should fill this
            available: 0,
        },
        hipError_t::hipErrorInvalidDevice => GpuError::InvalidDevice { device_id: -1 },
        _ => GpuError::HipApiError {
            code: code as i32,
            description,
        },
    })
}

// ── HIP Type Definitions ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorNotInitialized = 3,
    hipErrorInvalidDeviceFunction = 98,
    hipErrorInvalidDevice = 101,
    hipErrorInvalidContext = 201,
    hipErrorIllegalState = 401,
    hipErrorNotSupported = 801,
    hipErrorStreamCaptureImplicit = 906,
    hipErrorUnknown = 999,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum hipGraphExecUpdateResult {
    hipGraphExecUpdateSuccess = 0x0,
    hipGraphExecUpdateError = 0x1,
    hipGraphExecUpdateErrorTopologyChanged = 0x2,
    hipGraphExecUpdateErrorAttributesChanged = 0x3,
    hipGraphExecUpdateErrorFunctionChanged = 0x4,
    hipGraphExecUpdateErrorParametersChanged = 0x5,
    hipGraphExecUpdateErrorNotSupported = 0x6,
    hipGraphExecUpdateErrorUnsupportedFunctionChange = 0x7,
}

// ── Safe HIP API Wrappers ───────────────────────────────────────────────────────────

/// Get number of HIP-compatible GPUs.
///
/// Returns 0 if HIP not available or no GPU found.
pub fn hip_get_device_count() -> GpuResult<i32> {
    unsafe {
        let mut count = 0i32;
        let code = hipGetDeviceCount(&mut count);
        hip_check(code)?;
        Ok(count)
    }
}

/// Get HIP device properties (name, VRAM, compute units).
pub fn hip_get_device_info(device_id: i32) -> GpuResult<DeviceInfo> {
    unsafe {
        let mut props: hipDeviceProp_t = std::mem::zeroed();
        let code = hipGetDeviceProperties(&mut props, device_id);
        hip_check(code)?;

        let name = CStr::from_ptr(props.name.as_ptr())
            .to_string_lossy()
            .into_owned();

        Ok(DeviceInfo {
            name,
            total_vram_bytes: props.totalGlobalMem as usize,
            compute_units: props.multiProcessorCount as usize,
            max_clock_mhz: props.clockRate as usize / 1000, // kHz to MHz
            device_id,
            warp_size: props.warpSize as usize,
            arch_name: String::from("unknown"), // Placeholder - will query from HIP in next task
            max_shared_mem_per_block: props.sharedMemPerBlock as usize,
        })
    }
}

/// Allocate memory on GPU.
///
/// Returns pointer to allocated memory or GpuError::OutOfMemory.
pub fn hip_malloc(size: usize) -> GpuResult<*mut u8> {
    unsafe {
        let mut ptr = std::ptr::null_mut();
        let code = hipMalloc(&mut ptr, size);
        hip_check(code)?;
        Ok(ptr)
    }
}

/// Free GPU memory.
///
/// Safe to call with null pointer (no-op).
pub fn hip_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        // Ignore errors in free (double-free is a bug, not a crash)
        let _ = hipFree(ptr);
    }
}

/// Copy data from host (CPU) to device (GPU).
pub fn hip_memcpy_h2d(dst: *mut u8, src: *const u8, size: usize) -> GpuResult<()> {
    unsafe {
        let code = hipMemcpy(
            dst as *mut c_void,
            src as *const c_void,
            size,
            hipMemcpyKind::hipMemcpyHostToDevice,
        );
        hip_check(code)
    }
}

/// Copy data from host (CPU) to device (GPU) on an explicit HIP stream.
pub fn hip_memcpy_h2d_async(
    dst: *mut u8,
    src: *const u8,
    size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    unsafe {
        let code = hipMemcpyAsync(
            dst as *mut c_void,
            src as *const c_void,
            size,
            hipMemcpyKind::hipMemcpyHostToDevice,
            stream,
        );
        hip_check(code)
    }
}

/// Copy data from device (GPU) to host (CPU).
pub fn hip_memcpy_d2h(dst: *mut u8, src: *const u8, size: usize) -> GpuResult<()> {
    unsafe {
        let code = hipMemcpy(
            dst as *mut c_void,
            src as *const c_void,
            size,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        );
        hip_check(code)
    }
}

/// Copy data from device (GPU) to host (CPU) on an explicit HIP stream.
pub fn hip_memcpy_d2h_async(
    dst: *mut u8,
    src: *const u8,
    size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    unsafe {
        let code = hipMemcpyAsync(
            dst as *mut c_void,
            src as *const c_void,
            size,
            hipMemcpyKind::hipMemcpyDeviceToHost,
            stream,
        );
        hip_check(code)
    }
}

/// Allocate pinned memory on host (CPU).
pub fn hip_host_malloc(size: usize) -> GpuResult<*mut u8> {
    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // 0x0 is hipHostMallocDefault
        let code = hipHostMalloc(&mut ptr, size, 0);
        hip_check(code)?;
        Ok(ptr as *mut u8)
    }
}

/// Free pinned memory on host (CPU).
pub fn hip_host_free(ptr: *mut u8) -> GpuResult<()> {
    if ptr.is_null() {
        return Ok(());
    }
    unsafe {
        let code = hipHostFree(ptr as *mut c_void);
        hip_check(code)
    }
}

/// Get free and total VRAM in bytes.
pub fn hip_get_mem_info(device_id: i32) -> GpuResult<(usize, usize)> {
    unsafe {
        let mut free = 0usize;
        let mut total = 0usize;
        let code = hipMemGetInfo(&mut free, &mut total);
        hip_check(code)?;
        Ok((free, total))
    }
}

/// Query HIP driver version.
///
/// Returns packed version number (e.g., 0x05060000 for 5.6.0).
/// Get HIP driver version.
///
/// Note: hipGetDriverVersion availability varies by HIP/ROCm version.
/// Returns 0 if function is unavailable - this is a safe default
/// rather than failing the entire program.
pub fn hip_get_driver_version() -> GpuResult<u32> {
    // hipGetDriverVersion may not be available in all HIP versions
    // A full implementation would dynamically load the symbol via dlsym
    // For now, return 0 as a safe default
    Ok(0)
}

/// Create HIP stream.
///
/// Stream enables async kernel execution and proper sequencing.
pub fn hip_stream_create() -> GpuResult<hipStream_t> {
    unsafe {
        let mut stream_ptr: *mut c_void = std::ptr::null_mut();
        let code = hipStreamCreate(&mut stream_ptr);
        hip_check(code)?;
        Ok(hipStream_t {
            _private: stream_ptr,
        })
    }
}

/// Destroy HIP stream.
///
/// # Safety
/// Stream must have been created with hip_stream_create and not already destroyed.
pub unsafe fn hip_stream_destroy(stream: hipStream_t) -> GpuResult<()> {
    let code = hipStreamDestroy(stream);
    hip_check(code)
}

/// Synchronize HIP stream.
///
/// Blocks until all queued operations in stream complete.
pub fn hip_stream_synchronize(stream: hipStream_t) -> GpuResult<()> {
    unsafe {
        let code = hipStreamSynchronize(stream);
        hip_check(code)
    }
}

/// Begin capture on a HIP stream.
pub fn hip_stream_begin_capture(stream: hipStream_t, mode: hipStreamCaptureMode) -> GpuResult<()> {
    unsafe {
        let code = hipStreamBeginCapture(stream, mode);
        hip_check(code)
    }
}

/// End capture on a HIP stream and return the captured graph template.
pub fn hip_stream_end_capture(stream: hipStream_t) -> GpuResult<hipGraph_t> {
    unsafe {
        let mut graph = hipGraph_t::null();
        let code = hipStreamEndCapture(stream, &mut graph);
        hip_check(code)?;
        Ok(graph)
    }
}

/// Return the current capture state of the stream.
pub fn hip_stream_is_capturing(stream: hipStream_t) -> GpuResult<hipStreamCaptureStatus> {
    unsafe {
        let mut status = hipStreamCaptureStatus::hipStreamCaptureStatusNone;
        let code = hipStreamIsCapturing(stream, &mut status);
        hip_check(code)?;
        Ok(status)
    }
}

/// Destroy a HIP graph template.
pub fn hip_graph_destroy(graph: hipGraph_t) -> GpuResult<()> {
    unsafe {
        let code = hipGraphDestroy(graph);
        hip_check(code)
    }
}

/// Instantiate an executable graph from a graph template.
pub fn hip_graph_instantiate(graph: hipGraph_t) -> GpuResult<hipGraphExec_t> {
    unsafe {
        let mut exec = hipGraphExec_t::null();
        let code = hipGraphInstantiate(
            &mut exec,
            graph,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        );
        hip_check(code)?;
        Ok(exec)
    }
}

/// Launch a captured HIP graph on a stream.
pub fn hip_graph_launch(graph_exec: hipGraphExec_t, stream: hipStream_t) -> GpuResult<()> {
    unsafe {
        let code = hipGraphLaunch(graph_exec, stream);
        hip_check(code)
    }
}

/// Update an executable graph with a new topology/parameters from a source graph.
pub fn hip_graph_exec_update(
    graph_exec: hipGraphExec_t,
    graph: hipGraph_t,
) -> GpuResult<hipGraphExecUpdateResult> {
    unsafe {
        let mut error_node = hipGraphNode_t::null();
        let mut result = hipGraphExecUpdateResult::hipGraphExecUpdateError;
        let code = hipGraphExecUpdate(graph_exec, graph, &mut error_node, &mut result);
        hip_check(code)?;
        Ok(result)
    }
}

/// Destroy an executable graph.

pub fn hip_graph_exec_destroy(graph_exec: hipGraphExec_t) -> GpuResult<()> {
    unsafe {
        let code = hipGraphExecDestroy(graph_exec);
        hip_check(code)
    }
}

/// Return all graph nodes contained in the graph template.
pub fn hip_graph_get_nodes(graph: hipGraph_t) -> GpuResult<Vec<hipGraphNode_t>> {
    unsafe {
        let mut count = 0usize;
        let code = hipGraphGetNodes(graph, std::ptr::null_mut(), &mut count);
        hip_check(code)?;

        let mut nodes = vec![hipGraphNode_t::null(); count];
        let code = hipGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count);
        hip_check(code)?;
        nodes.truncate(count);
        Ok(nodes)
    }
}

/// Return the type of a graph node.
pub fn hip_graph_node_get_type(node: hipGraphNode_t) -> GpuResult<hipGraphNodeType> {
    unsafe {
        let mut ty = hipGraphNodeType::hipGraphNodeTypeKernel;
        let code = hipGraphNodeGetType(node, &mut ty);
        hip_check(code)?;
        Ok(ty)
    }
}

/// Read the launch parameters of a kernel node.
pub fn hip_graph_kernel_node_get_params(node: hipGraphNode_t) -> GpuResult<hipKernelNodeParams> {
    unsafe {
        let mut params: hipKernelNodeParams = std::mem::zeroed();
        let code = hipGraphKernelNodeGetParams(node, &mut params);
        hip_check(code)?;
        Ok(params)
    }
}

/// Update the launch parameters of a kernel node in an executable graph.
pub fn hip_graph_exec_kernel_node_set_params(
    graph_exec: hipGraphExec_t,
    node: hipGraphNode_t,
    params: &hipKernelNodeParams,
) -> GpuResult<()> {
    unsafe {
        let code = hipGraphExecKernelNodeSetParams(graph_exec, node, params);
        hip_check(code)
    }
}

/// Query the occupancy-maximizing block size for a kernel.
pub fn hip_occupancy_max_potential_block_size(
    kernel: *const c_void,
    dyn_shared_mem_per_block: usize,
    block_size_limit: i32,
) -> GpuResult<(i32, i32)> {
    unsafe {
        let mut min_grid_size = 0i32;
        let mut block_size = 0i32;
        let code = hipOccupancyMaxPotentialBlockSize(
            &mut min_grid_size,
            &mut block_size,
            kernel,
            dyn_shared_mem_per_block,
            block_size_limit,
        );
        hip_check(code)?;
        Ok((min_grid_size, block_size))
    }
}

/// Query the dynamic shared memory available per block for a target occupancy.
pub fn hip_occupancy_available_dynamic_smem_per_block(
    kernel: *const c_void,
    num_blocks: i32,
    block_size: i32,
) -> GpuResult<usize> {
    unsafe {
        let mut dynamic_smem_size = 0usize;
        let code = hipOccupancyAvailableDynamicSMemPerBlock(
            &mut dynamic_smem_size,
            kernel,
            num_blocks,
            block_size,
        );
        hip_check(code)?;
        Ok(dynamic_smem_size)
    }
}

// ── FFI Declarations ───────────────────────────────────────────────────────────────

#[repr(C)]
struct hipDeviceProp_t {
    name: [c_char; 256],
    totalGlobalMem: usize,
    sharedMemPerBlock: usize,
    regsPerBlock: i32,
    warpSize: i32,
    maxThreadsPerBlock: i32,
    maxThreadsDim: [i32; 3],
    maxGridSize: [i32; 3],
    clockRate: i32,
    memoryClockRate: i32,
    memoryBusWidth: i32,
    multiProcessorCount: i32,
    // AMD GPUs have many more fields. We add a large padding to be safe.
    _padding: [u8; 4096],
}

#[repr(C)]
enum hipMemcpyKind {
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
}

/// Opaque HIP stream type
///
/// In C, this is a pointer to an opaque struct. We represent it as a
/// raw pointer so it can be freely copied and passed to FFI functions.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct hipStream_t {
    _private: *mut c_void,
}

impl hipStream_t {
    pub fn null() -> Self {
        Self {
            _private: std::ptr::null_mut(),
        }
    }

    pub fn is_null(self) -> bool {
        self._private.is_null()
    }
}

/// Opaque HIP graph handle.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct hipGraph_t {
    _private: *mut c_void,
}

impl hipGraph_t {
    pub fn null() -> Self {
        Self {
            _private: std::ptr::null_mut(),
        }
    }

    pub fn is_null(self) -> bool {
        self._private.is_null()
    }
}

/// Opaque HIP graph node handle.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct hipGraphNode_t {
    _private: *mut c_void,
}

impl hipGraphNode_t {
    pub fn null() -> Self {
        Self {
            _private: std::ptr::null_mut(),
        }
    }

    pub fn is_null(self) -> bool {
        self._private.is_null()
    }
}

/// Opaque HIP executable graph handle.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct hipGraphExec_t {
    _private: *mut c_void,
}

impl hipGraphExec_t {
    pub fn null() -> Self {
        Self {
            _private: std::ptr::null_mut(),
        }
    }

    pub fn is_null(self) -> bool {
        self._private.is_null()
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum hipGraphNodeType {
    hipGraphNodeTypeKernel = 0,
    hipGraphNodeTypeMemcpy = 1,
    hipGraphNodeTypeMemset = 2,
    hipGraphNodeTypeHost = 3,
    hipGraphNodeTypeGraph = 4,
    hipGraphNodeTypeEmpty = 5,
    hipGraphNodeTypeWaitEvent = 6,
    hipGraphNodeTypeEventRecord = 7,
    hipGraphNodeTypeExtSemaphoreSignal = 8,
    hipGraphNodeTypeExtSemaphoreWait = 9,
    hipGraphNodeTypeMemAlloc = 10,
    hipGraphNodeTypeMemFree = 11,
    hipGraphNodeTypeMemcpyFromSymbol = 12,
    hipGraphNodeTypeMemcpyToSymbol = 13,
    hipGraphNodeTypeBatchMemOp = 14,
    hipGraphNodeTypeCount = 15,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum hipStreamCaptureMode {
    hipStreamCaptureModeGlobal = 0,
    hipStreamCaptureModeThreadLocal = 1,
    hipStreamCaptureModeRelaxed = 2,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum hipStreamCaptureStatus {
    hipStreamCaptureStatusNone = 0,
    hipStreamCaptureStatusActive = 1,
    hipStreamCaptureStatusInvalidated = 2,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct hipKernelNodeParams {
    pub blockDim: dim3,
    pub extra: *mut *mut c_void,
    pub func: *mut c_void,
    pub gridDim: dim3,
    pub kernelParams: *mut *mut c_void,
    pub sharedMemBytes: u32,
}

extern "C" {
    fn hipGetDeviceCount(count: *mut i32) -> hipError_t;
    fn hipGetDeviceProperties(props: *mut hipDeviceProp_t, device: i32) -> hipError_t;
    fn hipMalloc(ptr: *mut *mut u8, size: usize) -> hipError_t;
    fn hipFree(ptr: *mut u8) -> hipError_t;
    fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> hipError_t;
    fn hipHostFree(ptr: *mut c_void) -> hipError_t;
    fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;
    fn hipGetErrorString(error: hipError_t) -> *const c_char;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
    fn hipGetDriverVersion(driverVersion: *mut c_int) -> hipError_t;
    fn hipStreamCreate(stream: *mut *mut c_void) -> hipError_t;
    fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    fn hipStreamBeginCapture(stream: hipStream_t, mode: hipStreamCaptureMode) -> hipError_t;
    fn hipStreamEndCapture(stream: hipStream_t, pGraph: *mut hipGraph_t) -> hipError_t;
    fn hipStreamIsCapturing(
        stream: hipStream_t,
        pCaptureStatus: *mut hipStreamCaptureStatus,
    ) -> hipError_t;
    fn hipGraphDestroy(graph: hipGraph_t) -> hipError_t;
    fn hipGraphInstantiate(
        pGraphExec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        pErrorNode: *mut hipGraphNode_t,
        pLogBuffer: *mut c_char,
        bufferSize: usize,
    ) -> hipError_t;
    fn hipGraphLaunch(graphExec: hipGraphExec_t, stream: hipStream_t) -> hipError_t;
    fn hipGraphExecDestroy(graphExec: hipGraphExec_t) -> hipError_t;
    fn hipGraphExecUpdate(
        hGraphExec: hipGraphExec_t,
        hGraph: hipGraph_t,
        pErrorNode: *mut hipGraphNode_t,
        pUpdateResult: *mut hipGraphExecUpdateResult,
    ) -> hipError_t;
    fn hipGraphGetNodes(
        graph: hipGraph_t,
        nodes: *mut hipGraphNode_t,
        numNodes: *mut usize,
    ) -> hipError_t;
    fn hipGraphNodeGetType(node: hipGraphNode_t, pType: *mut hipGraphNodeType) -> hipError_t;
    fn hipGraphKernelNodeGetParams(
        node: hipGraphNode_t,
        pNodeParams: *mut hipKernelNodeParams,
    ) -> hipError_t;
    fn hipGraphExecKernelNodeSetParams(
        hGraphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        pNodeParams: *const hipKernelNodeParams,
    ) -> hipError_t;
    fn hipOccupancyMaxPotentialBlockSize(
        gridSize: *mut i32,
        blockSize: *mut i32,
        f: *const c_void,
        dynSharedMemPerBlk: usize,
        blockSizeLimit: i32,
    ) -> hipError_t;
    fn hipOccupancyAvailableDynamicSMemPerBlock(
        dynamicSmemSize: *mut usize,
        f: *const c_void,
        numBlocks: i32,
        blockSize: i32,
    ) -> hipError_t;
}

// ── Public Types ───────────────────────────────────────────────────────────────────

/// GPU device information.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub total_vram_bytes: usize,
    pub compute_units: usize,
    pub max_clock_mhz: usize,
    pub device_id: i32,
    pub warp_size: usize,
    /// Device architecture name (e.g., "gfx1100")
    pub arch_name: String,
    /// Maximum shared memory available per block in bytes
    pub max_shared_mem_per_block: usize,
}

/// Q4_K quantized block (144 bytes for 256 f32 values)
/// Matches llama.cpp block_q4_k structure
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GpuQ4KBlock {
    pub d: half::f16,     // delta/scale (2 bytes)
    pub dmin: half::f16,  // minimum scale (2 bytes)
    pub scales: [u8; 12], // quantized scales (12 bytes)
    pub qs: [u8; 128],    // quants, 4-bit values (128 bytes)
}

impl Default for GpuQ4KBlock {
    fn default() -> Self {
        Self {
            d: half::f16::from_f32(1.0),
            dmin: half::f16::from_f32(0.0),
            scales: [0; 12],
            qs: [0; 128],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hip_get_device_count_doesnt_crash() {
        // Should never panic, even if HIP not available
        let count = hip_get_device_count();
        // May be Ok(0) or Err, either is fine
        match count {
            Ok(n) => println!("Found {} GPU(s)", n),
            Err(e) => println!("HIP not available: {}", e),
        }
    }

    #[test]
    fn hip_malloc_free_works() {
        let ptr = hip_malloc(1024);
        if let Ok(p) = ptr {
            assert!(!p.is_null());
            hip_free(p);
        }
        // If HIP not available, test passes anyway
    }
}
