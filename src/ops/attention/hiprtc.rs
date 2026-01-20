//! HIP Runtime Compilation (HIPRTC) utilities

use crate::backend::{HipError, HipResult};
use std::ffi::{c_char, c_void, CString};
use std::ptr;

type HiprtcProgram = *mut c_void;
const HIPRTC_SUCCESS: i32 = 0;

#[link(name = "hiprtc")]
extern "C" {
    fn hiprtcCreateProgram(
        prog: *mut HiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: i32,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> i32;
    fn hiprtcCompileProgram(
        prog: HiprtcProgram,
        num_options: i32,
        options: *const *const c_char,
    ) -> i32;
    fn hiprtcGetProgramLogSize(prog: HiprtcProgram, log_size_ret: *mut usize) -> i32;
    fn hiprtcGetProgramLog(prog: HiprtcProgram, log: *mut c_char) -> i32;
    fn hiprtcGetCodeSize(prog: HiprtcProgram, code_size_ret: *mut usize) -> i32;
    fn hiprtcGetCode(prog: HiprtcProgram, code: *mut c_char) -> i32;
    fn hiprtcDestroyProgram(prog: *mut HiprtcProgram) -> i32;
}

pub fn compile_kernel(name: &str, source: &str) -> HipResult<Vec<u8>> {
    let name_c = CString::new(name)
        .map_err(|e| HipError::KernelLoadFailed(format!("Invalid kernel name: {}", e)))?;
    let source_c = CString::new(source)
        .map_err(|e| HipError::KernelLoadFailed(format!("Invalid kernel source: {}", e)))?;

    let mut program: HiprtcProgram = ptr::null_mut();
    let create_result = unsafe {
        hiprtcCreateProgram(
            &mut program,
            source_c.as_ptr(),
            name_c.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        )
    };

    if create_result != HIPRTC_SUCCESS {
        return Err(HipError::KernelLoadFailed(
            "hiprtcCreateProgram failed".to_string(),
        ));
    }

    let option = CString::new("--std=c++17").unwrap();
    let options = [option.as_ptr()];
    let compile_result =
        unsafe { hiprtcCompileProgram(program, options.len() as i32, options.as_ptr()) };

    if compile_result != HIPRTC_SUCCESS {
        let log = get_program_log(program);
        unsafe { hiprtcDestroyProgram(&mut program) };
        return Err(HipError::KernelLoadFailed(format!(
            "hiprtcCompileProgram failed: {}",
            log.unwrap_or_else(|| "unknown error".to_string())
        )));
    }

    let mut code_size: usize = 0;
    let size_result = unsafe { hiprtcGetCodeSize(program, &mut code_size) };
    if size_result != HIPRTC_SUCCESS {
        unsafe { hiprtcDestroyProgram(&mut program) };
        return Err(HipError::KernelLoadFailed(
            "hiprtcGetCodeSize failed".to_string(),
        ));
    }

    let mut code = vec![0u8; code_size];
    let code_result = unsafe { hiprtcGetCode(program, code.as_mut_ptr() as *mut c_char) };
    unsafe { hiprtcDestroyProgram(&mut program) };

    if code_result != HIPRTC_SUCCESS {
        return Err(HipError::KernelLoadFailed(
            "hiprtcGetCode failed".to_string(),
        ));
    }

    Ok(code)
}

fn get_program_log(program: HiprtcProgram) -> Option<String> {
    let mut size: usize = 0;
    if unsafe { hiprtcGetProgramLogSize(program, &mut size) } != HIPRTC_SUCCESS || size == 0 {
        return None;
    }

    let mut buffer = vec![0u8; size];
    if unsafe { hiprtcGetProgramLog(program, buffer.as_mut_ptr() as *mut c_char) }
        != HIPRTC_SUCCESS
    {
        return None;
    }

    Some(String::from_utf8_lossy(&buffer).into_owned())
}
