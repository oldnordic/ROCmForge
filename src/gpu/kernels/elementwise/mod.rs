mod arithmetic;
mod quant;
mod sampling;
mod svd;
mod state;

pub use arithmetic::*;
pub use quant::*;
pub use sampling::*;
pub use svd::*;
pub use state::*;

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::ffi::hipStream_t;

    #[test]
    fn add_rejects_zero_n() {
        let result = add(std::ptr::null(), std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn mul_on_stream_rejects_zero_n() {
        let result = mul_on_stream(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn gelu_rejects_zero_n() {
        let result = gelu(std::ptr::null(), std::ptr::null_mut(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn silu_on_stream_rejects_zero_n() {
        let result = silu_on_stream(
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            hipStream_t::null(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn add_batched_rejects_zero_seq_len() {
        let result = add_batched(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            128,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn argmax_rejects_zero_n() {
        let result = argmax_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn embed_q8_0_rejects_unaligned_hidden_size() {
        let result = embed_q8_0_token(std::ptr::null(), std::ptr::null_mut(), 33, 10, 0);
        assert!(result.is_err());
    }

    #[test]
    fn embed_q8_0_batch_rejects_zero_seq_len() {
        let result = embed_q8_0_batch(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            32,
            10,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn zero_fill_rejects_zero_n() {
        use super::super::super::GpuDevice;

        let device = GpuDevice::init(0);
        let result = match device {
            Ok(d) => zero_fill(std::ptr::null_mut(), 0, &d),
            Err(_) => return, // Skip test if GPU unavailable
        };
        assert!(result.is_err());
    }
}
