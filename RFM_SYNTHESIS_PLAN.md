# RFM Synthesis Plan: The Ultimate RDNA3 LLM Engine

This document defines the production-grade, zero-mock engineering roadmap to merge the mathematical precision of **SVD-Quant outlier corrections** with the raw execution speed of **mixed quantization (MQ)**, **RDNA Wave32 warp-shuffle GEMV kernels**, and **fused `hipGraph` decode capture**.

---

## 1. Architectural Overview & Mathematical Crossover

The standard `.rfm` format currently supports either standard `Q4_0` execution or SVD low-rank corrections in a synchronous layer-by-layer fallback loop (`DecodeStyle`). This plan establishes a highly optimized, asynchronous execution model specifically designed to saturate the **800 GB/s memory bandwidth** of RDNA3 hardware (Radeon RX 7900 XT).

### The Mathematical Synthesis
A standard transformer model quantized uniformly to 4-bit (`Q4_0`) experiences quality degradation (perplexity loss) due to high-magnitude outlier activations. Keep the FFN layers (bulk of parameters) in `Q4_0`, while isolating the attention projections using SVD:
$$W_{\text{attn}} = W_{\text{res}} + U \cdot V^T$$
Because the high-magnitude outliers are isolated in the low-rank float matrices ($U \cdot V^T$), the base residual matrix ($W_{\text{res}}$) is extremely easy to quantize down to `Q4_0` or `Q3_0` with **zero accuracy degradation**.

### The Execution Synthesis
To bypass the host-side launch overhead of the SVD addition, we capture both the quantized residual GEMV and the dual low-rank projection GEMVs into a static **`hipGraph`** during initialization, executing them as a single fused pipeline per token step with zero CPU synchronizations.

---

## 2. Phase 1: Mixed Quantization (MQ) in the Converter & Loader

We will extend `rocmforge-convert` and the RFM loader to support hybrid quantized weight packing, matching attention projections to higher precision formats and FFN layers to low-bitwidth residuals.

### Step 1.1: Extend `rocmforge-convert` to Support MQ4/MQ6
Add command-line flags to `--bin rocmforge-convert` to allow role-based quantization packing:
* **Attention Projections (Q, K, V, O):** Quantized to `RfmType::Q8_0` or `RfmType::Q4SvdQuant` (residual Q4 + rank-k outliers).
* **FFN Projections (Gate, Up, Down):** Quantized to `RfmType::Q4Split` or `RfmType::Mpo` (highly compressed).
* **Location:** `src/bin/convert.rs`

### Step 1.2: Support Mixed Layout Upload in `GpuModelWeights`
Modify `GpuModelWeights::load_rfm_for_device` to upload roles correctly:
```rust
// Location: src/gpu/weights/model.rs
// Read tensor role and type dynamically, dispatching appropriate dequantization/unpacking kernels:
let meta = WeightMeta {
    wtype: match view.wtype {
        RfmType::Q4SvdQuant { k } => GgmlType::Q4_0,
        RfmType::Q8_0 => GgmlType::Q8_0,
        _ => rfm_type_to_ggml(&view.wtype),
    },
    dims: view.dims.to_vec(),
    needs_transpose: false,
    role: detect_tensor_role(&name),
    svd_k: extract_svd_k(&view.wtype),
};
```

---

## 3. Phase 2: RDNA Wave32 Warp-Shuffle GEMV Kernels

We will bypass shared memory and block synchronization barriers inside the matrix-vector kernels by optimizing them specifically for **RDNA's 32-thread Wavefront size**.

### Step 2.1: Hand-Craft 32-Thread Wavefront Reductions
In `hip_kernels/elementwise.hip` and `hip_kernels/matmul.hip`, implement warp shuffle intrinsic reductions:
```cpp
// Location: hip_kernels/matmul.hip
// Sum values across 32 threads in a wave without __syncthreads()
__device__ inline float wave_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset, 32);
    }
    return val;
}
```

### Step 2.2: Implement Adaptive Multi-Row Bucketing Dispatch
Modify `dispatch_gemv_impl` inside `src/gpu/ops/gemv.rs` to dynamically route execution:
1. **Narrow Matrices ($K \le 1536$):** Dispatch multi-row GEMV (using 64 threads, 2 warps per block, each warp working independently on a different row).
2. **Wide Matrices ($K \ge 2048$):** Dispatch single-warp Wave32 kernel (maximum compute occupancy of 20 wavefronts per Compute Unit).

---

## 4. Phase 3: Fused `hipGraph` Decode DAG Capture

We will eliminate all host-device layer synchronization latency by re-architecting our graph recorder to support SVD and MQ execution paths.

### Step 3.1: Support SVD Corrections in Graph Capture
Modify the graph capture flow in `src/gpu/graph.rs` to support recording both the base quantized gemv node and the SVD correction nodes:
```rust
// Location: src/gpu/graph.rs
// Capture standard Q4_0 / Q8_0 base GEMV:
let base_node = record_gemv_node(graph, base_weights, input, output)?;
// If SVD is present, add the low-rank corrections as dependent nodes:
if let Some(svd) = layer.attn_q_svd.as_ref() {
    let temp_node = record_gemv_node(graph, &svd.v, input, temp_vector)?;
    let corr_node = record_gemv_node_accumulate(graph, &svd.u, temp_vector, output)?;
    graph.add_dependency(base_node, temp_node)?;
    graph.add_dependency(temp_node, corr_node)?;
}
```

### Step 3.2: Eliminate Inter-Layer Synchronizations
Remove the blocking `device.synchronize()` calls from `gpu_full_forward_hybrid` during graph execution. The GPU hardware scheduler will coordinate dependencies between kernel nodes automatically inside the single graph execution queue.

---

## 5. Phase 4: GFX11 WMMA Prefill & Q8 Asymmetric KV Cache

We will optimize the attention prefill pass using hardware tensor cores and compress the key-value cache to minimize memory bus pressure.

### Step 4.1: Enable GFX11 WMMA Prefill Kernels
Integrate RDNA3 WMMA hardware assembly instructions (`v_coalesce_b32` / wave matrix dot products) inside the prefill matrix-vector math in `hip_kernels/attention.hip` for batched multi-token prefill.

### Step 4.2: Dynamic Q8 Asymmetric KV Cache
Implement dynamic on-the-fly quantization of K/V states to 8-bit asymmetric scales (`asym3`/`asym4`) inside the KV rope-write kernels. This cuts attention memory bandwidth in half during decode.

---

## 6. Verification & Staged Integration Gates

To maintain strict, production-grade correctness, every step must pass our verification gates:

### Gate 1: Mathematical Equivalence (CPU Reference)
* Compare every newly optimized GPU path against the sequential CPU reference outputs.
* **Pass criteria:** Maximum absolute error (L-infinity norm) between GPU and CPU logits must be **$\le 10^{-5}$** across the entire 17-token prompt sequence.

### Gate 2: VRAM Safety Pre-flight Verification
* Execute all validation passes under the exclusive process lock and execution timeout runner:
  `./scripts/gpu_safe_run.sh --timeout 30 --max-tokens 10 ./target/release/rocmforge --gpu ...`
* Confirm the pre-flight safety gate prints a correct, detailed memory budget report and cleanly aborts if simulated free VRAM is insufficient.

### Gate 3: Throughput Sweep Benchmarks
* Run steady-state decoding throughput sweeps using Criterion:
  `cargo bench --bench gpu_decode --features gpu -- --noplot`
* Confirm decode throughput shows a measurable step-by-step increase towards the memory bandwidth ceiling.
