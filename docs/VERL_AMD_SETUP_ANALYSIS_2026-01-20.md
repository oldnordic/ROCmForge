# Verl AMD/ROCm Setup Analysis

**Date**: 2026-01-20
**Source**: Verl Documentation - "Getting started with AMD (ROCM Kernel)"
**URL**: https://verl.readthedocs.io/en/latest/amd_tutorial/amd_build_dockerfile_page.html
**Author**: Yusheng Su
**Last Updated**: 07/06/2025
**Purpose**: Understanding production AMD GPU setup for LLM training/inference

---

## Executive Summary

Verl is an RLHF training framework that supports AMD GPUs (MI300) via ROCm. This document analyzes their Docker-based setup, environment variables, and integration points with the AMD ecosystem.

**Target Hardware**: AMD Instinct MI300 (CDNA architecture)
**Key Components**: ROCm 6.3.4, vLLM, SGLang, Ray, TransformerEngine

---

## Part 1: Docker Base Image

### Base Image Used

```dockerfile
FROM "rlfoundation.azurecr.io/rocm6.3.4:vllm-0.8.5-numa-patch-ubuntu-22.04"
```

**Components**:
- Ubuntu 22.04
- ROCm 6.3.4
- vLLM 0.8.5 (with NUMA patch)
- Python 3.12

---

## Part 2: Critical Environment Variables

### Ray/ROCm Visibility Settings

```bash
# For ray >= 2.45.0
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# For ray < 2.45.0
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
```

**Why This Matters**: Ray's device management conflicts with ROCm's device enumeration. These flags prevent Ray from modifying `HIP_VISIBLE_DEVICES`.

### HIP/ROCm Compilation Flags

```bash
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx942"
export AMDGPU_TARGETS=gfx942
export ROCM_ARCH=gfx942
export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
```

**Architecture Codes**:
- `gfx942`: MI300X (CDNA3)
- `gfx90a`: MI200 (CDNA2)
- `gfx1100`: RDNA3 (Radeon 7900 XTX - consumer)

### SGLang-Specific Variables

```bash
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export NCCL_MIN_NCHANNELS=112
export MOE_PADDING=1
export VLLM_FP8_PADDING=1
export VLLM_FP8_ACT_PADDING=1
export VLLM_FP8_WEIGHT_PADDING=1
export VLLM_FP8_REDUCE_CONV=1
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1
```

---

## Part 3: Key Components Built

### 1. TransformerEngine (ROCm Version)

```dockerfile
WORKDIR /workspace/
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git
WORKDIR /workspace/TransformerEngine
git checkout 236178e5

ENV NVTE_FRAMEWORK=pytorch
ENV NVTE_ROCM_ARCH=gfx942
ENV NVTE_USE_HIPBLASLT=1
ENV NVTE_USE_ROCM=1
ENV CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/hip:/usr/local:/usr"

RUN MAX_JOBS=512 pip install . -vvv
```

**Purpose**: AMD's optimized Transformer blocks for training

### 2. vLLM (ROCm Version)

```dockerfile
RUN git clone https://github.com/ROCm/vllm.git
WORKDIR /workspace/vllm
RUN git checkout 113274a0

ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"
ENV MAX_JOBS=512

RUN pip install "boto3>=1.26.0"
RUN pip install setuptools_scm
RUN python3 setup.py install
```

**Note**: Uses ROCm fork, not upstream vLLM

### 3. SGLang (ROCm Fork)

```dockerfile
ENV SGL_REPO=https://github.com/sgl-project/sglang
ENV SGL_BRANCH=v0.4.6.post5

RUN git clone ${SGL_REPO}
WORKDIR sglang/sgl-kernel
RUN rm -f pyproject.toml
RUN mv pyproject_rocm.toml pyproject.toml
RUN python setup_rocm.py install

WORKDIR ..
RUN python -m pip --no-cache-dir install -e "python[all_hip]"
```

**Key**: Uses `setup_rocm.py` instead of standard setup

### 4. Triton (ROCm Fork)

```dockerfile
ENV TRITON_REPO=https://github.com/ROCm/triton.git
ENV TRITON_COMMIT=improve_fa_decode_3.0.0

RUN git clone ${TRITON_REPO}
WORKDIR triton/python
RUN python3 setup.py install
```

### 5. Aiter

```dockerfile
ENV AITER_REPO=https://github.com/ROCm/aiter.git
ENV AITER_COMMIT=v0.1.2

RUN git clone ${AITER_REPO}
WORKDIR aiter
RUN git checkout ${AITER_COMMIT}
RUN git submodule sync
RUN git submodule update --init --recursive
RUN PREBUILD_KERNELS=1 GPU_ARCHS=gfx942 python3 setup.py install
```

**Key**: `PREBUILD_KERNELS=1` - pre-compiles kernels for target architecture

### 6. vLLM v0.8.5 (Patched)

```dockerfile
RUN git clone https://github.com/RLFoundation/vllm-patch.git
WORKDIR vllm-patch
RUN git checkout v0.8.5-sleep-numa

ENV VLLM_TARGET_DEVICE=rocm
ENV ROCM_PATH=/opt/rocm
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.8.5.dev
ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"

RUN ln -sf /opt/rocm/lib/libamdhip64.so /usr/lib/libamdhip64.so
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.8.5.dev \
    PYTORCH_ROCM_ARCH="gfx90a;gfx942" \
    MAX_JOBS=${MAX_JOBS} \
    python3 setup.py install
```

---

## Part 4: Configuration File Copying

### MI300X Config Replication

```dockerfile
RUN find /sgl-workspace/sglang/python/sglang/srt/layers/quantization/configs/ \
        /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/ \
        -type f -name '*MI300X*' | \
        xargs -I {} sh -c 'vf_config=$(echo "$1" | sed "s/MI300X/MI300X_VF/"); cp "$1" "$vf_config"' -- {}
```

**Pattern**: Copy MI300X configs to `MI300X_VF` variant

---

## Part 5: Memory Optimization

### torch_memory_saver_numa

```dockerfile
ENV HIPCC_COMPILE_FLAGS_APPEND="--amdgpu-target=gfx90a;gfx942 -D__HIP_PLATFORM_AMD__"
ENV CFLAGS="-D__HIP_PLATFORM_AMD__"
ENV CXXFLAGS="-D__HIP_PLATFORM_AMD__"
RUN pip install "git+https://github.com/YangWang92/torch_memory_saver_numa.git@numa"
```

**Purpose**: NUMA-aware memory management for multi-GPU systems

---

## Part 6: Container Launch Configuration

### Docker Run Command

```bash
docker run --rm -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8265:8265 \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  --shm-size 128G \
  -w $PWD \
  verl-rocm \
  /bin/bash
```

**Key Flags**:
- `--device /dev/dri` - Direct rendering access
- `--device /dev/kfd` - ROCm kernel driver access
- `--shm-size 128G` - Large shared memory for multi-GPU
- `--privileged` - Required for GPU access in some setups
- `--cap-add SYS_PTRACE` - For debugging/profiling

### Non-root Option (Optional)

```bash
-e HOST_UID=$(id -u)
-e HOST_GID=$(id -g)
```

---

## Part 7: Multi-Node Slurm Configuration

### SLURM Job Setup

```bash
#SBATCH --job-name=verl-ray-on-slurm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=200G
#SBATCH --time=30-00:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=28
#SBATCH --output=../verl_log/slurm-%j.out
#SBATCH --error=../verl_log/slurm-%j.err
#SBATCH --nodelist=gpu-[0,1]
```

### NCCL/RCCL Configuration

```bash
export NCCL_DEBUG=TRACE
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1
```

**Note**: Uses InfiniBand (`mlx5_*`) for multi-node communication

---

## Part 8: Ray Cluster Initialization

### Head Node Setup

```bash
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
ip_head=$head_node_ip:$port

srun --nodes=1 --ntasks=1 -w "$head_node" \
    docker exec "${CONTAINER_NAME}" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --dashboard-port=8266 \
        --num-cpus "${SLURM_CPUS_PER_TASK}" \
        --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
```

### Worker Nodes

```bash
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        docker exec "${CONTAINER_NAME}" \
            ray start --address "$ip_head" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" \
            --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
done
```

---

## Part 9: Training Configuration Examples

### PPO Training

```bash
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
ENGINE=vllm  # or sglang

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.optim.lr=1e-5 \
    critic.model.path=$MODEL_PATH \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.total_epochs=15
```

### GRPO Training

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=1024 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True
```

---

## Part 10: ROCm Ecosystem Repositories

### Key AMD/ROCm Forks

| Component | ROCm Fork | Purpose |
|-----------|-----------|---------|
| vLLM | github.com/ROCm/vllm | Inference engine |
| Triton | github.com/ROCm/triton | Kernel compiler |
| TransformerEngine | github.com/ROCm/TransformerEngine | Training optimizations |
| Apex | github.com/ROCm/apex | Training utilities |

### Community Patches

| Repo | Branch/Commit | Purpose |
|------|---------------|---------|
| vllm-patch | v0.8.5-sleep-numa | NUMA fixes |
| Megatron-LM | amd_version | AMD training |

---

## Part 11: Architecture Support Matrix

### GPU Architecture Support

| Arch Code | GPU | Verl Support | ROCmForge Relevance |
|-----------|-----|--------------|---------------------|
| gfx942 | MI300X | ✓ Primary | Datacenter target |
| gfx90a | MI200 | ✓ Supported | Legacy datacenter |
| gfx1100 | RX 7900 XTX | Not tested | **Our hardware** |

### Key Insight for ROCmForge

Verl targets **MI300 (CDNA)** architecture. ROCmForge targets **RDNA3** (consumer Radeon).

**Differences**:
- CDNA has MFMA (matrix cores) for FP4/FP6
- RDNA3 relies on VALU optimization
- Different tuning parameters needed

---

## Part 12: Build System Analysis

### Python Build Tools

```python
# Standard pattern for ROCm components
ENV NVTE_ROCM_ARCH=gfx942
ENV NVTE_USE_ROCM=1
ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"

# Setup with ROCm-specific script
python setup_rocm.py install  # SGLang pattern
python3 setup.py install        # Standard pattern
```

### Kernel Pre-compilation

```bash
# Aiter pattern
PREBUILD_KERNELS=1 GPU_ARCHS=gfx942 python3 setup.py install
```

**Purpose**: Avoid JIT compilation during training

---

## Part 13: Lessons for ROCmForge

### 1. Architecture Detection

```rust
// Need proper gfx1100 support for RDNA3
const ROCM_ARCH: &str = std::env::var("ROCM_ARCH")
    .unwrap_or_else(|_| "gfx1100".to_string());
```

### 2. Visibility Management

```rust
// Ray compatibility consideration
let _hip_visible_devices = std::env::var("HIP_VISIBLE_DEVICES")
    .or(std::env::var("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"));
```

### 3. Memory Environment

```rust
// Critical for RDNA3
env::set_var("HSA_NO_SCRATCH_RECLAIM", "1");
env::set_var("HIP_FORCE_DEV_KERNARG", "1");
```

### 4. NUMA Considerations

Even single-GPU systems benefit from:
- Correct memory allocation
- CPU affinity
- Proper NUMA policy

### 5. Configuration Copy Pattern

```rust
// Like MI300X_VF configs
fn replicate_configs(base: &Path, target: &str) -> Result<()> {
    for entry in walkdir(base)? {
        if entry.path_name().contains("MI300X") {
            let new_name = entry.path_name().replace("MI300X", target);
            // ...
        }
    }
}
```

---

## Part 14: Dependency Version Analysis

### Key Version Constraints

| Component | Version | Notes |
|-----------|---------|--------|
| ROCm | 6.3.4 | Base image |
| Python | 3.12 | Forward-looking |
| vLLM | 0.8.5 | Patched version |
| SGLang | v0.4.6.post5 | ROCm fork |
| Triton | ROCm fork | Special commit |
| Ray | >=2.47.0 | Core requirement |

### Build Parallelism

```dockerfile
ENV MAX_JOBS=512
```

**Purpose**: Leverage 512-core systems for compilation

---

## Part 15: Integration Checklist

### For AMD GPU Support

- [ ] ROCm development environment (`/opt/rocm`)
- [ ] HIP compiler (hipcc)
- [ ] Architecture-specific flags (`gfx1100` for RDNA3)
- [ ] Ray visibility variables set
- [ ] NUMA awareness (even for single GPU)
- [ ] Memory optimization (HSA_NO_SCRATCH_RECLAIM)
- [ ] Proper device access (`/dev/kfd`, `/dev/dri`)

---

## Conclusion

Verl's AMD setup demonstrates:
1. **Docker-based isolation** for ROCm environments
2. **Multiple ROCm forks** of key projects (vLLM, Triton)
3. **Architecture-specific tuning** (gfx942 for MI300)
4. **Complex multi-node setup** with Slurm/Ray

**Key Takeaway for ROCmForge**:
- Target `gfx1100` architecture (RDNA3), not `gfx942`
- Use ROCm forks where available
- Implement memory pooling patterns
- Support environment-driven configuration

---

**End of Report** - Investigation only, no code changes.
