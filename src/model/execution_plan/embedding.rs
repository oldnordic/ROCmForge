//! Embedding and LM head operations
//!
//! Handles token embedding lookup, LM head application, and related helper functions.

use crate::backend::{DeviceTensor, HipBackend, HipError, HipResult};
use crate::ggml::{Layout, GgmlBackend};
use crate::loader::lazy_tensor::LazyTensor;
use crate::loader::TensorShape;
use crate::model::config::ModelConfig;
use crate::model::execution_plan::Architecture;
use std::collections::HashMap;
use std::sync::Arc;

use super::types::ExecutionPlan;

/// Map embedding weights to LazyTensor handle
///
/// This function implements llama.cpp-compatible embedding detection:
/// - vocab_size == 0 means "unknown", not "invalid"
/// - Accepts both [vocab_size, hidden] and [hidden, vocab_size] layouts
/// - Infers vocab_size from tensor shape when metadata is missing
pub fn map_embedding_lazy(
    lazy_tensors: &HashMap<String, LazyTensor>,
    config: &ModelConfig,
    _architecture: &Architecture,
) -> HipResult<(Arc<LazyTensor>, Layout)> {
    let embedding_names = [
        "token_embd.weight",
        "embed_tokens.weight",
        "word_embeddings.weight",
    ];

    eprintln!(">>> map_embedding_lazy: config.hidden_size={}, config.vocab_size={}",
        config.hidden_size, config.vocab_size);

    // Try to find and validate embedding tensor
    for name in &embedding_names {
        if let Some(lazy) = lazy_tensors.get(*name) {
            if let Some(shape) = lazy.shape() {
                eprintln!(">>> map_embedding_lazy: Found tensor '{}' with shape {:?}", name, shape);
                if shape.len() != 2 {
                    eprintln!(">>> map_embedding_lazy: Skipping (not 2D)");
                    continue;
                }

                let (d0, d1) = (shape[0], shape[1]);
                let hidden_size = config.hidden_size;
                eprintln!(">>> map_embedding_lazy: d0={}, d1={}, hidden_size={}", d0, d1, hidden_size);

                // Determine vocab_size if unknown (== 0)
                let actual_vocab_size = if config.vocab_size == 0 {
                    // Infer from shape: whichever dimension ISN'T hidden_size is vocab_size
                    if d0 == hidden_size && d1 != hidden_size {
                        d1 // [hidden, vocab] layout
                    } else if d1 == hidden_size && d0 != hidden_size {
                        d0 // [vocab, hidden] layout
                    } else {
                        // Can't determine - use larger dimension as vocab (llama.cpp heuristic)
                        d0.max(d1)
                    }
                } else {
                    config.vocab_size
                };
                eprintln!(">>> map_embedding_lazy: actual_vocab_size={}", actual_vocab_size);

                // Check if this tensor matches expected patterns
                // Accept: [vocab, hidden] OR [hidden, vocab]
                if d0 == actual_vocab_size && d1 == hidden_size {
                    eprintln!(">>> map_embedding_lazy: Match [vocab, hidden] pattern -> RowMajor");
                    tracing::info!(
                        "Found embedding tensor '{}' with shape {:?}, inferred vocab_size={}",
                        name,
                        shape,
                        actual_vocab_size
                    );
                    return Ok((Arc::new(lazy.clone()), Layout::RowMajor));
                }

                if d0 == hidden_size && d1 == actual_vocab_size {
                    eprintln!(">>> map_embedding_lazy: Match [hidden, vocab] pattern -> ColMajor");
                    tracing::info!(
                        "Found embedding tensor '{}' with shape {:?}, inferred vocab_size={}",
                        name,
                        shape,
                        actual_vocab_size
                    );
                    return Ok((Arc::new(lazy.clone()), Layout::ColMajor));
                }
            }
        }
    }

    // Fail with evidence (llama.cpp style - list what we found)
    let mut found_tensors = Vec::new();
    for name in &embedding_names {
        if let Some(lazy) = lazy_tensors.get(*name) {
            if let Some(shape) = lazy.shape() {
                found_tensors.push(format!("{}: {:?}", name, shape));
            }
        }
    }

    let error_msg = if found_tensors.is_empty() {
        format!("No embedding tensor found (tried: {}). lazy_tensors has {} total keys. vocab_size={}, hidden_size={}",
               embedding_names.join(", "), lazy_tensors.len(), config.vocab_size, config.hidden_size)
    } else {
        format!("Found embedding tensors but shape validation failed. Found: {}. Expected 2D tensor with vocab_size={} and hidden_size={}",
               found_tensors.join(", "), config.vocab_size, config.hidden_size)
    };

    Err(HipError::GenericError(error_msg))
}

/// Map LM head to LazyTensor handle
///
/// This function implements llama.cpp-compatible LM head detection:
/// - Accepts both [vocab, hidden] and [hidden, vocab] layouts
/// - Falls back to tied embeddings (token_embd.weight) when no separate LM head exists
pub fn map_lm_head_lazy(
    lazy_tensors: &HashMap<String, LazyTensor>,
    config: &ModelConfig,
    _architecture: &Architecture,
) -> HipResult<Arc<LazyTensor>> {
    let lm_head_names = ["output.weight", "lm_head.weight", "logits.weight"];

    // Try explicit LM head tensors first
    for name in &lm_head_names {
        if let Some(lazy) = lazy_tensors.get(*name) {
            if let Some(shape) = lazy.shape() {
                if shape.len() == 2 {
                    // Accept either layout - validation happens at usage time
                    tracing::info!("Found LM head tensor '{}' with shape {:?}", name, shape);
                    return Ok(Arc::new(lazy.clone()));
                }
            }
        }
    }

    // For tied embeddings (Qwen2 style), try embedding tensors
    let tied_names = ["token_embd.weight", "embed_tokens.weight"];
    for name in &tied_names {
        if let Some(lazy) = lazy_tensors.get(*name) {
            tracing::info!("Using tied embedding '{}' as LM head", name);
            return Ok(Arc::new(lazy.clone()));
        }
    }

    Err(HipError::GenericError(format!(
        "No LM head tensor found (tried: {}). vocab_size={}, hidden_size={}",
        lm_head_names.join(", "),
        config.vocab_size,
        config.hidden_size
    )))
}

/// Build embedding GGML plan for token lookup
pub fn build_embedding_plan(
    plan: &ExecutionPlan,
    backend: &HipBackend,
    embedding_weights: &DeviceTensor,
) -> HipResult<crate::model::execution_plan::ggml_plan::EmbeddingGgmlPlan> {
    use crate::ggml::hip_backend::HipGgmlBackend;
    use crate::ggml::{Graph, Layout, Op, TensorDesc, DType};
    use crate::model::execution_plan::ggml_plan::EmbeddingGgmlPlan;
    use crate::model::execution_plan::types::ExecutionPlan;
    use std::sync::Mutex as StdMutex;
    use std::sync::Arc;

    let embed_shape = embedding_weights.shape().dims();
    eprintln!(">>> build_embedding_plan: embed_shape={:?}", embed_shape);
    eprintln!(">>> build_embedding_plan: plan.config.hidden_size={}, plan.config.vocab_size={}",
        plan.config_internal().hidden_size, plan.config_internal().vocab_size);

    if embed_shape.len() != 2 {
        return Err(HipError::GenericError(format!(
            "Embedding weight shape must be 2D, got {:?}",
            embed_shape
        )));
    }

    // Detect layout from actual tensor shape (not plan.embedding_layout which may be stale)
    // After transpose in embedding_weights(), the tensor is always [vocab, hidden]
    let (n_embd, vocab_size) = (embed_shape[1], embed_shape[0]);
    eprintln!(">>> build_embedding_plan: n_embd={}, vocab_size={}", n_embd, vocab_size);

    if n_embd != plan.config_internal().hidden_size {
        return Err(HipError::GenericError(format!(
            "Embedding hidden size mismatch: expected {}, got {}",
            plan.config_internal().hidden_size, n_embd
        )));
    }

    let max_seq_len = std::cmp::max(1, plan.config_internal().max_position_embeddings);
    let tokens_bytes = max_seq_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            HipError::GenericError("Token buffer size overflow".to_string())
        })?;
    let output_elems = max_seq_len.checked_mul(n_embd).ok_or_else(|| {
        HipError::GenericError("Output buffer element overflow".to_string())
    })?;
    let output_bytes = output_elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HipError::GenericError("Output buffer size overflow".to_string()))?;

    let mut graph = Graph::new();
    // After transpose, embedding is [vocab, hidden] which is RowMajor layout
    // Do NOT use plan.embedding_layout which is for the original GGUF [hidden, vocab] format
    let weights_id = graph.add_tensor(TensorDesc::new(
        embed_shape.to_vec(),
        DType::F32,
        Layout::RowMajor,  // Transposed format is always RowMajor [vocab, hidden]
    ));
    let tokens_id = graph.add_tensor(TensorDesc::new(
        vec![max_seq_len],
        DType::U32,
        Layout::RowMajor,
    ));
    let output_id = graph.add_tensor(TensorDesc::new(
        vec![max_seq_len, n_embd],
        DType::F32,
        Layout::RowMajor,
    ));
    graph.add_node(Op::GetRows, vec![weights_id, tokens_id], vec![output_id]);

    let mut ggml_backend = HipGgmlBackend::new(Arc::clone(plan.backend()));
    let weights_desc = graph.tensors[weights_id.0].clone();
    ggml_backend
        .bind(&weights_desc, embedding_weights.buffer().clone())
        .map_err(|e| HipError::GenericError(format!("GetRows bind weights failed: {:?}", e)))?;

    let tokens_desc = graph.tensors[tokens_id.0].clone();
    let tokens_buffer = backend.allocate_buffer(tokens_bytes)?;
    ggml_backend
        .bind(&tokens_desc, tokens_buffer.clone())
        .map_err(|e| HipError::GenericError(format!("GetRows bind tokens failed: {:?}", e)))?;

    let output_desc = graph.tensors[output_id.0].clone();
    let output_buffer = backend.allocate_buffer(output_bytes)?;
    ggml_backend
        .bind(&output_desc, output_buffer.clone())
        .map_err(|e| HipError::GenericError(format!("GetRows bind output failed: {:?}", e)))?;

    Ok(EmbeddingGgmlPlan {
        graph,
        backend: StdMutex::new(ggml_backend),
        tokens_buffer,
        output_buffer,
        max_seq_len,
        hidden_size: n_embd,
    })
}

/// Token embedding lookup
///
/// Converts token IDs to embeddings using the embedding weight matrix.
pub fn embedding_lookup(
    plan: &ExecutionPlan,
    backend: &HipBackend,
    input_tokens: &[u32],
    embedding_weights: &DeviceTensor,
) -> HipResult<DeviceTensor> {
    use crate::ggml::executor::execute_graph;
    use crate::model::execution_plan::ggml_plan::EmbeddingGgmlPlan;

    eprintln!(
        ">>> embedding_lookup: Starting with {} tokens",
        input_tokens.len()
    );
    eprintln!(">>> embedding_lookup: Getting seq_len and hidden_size...");
    let seq_len = input_tokens.len();
    let hidden_size = plan.config_internal().hidden_size;
    eprintln!(
        ">>> embedding_lookup: seq_len={}, hidden_size={}",
        seq_len, hidden_size
    );

    eprintln!(">>> embedding_lookup: About to call embedding_weights.shape().dims()...");
    let embed_shape = embedding_weights.shape().dims();
    eprintln!(">>> embedding_lookup: Got shape {:?}", embed_shape);

    eprintln!(">>> embedding_lookup: About to call get_or_try_init on embedding_plan...");
    let plan_cached = match plan.embedding_plan().get_or_try_init(|| {
        eprintln!(">>> embedding_lookup: Inside get_or_try_init closure, calling build_embedding_plan...");
        let result = build_embedding_plan(plan, backend, embedding_weights);
        match &result {
            Ok(_) => eprintln!(">>> embedding_lookup: build_embedding_plan returned Ok"),
            Err(e) => eprintln!(">>> embedding_lookup: build_embedding_plan returned Err: {}", e),
        }
        result
    }) {
        Ok(p) => {
            eprintln!(">>> embedding_lookup: get_or_try_init succeeded");
            p
        }
        Err(e) => {
            eprintln!(">>> embedding_lookup: get_or_try_init failed: {}", e);
            return Err(e);
        }
    };

    if seq_len > plan_cached.max_seq_len {
        return Err(HipError::GenericError(format!(
            "Sequence length {} exceeds embedding capacity {}",
            seq_len, plan_cached.max_seq_len
        )));
    }

    if plan_cached.hidden_size != hidden_size {
        return Err(HipError::GenericError(format!(
            "Embedding hidden size mismatch: expected {}, got {}",
            hidden_size, plan_cached.hidden_size
        )));
    }

    // Validate token IDs
    // After transpose, embedding is always [vocab, hidden], so vocab is embed_shape[0]
    let vocab_size = embed_shape[0];
    for &token_id in input_tokens {
        if token_id as usize >= vocab_size {
            return Err(HipError::GenericError(format!(
                "Token ID {} exceeds vocabulary size {}",
                token_id, vocab_size
            )));
        }
    }

    // Create output tensor: [seq_len, hidden_size]
    let output_shape = TensorShape::from_dims(&[seq_len, hidden_size]);
    eprintln!(">>> embedding_lookup: Creating output tensor and executing GetRows...");
    eprintln!(
        ">>> embedding_lookup: About to write {} tokens into shared buffer",
        input_tokens.len()
    );
    let mut padded_tokens = vec![0u32; plan_cached.max_seq_len];
    padded_tokens[..seq_len].copy_from_slice(input_tokens);
    eprintln!(">>> embedding_lookup: About to call copy_to_device...");
    backend.copy_to_device(&plan_cached.tokens_buffer, &padded_tokens)?;
    eprintln!(">>> embedding_lookup: copy_to_device complete");

    eprintln!(">>> embedding_lookup: About to lock ggml_backend...");
    let mut ggml_backend = plan_cached.backend.lock().map_err(|_| {
        HipError::GenericError("GetRows backend lock poisoned".to_string())
    })?;
    eprintln!(">>> embedding_lookup: ggml_backend locked, about to execute_graph...");
    execute_graph(&mut *ggml_backend, &plan_cached.graph)
        .map_err(|e| HipError::GenericError(format!("GetRows execute failed: {:?}", e)))?;
    eprintln!(">>> embedding_lookup: execute_graph complete");

    let output_bytes = seq_len
        .checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| HipError::GenericError("Output view size overflow".to_string()))?;
    let output_view = plan_cached.output_buffer.sub_buffer_view(0, output_bytes)?;
    let hidden_states = DeviceTensor::from_buffer(backend, output_view, output_shape)?;

    eprintln!(">>> embedding_lookup: Complete");
    Ok(hidden_states)
}
