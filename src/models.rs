//! Helper utilities for discovering available GGUF models on disk.

use crate::loader::{GgufLoader, GgufMetadata};
use crate::tokenizer::infer_tokenizer_path;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;
use tracing::{info, warn};

const MAX_TOKENIZER_CACHE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadataSummary {
    pub architecture: String,
    pub num_layers: usize,
    pub num_heads: usize,
    pub hidden_size: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub file_type: u32,
    pub has_tokenizer: bool,
}

impl From<GgufMetadata> for ModelMetadataSummary {
    fn from(meta: GgufMetadata) -> Self {
        let has_tokenizer = meta.embedded_tokenizer_json.is_some();
        Self {
            architecture: meta.architecture,
            num_layers: meta.num_layers,
            num_heads: meta.num_heads,
            hidden_size: meta.hidden_size,
            head_dim: if meta.head_dim > 0 {
                meta.head_dim
            } else if meta.num_heads > 0 {
                meta.hidden_size / meta.num_heads
            } else {
                0
            },
            max_position_embeddings: meta.max_position_embeddings,
            vocab_size: meta.vocab_size,
            file_type: meta.file_type,
            has_tokenizer,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub tokenizer: Option<String>,
    pub metadata: Option<ModelMetadataSummary>,
    pub cache_status: Option<CacheStatus>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheStatus {
    pub cached: bool,
    pub refreshed: bool,
}

#[derive(Debug, Clone)]
pub struct CachedTokenizer {
    pub json: String,
    pub cached: bool,
    pub refreshed: bool,
}

pub fn discover_models(dir_override: Option<&str>) -> Result<Vec<ModelInfo>> {
    let (models, _) = discover_models_with_cache(dir_override)?;
    Ok(models)
}

/// Recursively collect all GGUF files in a directory tree
fn collect_gguf_files(base_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut gguf_files = Vec::new();
    collect_gguf_files_recursive(base_dir, &mut gguf_files)?;
    Ok(gguf_files)
}

/// Helper function to recursively collect GGUF files
fn collect_gguf_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory {}", dir.display()))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recursively search subdirectories
            collect_gguf_files_recursive(&path, files)?;
        } else if path.is_file() {
            // Check if it's a GGUF file
            if path
                .extension()
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
            {
                files.push(path);
            }
        }
    }

    Ok(())
}

pub fn discover_models_with_cache(dir_override: Option<&str>) -> Result<(Vec<ModelInfo>, u64)> {
    let base_dir = resolve_models_dir(dir_override)?;
    if !base_dir.exists() {
        return Ok((Vec::new(), 0));
    }

    let mut entries = Vec::new();
    let mut cache = MetadataCache::load(&base_dir);
    let mut cache_dirty = false;
    let mut seen_paths = HashSet::new();

    // Recursively collect all GGUF files
    let gguf_files = collect_gguf_files(&base_dir)?;
    tracing::debug!("Found {} GGUF files in {}", gguf_files.len(), base_dir.display());

    for path in gguf_files {
        let tokenizer = infer_tokenizer_path(path.to_string_lossy().as_ref());
        let (metadata, cache_status) =
            match load_or_cache_metadata(&path, &base_dir, &mut cache, false).map(|info| {
                cache_dirty |= info.updated_cache;
                (
                    info.summary,
                    CacheStatus {
                        cached: info.cached,
                        refreshed: info.updated_cache,
                    },
                )
            }) {
                Ok(summary) => summary,
                Err(err) => {
                    warn!(
                        "Failed to inspect GGUF metadata for {}: {}",
                        path.display(),
                        err
                    );
                    (
                        None,
                        CacheStatus {
                            cached: false,
                            refreshed: false,
                        },
                    )
                }
            };
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        seen_paths.insert(path.to_string_lossy().into_owned());
        entries.push(ModelInfo {
            name,
            path: path.to_string_lossy().into_owned(),
            tokenizer,
            metadata,
            cache_status: Some(cache_status),
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    cache_dirty |= purge_stale_entries(&base_dir, &mut cache, &seen_paths);
    cache_dirty |= cleanup_tokenizer_blobs(&base_dir, &cache);
    if cache_dirty {
        cache.save(&base_dir);
    }
    let cache_bytes = cache.total_blob_bytes(&base_dir);
    Ok((entries, cache_bytes))
}

fn resolve_models_dir(dir_override: Option<&str>) -> Result<PathBuf> {
    if let Some(dir) = dir_override {
        return Ok(PathBuf::from(dir));
    }

    if let Ok(env_dir) = std::env::var("ROCMFORGE_MODELS") {
        return Ok(PathBuf::from(env_dir));
    }

    Ok(Path::new("models").to_path_buf())
}

pub fn cached_embedded_tokenizer(path: &str) -> Option<CachedTokenizer> {
    let path_buf = PathBuf::from(path);
    let dir = path_buf.parent()?;
    let mut cache = MetadataCache::load(dir);
    let result = load_or_cache_metadata(&path_buf, dir, &mut cache, true).ok()?;
    if result.updated_cache {
        cache.save(dir);
    }
    result.tokenizer_json.map(|json| CachedTokenizer {
        json,
        cached: result.cached,
        refreshed: result.updated_cache,
    })
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct MetadataCache {
    entries: HashMap<String, CachedMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedMetadata {
    modified: u64,
    metadata: ModelMetadataSummary,
    tokenizer_blob: Option<String>,
}

struct MetadataResult {
    summary: Option<ModelMetadataSummary>,
    tokenizer_json: Option<String>,
    cached: bool,
    updated_cache: bool,
}

impl MetadataCache {
    fn total_blob_bytes(&self, base_dir: &Path) -> u64 {
        self.entries
            .values()
            .filter_map(|entry| entry.tokenizer_blob.as_ref())
            .filter_map(|name| {
                fs::metadata(tokenizer_blob_dir(base_dir).join(name))
                    .ok()
                    .map(|m| m.len())
            })
            .sum()
    }
}

impl MetadataCache {
    fn load(base_dir: &Path) -> Self {
        let path = cache_path(base_dir);
        if let Ok(bytes) = fs::read(&path) {
            if let Ok(cache) = serde_json::from_slice(&bytes) {
                return cache;
            }
        }
        MetadataCache::default()
    }

    fn save(&self, base_dir: &Path) {
        let path = cache_path(base_dir);
        if let Ok(json) = serde_json::to_vec_pretty(self) {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let _ = fs::write(path, json);
        }
    }
}

fn cache_path(base_dir: &Path) -> PathBuf {
    base_dir.join(".rocmforge_metadata_cache.json")
}

fn purge_stale_entries(base_dir: &Path, cache: &mut MetadataCache, seen: &HashSet<String>) -> bool {
    let mut removed = false;
    let keys: Vec<String> = cache
        .entries
        .keys()
        .filter(|path| !seen.contains(*path))
        .cloned()
        .collect();
    for key in keys {
        if let Some(entry) = cache.entries.remove(&key) {
            if let Some(blob) = entry.tokenizer_blob {
                remove_tokenizer_blob(base_dir, &blob);
            }
            removed = true;
        }
    }
    removed
}

fn cleanup_tokenizer_blobs(base_dir: &Path, cache: &MetadataCache) -> bool {
    let dir = tokenizer_blob_dir(base_dir);
    let Ok(read_dir) = fs::read_dir(&dir) else {
        return false;
    };
    let valid: HashSet<String> = cache
        .entries
        .values()
        .filter_map(|entry| entry.tokenizer_blob.clone())
        .collect();
    let mut removed = false;
    let mut valid_files = Vec::new();
    let mut total_size = 0u64;
    for entry in read_dir.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let metadata = match fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !valid.contains(name) {
            if fs::remove_file(&path).is_ok() {
                info!("Removed stale tokenizer blob {}", path.display());
                removed = true;
            }
            continue;
        }
        let len = metadata.len();
        let modified = metadata.modified().ok();
        total_size += len;
        valid_files.push((path, len, modified));
    }

    if total_size > MAX_TOKENIZER_CACHE_BYTES {
        valid_files.sort_by_key(|(_, len, modified)| {
            (
                modified
                    .and_then(|ts| ts.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                std::cmp::Reverse(*len), // Sort by size descending (largest first) when times are equal
            )
        });
        for (path, len, _) in valid_files {
            if total_size <= MAX_TOKENIZER_CACHE_BYTES {
                break;
            }
            if fs::remove_file(&path).is_ok() {
                info!(
                    "Pruned tokenizer blob {} to keep cache under {} bytes",
                    path.display(),
                    MAX_TOKENIZER_CACHE_BYTES
                );
                removed = true;
                total_size = total_size.saturating_sub(len);
            }
        }
    }

    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn dummy_metadata() -> ModelMetadataSummary {
        ModelMetadataSummary {
            architecture: "test".into(),
            num_layers: 1,
            num_heads: 1,
            hidden_size: 1,
            head_dim: 1,
            max_position_embeddings: 1,
            vocab_size: 1,
            file_type: 0,
            has_tokenizer: true,
        }
    }

    #[test]
    fn cleanup_removes_orphans_and_enforces_limit() {
        let temp = tempdir().unwrap();
        let base = temp.path();
        let tokenizer_dir = tokenizer_blob_dir(base);
        fs::create_dir_all(&tokenizer_dir).unwrap();

        let keep_path = tokenizer_dir.join("keep.json");
        fs::write(&keep_path, b"valid").unwrap();
        let orphan_path = tokenizer_dir.join("orphan.json");
        fs::write(&orphan_path, b"stale").unwrap();
        // Large file to trigger cap (70MB)
        let large_blob = tokenizer_dir.join("large.json");
        fs::write(
            &large_blob,
            vec![0u8; (MAX_TOKENIZER_CACHE_BYTES as usize) + 1024],
        )
        .unwrap();

        let mut cache = MetadataCache::default();
        cache.entries.insert(
            "model.gguf".into(),
            CachedMetadata {
                modified: 0,
                metadata: dummy_metadata(),
                tokenizer_blob: Some("keep.json".into()),
            },
        );
        cache.entries.insert(
            "model_large.gguf".into(),
            CachedMetadata {
                modified: 0,
                metadata: dummy_metadata(),
                tokenizer_blob: Some("large.json".into()),
            },
        );

        assert!(cleanup_tokenizer_blobs(base, &cache));
        assert!(keep_path.exists());
        assert!(!orphan_path.exists());

        // After cleanup, cache directory should be under limit
        let total = cache.total_blob_bytes(base);
        assert!(total <= MAX_TOKENIZER_CACHE_BYTES);
    }
}
fn load_or_cache_metadata(
    path: &Path,
    base_dir: &Path,
    cache: &mut MetadataCache,
    load_tokenizer: bool,
) -> Result<MetadataResult> {
    let path_str = path.to_string_lossy().into_owned();
    let modified = fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|ts| ts.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs());

    if let (Some(mtime), Some(entry)) = (modified, cache.entries.get(&path_str)) {
        if entry.modified == mtime {
            let tokenizer_json = if load_tokenizer {
                entry
                    .tokenizer_blob
                    .as_ref()
                    .and_then(|rel| read_tokenizer_blob(base_dir, rel))
            } else {
                None
            };
            return Ok(MetadataResult {
                summary: Some(entry.metadata.clone()),
                tokenizer_json,
                cached: true,
                updated_cache: false,
            });
        }
    }

    let meta = GgufLoader::metadata_from_file(&path_str)?;
    let tokenizer_json = if load_tokenizer {
        meta.embedded_tokenizer_json.clone()
    } else {
        None
    };
    let tokenizer_rel = meta
        .embedded_tokenizer_json
        .as_ref()
        .and_then(|json| store_tokenizer_blob(base_dir, json));
    let summary = ModelMetadataSummary::from(meta);
    if let Some(mtime) = modified {
        if let Some(old) = cache.entries.insert(
            path_str.clone(),
            CachedMetadata {
                modified: mtime,
                metadata: summary.clone(),
                tokenizer_blob: tokenizer_rel.clone(),
            },
        ) {
            if let Some(blob) = old.tokenizer_blob {
                if Some(blob.as_str()) != tokenizer_rel.as_deref() {
                    remove_tokenizer_blob(base_dir, &blob);
                }
            }
        }
    }
    Ok(MetadataResult {
        summary: Some(summary),
        tokenizer_json,
        cached: false,
        updated_cache: modified.is_some(),
    })
}

fn tokenizer_blob_dir(base_dir: &Path) -> PathBuf {
    base_dir.join(".rocmforge_tokenizers")
}

fn read_tokenizer_blob(base_dir: &Path, rel: &str) -> Option<String> {
    let path = tokenizer_blob_dir(base_dir).join(rel);
    fs::read_to_string(path).ok()
}

fn store_tokenizer_blob(base_dir: &Path, json: &str) -> Option<String> {
    let mut hasher = DefaultHasher::new();
    hasher.write(json.as_bytes());
    let hash = hasher.finish();
    let filename = format!("{:016x}.json", hash);
    let dir = tokenizer_blob_dir(base_dir);
    if fs::create_dir_all(&dir).is_err() {
        return None;
    }
    let path = dir.join(&filename);
    if fs::write(&path, json).is_err() {
        return None;
    }
    Some(filename)
}

fn remove_tokenizer_blob(base_dir: &Path, rel: &str) {
    let path = tokenizer_blob_dir(base_dir).join(rel);
    let _ = fs::remove_file(path);
}
