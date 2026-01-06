# Knowledge-Level Caching with SQLiteGraph

**Date**: January 6, 2026
**Status**: Design Document
**Goal**: Transform GPU reasoning from a recurring expense into a capital investment

---

## Executive Summary

This document describes how to use **sqlitegraph** to implement a knowledge-level caching system for ROCmForge. The system avoids re-paying GPU compute for facts, invariants, and conclusions that are already settled—while allowing them to decay, be challenged, and be superseded.

**One-Line Summary**: We're turning GPU reasoning from a recurring expense into a capital investment.

---

## What This Is (and What It Isn't)

### This IS:
- A mechanism to cache GPU-derived facts as versioned, testable artifacts
- A system to reuse settled knowledge without re-computation
- A graph-based provenance tracking system ("why do we believe this?")
- A confidence-decay system for knowledge freshness

### This is NOT:
- Chat memory
- RAG (Retrieval-Augmented Generation)
- Prompt caching
- "The model remembers conversations"
- "Self-modifying weights"

---

## Why SQLiteGraph?

SQLiteGraph provides the ideal foundation for this system:

| Requirement | SQLiteGraph Capability |
|-------------|------------------------|
| **Graph Structure** | Native nodes/edges with typed relationships |
| **Provenance Tracking** | Traversal algorithms (BFS, k-hop, chain queries) |
| **Versioning** | Snapshot export/import, entity versioning |
| **Persistence** | SQLite backend (ACID) or Native V2 (performance) |
| **Query Flexibility** | Pattern matching, filtered traversals |
| **Embeddings** | HNSW vector search for semantic similarity |

---

## Data Model Design

### Core Concept: Facts as Graph Entities

```rust
/// A cached fact derived from GPU inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    /// Unique identifier (content hash + version)
    pub id: FactId,

    /// The fact content
    pub content: FactContent,

    /// Current confidence [0.0, 1.0]
    pub confidence: f64,

    /// Fact type (determines caching behavior)
    pub fact_type: FactType,

    /// When this was derived
    pub derived_at: SystemTime,

    /// Version for superseding
    pub version: u64,

    /// Supersedes previous fact (if any)
    pub supersedes: Option<FactId>,

    /// Evidence artifacts supporting this fact
    pub evidence: Vec<EvidenceId>,

    /// Tags for efficient lookup
    pub tags: BTreeSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactType {
    /// Stable factual knowledge (slow decay)
    StableFact,

    /// Derived invariant (medium decay)
    Invariant,

    /// Conclusion from reasoning (fast decay)
    Conclusion,

    /// Code analysis result
    CodeAnalysis,

    /// API contract information
    ApiContract,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactContent {
    /// A textual statement
    Statement { text: String },

    /// Code analysis result
    CodeAnalysis {
        language: String,
        file_path: PathBuf,
        finding: String,
        location: CodeLocation,
    },

    /// API contract
    ApiContract {
        api_name: String,
        signature: String,
        behavior: String,
    },

    /// Invariant
    Invariant {
        description: String,
        scope: InvariantScope,
    },

    /// Conclusion from reasoning
    Conclusion {
        premise: Vec<String>,
        reasoning: String,
        conclusion: String,
    },
}

/// Evidence artifact (immutable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub id: EvidenceId,

    /// Evidence type
    pub evidence_type: EvidenceType,

    /// Content reference
    pub content_ref: ContentRef,

    /// Timestamp of evidence collection
    pub collected_at: SystemTime,

    /// Hash for immutability verification
    pub content_hash: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Source code
    SourceCode { language: String, file_path: PathBuf },

    /// Documentation
    Documentation { source: String, url: Option<String> },

    /// Execution trace
    ExecutionTrace { format: String },

    /// External reference
    ExternalReference { url: String },
}
```

### Mapping to SQLiteGraph Schema

```sql
-- ==========================================
-- Fact Entities (Nodes)
-- ==========================================
-- Graph entity represents a Fact
-- kind = "Fact"
-- name = fact content hash
-- data = JSON with {confidence, version, derived_at, supersedes, tags}

-- ==========================================
-- Derivation Edges
-- ==========================================
-- Graph edge represents "derived_from" relationship
-- from_id = dependent fact (derived)
-- to_id = source fact (dependency)
-- edge_type = "derived_from"
-- data = JSON with {operation, confidence_delta}

-- ==========================================
-- Evidence Entities (Nodes)
-- ==========================================
-- Graph entity represents Evidence
-- kind = "Evidence"
-- name = evidence ID
-- data = JSON with {evidence_type, content_ref, collected_at, hash}

-- ==========================================
-- Support Edges
-- ==========================================
-- Graph edge represents "supported_by" relationship
-- from_id = fact
-- to_id = evidence
-- edge_type = "supported_by"
-- data = JSON with {role, strength}

-- ==========================================
-- Supersedes Edges
-- ==========================================
-- Graph edge represents "supersedes" relationship
-- from_id = newer fact
-- to_id = older fact
-- edge_type = "supersedes"
-- data = JSON with {reason, timestamp}
```

---

## Implementation with SQLiteGraph

### 1. Database Schema Setup

```rust
use sqlitegraph::{SqliteGraph, GraphEntity, GraphEdge, GraphConfig};
use serde_json::json;

/// Knowledge cache built on sqlitegraph
pub struct KnowledgeCache {
    graph: SqliteGraph,
    config: CacheConfig,
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Default confidence threshold for cache hits
    pub min_confidence: f64,

    /// Maximum age for cached facts
    pub max_fact_age: Duration,

    /// Confidence decay rate per day
    pub decay_rate_per_day: f64,

    /// Minimum confidence before recomputation
    pub min_recompute_confidence: f64,
}

impl KnowledgeCache {
    /// Open or create a knowledge cache database
    pub fn open(path: &Path) -> Result<Self> {
        let cfg = GraphConfig::sqlite(); // Could use native for performance
        let mut graph = SqliteGraph::open(path, &cfg)?;

        // Run migrations if needed
        Self::ensure_schema(&mut graph)?;

        Ok(Self {
            graph,
            config: CacheConfig::default(),
        })
    }

    fn ensure_schema(graph: &mut SqliteGraph) -> Result<()> {
        // Create indexes for common queries
        // (SQLiteGraph handles basic schema)
        Ok(())
    }
}
```

### 2. Storing Facts

```rust
impl KnowledgeCache {
    /// Store a newly derived fact
    pub fn store_fact(
        &mut self,
        content: FactContent,
        fact_type: FactType,
        evidence: Vec<Evidence>,
        dependencies: Vec<FactId>,
        confidence: f64,
    ) -> Result<FactId> {
        // 1. Generate fact ID from content hash
        let content_json = serde_json::to_string(&content)?;
        let content_hash = blake3::hash(content_json.as_bytes());
        let fact_id = FactId::from_hash(content_hash, 0);

        // 2. Create fact entity
        let entity = GraphEntity {
            id: 0, // Auto-assigned
            kind: "Fact".to_string(),
            name: fact_id.to_string(),
            file_path: None,
            data: json!({
                "content": content,
                "fact_type": fact_type,
                "confidence": confidence,
                "derived_at": SystemTime::now()
                    .duration_since(UNIX_EPOCH)?
                    .as_secs(),
                "version": 0,
                "supersedes": null::<String>,
                "tags": []
            }),
        };

        let fact_entity_id = self.graph.insert_entity(&entity)?;

        // 3. Store evidence entities and support edges
        for evidence_item in evidence {
            let ev_id = self.store_evidence(evidence_item)?;
            self.add_support_edge(fact_entity_id, ev_id, "primary")?;
        }

        // 4. Create derivation edges (dependencies)
        for dep_id in dependencies {
            self.add_derivation_edge(fact_entity_id, dep_id)?;
        }

        // 5. Store in confidence tracking
        self.store_confidence(fact_entity_id, confidence)?;

        Ok(fact_id)
    }

    /// Store an evidence artifact
    fn store_evidence(&mut self, evidence: Evidence) -> Result<i64> {
        let entity = GraphEntity {
            id: 0,
            kind: "Evidence".to_string(),
            name: evidence.id.to_string(),
            file_path: Some(format!("{:?}", evidence.content_ref)),
            data: json!({
                "evidence_type": evidence.evidence_type,
                "content_ref": evidence.content_ref,
                "collected_at": evidence.collected_at,
                "hash": evidence.content_hash
            }),
        };

        self.graph.insert_entity(&entity)
    }

    /// Add "supported_by" edge from fact to evidence
    fn add_support_edge(&mut self, fact_id: i64, evidence_id: i64, role: &str)
        -> Result<i64>
    {
        let edge = GraphEdge {
            id: 0,
            from_id: fact_id,
            to_id: evidence_id,
            edge_type: "supported_by".to_string(),
            data: json!({
                "role": role,
                "strength": 1.0
            }),
        };

        self.graph.insert_edge(&edge)
    }

    /// Add "derived_from" edge from fact to dependency
    fn add_derivation_edge(&mut self, fact_id: i64, dep_id: FactId) -> Result<i64> {
        let edge = GraphEdge {
            id: 0,
            from_id: fact_id,
            to_id: self.find_entity_by_name(&dep_id.to_string())?,
            edge_type: "derived_from".to_string(),
            data: json!({
                "operation": "and",
                "confidence_delta": 0.0
            }),
        };

        self.graph.insert_edge(&edge)
    }
}
```

### 3. Cache Lookup

```rust
impl KnowledgeCache {
    /// Look up a fact by query semantic
    pub fn lookup(&self, query: &CacheQuery) -> Result<CacheLookup> {
        // 1. Normalize query to semantic hash
        let query_hash = self.normalize_query(query)?;

        // 2. Look for fact entity
        let fact_entity = self.graph.entity_by_name(&query_hash.to_string())?;

        let fact_entity = match fact_entity {
            Some(e) => e,
            None => return Ok(CacheLookup::Miss),
        };

        // 3. Parse fact data
        let fact: Fact = self.parse_fact_entity(&fact_entity)?;

        // 4. Check confidence threshold
        if fact.confidence < query.min_confidence {
            return Ok(CacheLookup::Stale {
                reason: StaleReason::LowConfidence {
                    current: fact.confidence,
                    required: query.min_confidence,
                },
            });
        }

        // 5. Check age constraint
        let age = SystemTime::now().duration(fact.derived_at)?;
        if let Some(max_age) = query.max_age {
            if age > max_age {
                return Ok(CacheLookup::Stale {
                    reason: StaleReason::Expired { age, max_age },
                });
            }
        }

        // 6. Check if superseded
        if let Some(superseder_id) = self.check_superseded(fact_entity.id)? {
            return Ok(CacheLookup::Stale {
                reason: StaleReason::Superseded { by: superseder_id },
            });
        }

        // 7. Verify evidence integrity
        if let Some(changed) = self.verify_evidence(&fact)? {
            return Ok(CacheLookup::Stale {
                reason: StaleReason::EvidenceChanged { changed_evidence: changed },
            });
        }

        // 8. Build provenance chain
        let provenance = self.build_provenance(fact_entity.id)?;

        // 9. Update access statistics
        self.record_access(fact_entity.id)?;

        Ok(CacheLookup::Hit {
            fact,
            provenance,
        })
    }

    /// Build provenance chain using graph traversal
    fn build_provenance(&self, fact_entity_id: i64) -> Result<ProvenanceChain> {
        let mut hops = Vec::new();
        let mut visited = std::collections::HashSet::new();

        // Use sqlitegraph's BFS traversal
        let neighbors = self.graph.bfs(fact_entity_id, 10)?;

        for (depth, neighbor_id) in neighbors.iter().enumerate() {
            if !visited.insert(*neighbor_id) {
                continue;
            }

            if let Some(entity) = self.graph.entity(*neighbor_id)? {
                if entity.kind == "Fact" {
                    hops.push(ProvenanceHop {
                        fact_id: entity.name.clone(),
                        content: self.parse_content_from_data(&entity.data)?,
                        depth: depth as u32,
                    });
                }
            }
        }

        Ok(ProvenanceChain { hops })
    }

    /// Verify evidence hasn't changed
    fn verify_evidence(&self, fact: &Fact) -> Result<Option<Vec<EvidenceId>>> {
        let mut changed = Vec::new();

        // Get evidence neighbors
        let evidence_ids = self.graph.neighbors(
            self.find_entity_by_name(&fact.id.to_string())?,
            NeighborQuery::OutgoingWithTypes(&["supported_by"]),
        )?;

        for ev_id in evidence_ids {
            if let Some(ev_entity) = self.graph.entity(ev_id)? {
                let ev: Evidence = self.parse_evidence_entity(&ev_entity)?;

                // Re-verify hash
                match &ev.content_ref {
                    ContentRef::File { path } => {
                        let current_hash = hash_file(path)?;
                        if current_hash != ev.content_hash {
                            changed.push(ev.id.clone());
                        }
                    }
                    _ => continue,
                }
            }
        }

        Ok(if changed.is_empty() { None } else { Some(changed) })
    }
}
```

### 4. Confidence Decay

```rust
impl KnowledgeCache {
    /// Calculate current confidence with time decay
    pub fn calculate_confidence(&self, fact_id: &FactId) -> Result<f64> {
        let entity = self.graph.entity_by_name(&fact_id.to_string())?
            .ok_or anyhow!("Fact not found")?;

        let data: serde_json::Value = serde_json::from_str(&entity.data)?;
        let initial_confidence = data["confidence"].as_f64().unwrap_or(1.0);
        let derived_at_secs = data["derived_at"].as_u64().unwrap_or(0);

        // Calculate elapsed days
        let elapsed_days = (SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs() - derived_at_secs) as f64 / 86400.0;

        // Apply exponential decay
        let decayed = initial_confidence * (-self.config.decay_rate_per_day * elapsed_days).exp();

        // Apply access boost (read from access counter)
        let access_count = self.get_access_count(fact_id)?;
        let access_bonus = 0.05 * (access_count as f64).sqrt().min(3.0);

        Ok((decayed + access_bonus).min(1.0).max(0.0))
    }

    /// Run maintenance: decay all facts and clean up expired ones
    pub fn run_maintenance(&mut self) -> Result<MaintenanceStats> {
        let mut stats = MaintenanceStats::default();

        // Get all fact entities
        let all_facts = self.graph.entities_by_kind("Fact")?;

        for fact_entity in all_facts {
            let fact_id: FactId = fact_entity.name.clone().into();

            // Calculate current confidence
            let current_conf = self.calculate_confidence(&fact_id)?;

            if current_conf < self.config.min_recompute_confidence {
                // Mark for potential recomputation
                stats.expired_count += 1;

                // Store updated confidence
                self.update_confidence(&fact_id, current_conf)?;

                // Create "revalidate" task
                self.create_revalidation_task(&fact_id)?;
            } else {
                // Store updated confidence
                self.update_confidence(&fact_id, current_conf)?;
                stats.preserved_count += 1;
            }
        }

        // Check evidence changes
        let changed = self.check_all_evidence()?;
        stats.evidence_changed = changed.len();

        Ok(stats)
    }

    /// Check if fact has been superseded
    fn check_superseded(&self, fact_entity_id: i64) -> Result<Option<FactId>> {
        // Look for "supersedes" incoming edges
        let superseding = self.graph.neighbors(
            fact_entity_id,
            NeighborQuery::IncomingWithTypes(&["supersedes"]),
        )?;

        if let Some(first_superseder) = superseding.first() {
            // Get the fact entity that supersedes this one
            if let Some(entity) = self.graph.entity(*first_superseder)? {
                return Ok(Some(FactId::from_string(&entity.name)));
            }
        }

        Ok(None)
    }
}
```

### 5. Invalidation

```rust
impl KnowledgeCache {
    /// Invalidate a fact (mark as inactive)
    pub fn invalidate(&mut self, fact_id: &FactId, reason: &str) -> Result<()> {
        let entity_id = self.find_entity_by_name(&fact_id.to_string())?
            .ok_or anyhow!("Fact not found")?;

        // Mark as inactive in data
        let updated_data = json!({
            "active": false,
            "invalidated_at": SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_secs(),
            "invalidation_reason": reason
        });

        self.graph.update_entity_data(entity_id, updated_data)?;

        // Cascade to dependents
        self.cascade_invalidation(entity_id)?;

        Ok(())
    }

    /// Cascade invalidation to all facts that depend on this one
    fn cascade_invalidation(&mut self, invalidated_id: i64) -> Result<()> {
        // Find all facts that derive_from this one
        let dependents = self.graph.neighbors(
            invalidated_id,
            NeighborQuery::IncomingWithTypes(&["derived_from"]),
        )?;

        for dep_id in dependents {
            // Recursively invalidate
            self.invalidate(
                &FactId::from_entity_id(dep_id)?,
                "dependency_invalidated"
            )?;
        }

        Ok(())
    }

    /// Challenge a fact (user-initiated)
    pub fn challenge(&mut self, fact_id: &FactId, reason: String) -> Result<()> {
        // Create challenge record
        let challenge_entity = GraphEntity {
            id: 0,
            kind: "Challenge".to_string(),
            name: format!("challenge_{}", uuid::Uuid::new_v4()),
            file_path: None,
            data: json!({
                "fact_id": fact_id.to_string(),
                "reason": reason,
                "challenged_at": SystemTime::now()
                    .duration_since(UNIX_EPOCH)?
                    .as_secs(),
                "resolved": false
            }),
        };

        let challenge_id = self.graph.insert_entity(&challenge_entity)?;

        // Link challenge to fact
        let edge = GraphEdge {
            id: 0,
            from_id: challenge_id,
            to_id: self.find_entity_by_name(&fact_id.to_string())?,
            edge_type: "challenges".to_string(),
            data: json!({}),
        };

        self.graph.insert_edge(&edge)?;

        // Reduce confidence
        let current_conf = self.calculate_confidence(fact_id)?;
        let new_confidence = (current_conf - 0.3).max(0.0);
        self.update_confidence(fact_id, new_confidence)?;

        // If confidence too low, invalidate
        if new_confidence < 0.5 {
            self.invalidate(fact_id, "user_challenge")?;
        }

        Ok(())
    }

    /// Supersede a fact with a new version
    pub fn supersede(
        &mut self,
        old_id: &FactId,
        new_content: FactContent,
        new_evidence: Vec<Evidence>,
        reason: &str,
    ) -> Result<FactId> {
        // Generate new fact ID (version increment)
        let new_id = FactId::from_hash(old_id.hash(), old_id.version() + 1);

        // Store new fact
        let new_fact_entity_id = self.store_fact(
            new_content,
            FactType::StableFact, // Inherit type
            new_evidence,
            vec![old_id.clone()], // depends on old
            1.0, // Full confidence for new version
        )?;

        // Create "supersedes" edge from new to old
        let old_entity_id = self.find_entity_by_name(&old_id.to_string())?;

        let edge = GraphEdge {
            id: 0,
            from_id: new_fact_entity_id,
            to_id: old_entity_id,
            edge_type: "supersedes".to_string(),
            data: json!({
                "reason": reason,
                "timestamp": SystemTime::now()
                    .duration_since(UNIX_EPOCH)?
                    .as_secs(),
            }),
        };

        self.graph.insert_edge(&edge)?;

        // Update old fact's data to note it's superseded
        let updated_data = json!({
            "superseded_by": new_id.to_string(),
            "active": false
        });

        self.graph.update_entity_data(old_entity_id, updated_data)?;

        Ok(new_id)
    }
}
```

### 6. Query API

```rust
#[derive(Debug, Clone)]
pub struct CacheQuery {
    /// Semantic hash of what's being asked
    pub query_hash: [u8; 32],

    /// Required minimum confidence
    pub min_confidence: f64,

    /// Maximum age of fact
    pub max_age: Option<Duration>,

    /// Required fact type
    pub required_type: Option<FactType>,

    /// Context tags for filtering
    pub context_tags: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub enum CacheLookup {
    Hit {
        fact: Fact,
        provenance: ProvenanceChain,
    },
    Miss,
    Stale {
        reason: StaleReason,
    },
}

#[derive(Debug, Clone)]
pub enum StaleReason {
    LowConfidence { current: f64, required: f64 },
    Expired { age: Duration, max_age: Duration },
    EvidenceChanged { changed_evidence: Vec<EvidenceId> },
    Superseded { by: FactId },
}

#[derive(Debug, Clone)]
pub struct ProvenanceChain {
    pub hops: Vec<ProvenanceHop>,
}

#[derive(Debug, Clone)]
pub struct ProvenanceHop {
    pub fact_id: FactId,
    pub content: FactContent,
    pub depth: u32,
}
```

---

## Integration with ROCmForge

### Wrapper Pattern

```rust
/// ROCmForge inference engine with knowledge caching
pub struct CachedInferenceEngine<E> {
    inner: E,
    cache: Arc<Mutex<KnowledgeCache>>,
    enabled: AtomicBool,
}

impl<E: InferenceEngine> CachedInferenceEngine<E> {
    pub async fn infer(&self, query: &Query) -> Result<Response> {
        // Skip cache if disabled
        if !self.enabled.load(Ordering::Relaxed) {
            return self.inner.infer(query).await;
        }

        // 1. Check cache
        let cache_key = self.normalize_query(query)?;

        let mut cache = self.cache.lock().await;
        match cache.lookup(&cache_key) {
            Ok(CacheLookup::Hit { fact, provenance }) => {
                // Return cached response
                return Ok(Response {
                    content: fact.content.to_response(),
                    source: ResponseSource::Cache {
                        fact_id: fact.id.clone(),
                        confidence: fact.confidence,
                        hops: provenance.hops.len(),
                    },
                    gpu_compute_ms: 0,
                });
            }
            Ok(CacheLookup::Miss) => {
                // Fall through to inference
            }
            Ok(CacheLookup::Stale { .. }) => {
                // Log and fall through
                tracing::debug!("Stale cache hit, running inference");
            }
            Err(e) => {
                tracing::warn!("Cache lookup failed: {}", e);
            }
        }

        // 2. Run GPU inference
        let start = Instant::now();
        let response = self.inner.infer(query).await?;
        let gpu_time = start.elapsed();

        // 3. Extract cacheable facts from response
        if let Some(facts) = self.extract_facts(query, &response)? {
            for fact in facts {
                cache.store_fact(
                    fact.content,
                    fact.fact_type,
                    fact.evidence,
                    fact.dependencies,
                    fact.confidence,
                )?;
            }
        }

        Ok(response)
    }

    /// Extract cacheable facts from inference response
    fn extract_facts(
        &self,
        query: &Query,
        response: &Response,
    ) -> Result<Option<Vec<ExtractedFact>>> {
        // This uses pattern matching to identify stable conclusions
        // Example: "This function returns a Result<T, E>"
        // -> becomes a Fact about the function's signature

        let facts = match query {
            Query::CodeAnalysis { .. } => self.extract_code_facts(response)?,
            Query::ApiContract { .. } => self.extract_api_facts(response)?,
            _ => return Ok(None),
        };

        Ok(if facts.is_empty() { None } else { Some(facts) })
    }
}
```

---

## Usage Example

### CLI Administration

```bash
# Check cache status
$ rocmforge cache status

Knowledge Cache Status
======================
Enabled: false
Total Facts: 1,247
Cache Hit Rate: 67.3%
GPU Compute Saved: 432.1s
Invalidated Facts: 12
Pending Revalidation: 5

# Look up a fact
$ rocmforge cache lookup "Result<T,E> cannot contain both T and E"

Cache Hit:
  Fact: f64a3b2c1...
  Content: "Rust's Result<T,E> cannot contain both T and E"
  Confidence: 0.95
  Derived: 2026-01-06T10:30:00Z
  Hops: 1

  Provenance:
    [0] Type invariant from Rust language spec (conf: 1.0)

# Show provenance chain
$ rocmforge cache provenance f64a3b2c1

Provenance for: f64a3b2c1...
Content: Rust's Result<T,E> cannot contain both T and E
Confidence: 0.95

Derivation Chain:
  [0] "Rust type system invariant" (conf: 1.0, docs/lang/ref.md:1240)
      └─ "Result enum definition" (conf: 1.0, src/std/result.rs:42)

# Invalidate a fact
$ rocmforge cache invalidate f64a3b2c1 "User reported edge case"
Fact f64a3b2c1 invalidated (reason: User reported edge case)

# Show cache statistics
$ rocmforge cache stats

Cache Statistics:
----------------
Lookups today: 12,453
Hits: 8,372 (67.2%)
Misses: 4,081 (32.8%)
GPU time saved: ~7.2 minutes

Top fact types:
- StableFact: 543
- Invariant: 412
- Conclusion: 292

Pending revalidations: 5
```

### Configuration

```toml
# ~/.config/rocmforge/cache.toml

[cache]
# Feature flag: OFF by default
enabled = false

# Confidence thresholds
min_confidence = 0.7
min_recompute_confidence = 0.5

# Time limits
max_fact_age_days = 365
default_ttl_days = 30

# Decay parameters
decay_rate_per_day = 0.01
access_boost_factor = 0.05

# Storage
database_path = "~/.local/share/rocmforge/knowledge_cache.db"

# Security
max_fact_size_bytes = 1048576  # 1MB
require_evidence_for_confidence_above = 0.8
```

---

## Key Design Principles

### 1. Facts Decay, Not Rot

Facts are not eternal. Confidence decreases over time unless reinforced by access or new evidence.

### 2. Newer Evidence Trumps Older

When evidence changes, dependent facts are invalidated. Newer versions supersede older ones.

### 3. Provenance is Inspectable

Every fact has a derivation chain showing "why do we believe this?" Users can trace reasoning.

### 4. Challenges are Recorded

User challenges are tracked. Too many challenges → fact invalidated. High confidence → withstands challenges.

### 5. OFF by Default

This is a feature flag. Must be explicitly enabled. Auditable state at all times.

---

## What to Cache (and What Not To)

### ✅ Cache These:

| Type | Example | Confidence |
|------|---------|------------|
| **Language Invariants** | "Result<T,E> cannot contain both T and E" | High |
| **API Contracts** | "Vec::push returns usize" | High |
| **Code Analysis** | "This function has a race condition on line 42" | Medium |
| **Documentation Facts** | "Python 3.12+ uses tomllib for TOML" | High |
| **Framework Behavior** | "tokio::spawn is async" | Medium |

### ❌ Don't Cache These:

| Type | Reason |
|------|--------|
| Conversational context | Ephemeral, user-specific |
| User preferences | Must respect current session |
| Ephemeral session state | Not reusable |
| Real-time data (market prices, etc.) | Stale immediately |
| Subjective opinions | Not factual |

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

- [ ] Set up sqlitegraph database schema
- [ ] Implement Fact/Evidence data models
- [ ] Basic store_fact() lookup()
- [ ] CLI: cache status, cache lookup

### Phase 2: Provenance (Week 2)

- [ ] Derivation edge tracking
- [ ] BFS-based provenance chain building
- [ ] CLI: cache provenance command

### Phase 3: Decay & Invalidation (Week 3)

- [ ] Confidence decay calculation
- [ ] Evidence change detection
- [ ] Cascading invalidation
- [ ] CLI: cache invalidate, cache challenge

### Phase 4: Integration (Week 4)

- [ ] CachedInferenceEngine wrapper
- [ ] Fact extraction from responses
- [ ] Cache instrumentation and metrics
- [ ] Enable as feature flag

### Phase 5: Hardening (Week 5)

- [ ] Security policies and validation
- [ ] Audit trail logging
- [ ] Performance optimization
- [ ] Comprehensive testing

---

## Summary

SQLiteGraph provides an ideal foundation for knowledge-level caching:

- **Graph Structure**: Facts as entities, derivations as edges
- **Provenance**: BFS/k-hop traversals show derivation chains
- **Versioning**: Snapshot export/import, supersession edges
- **Query**: Pattern matching for semantic lookup
- **Persistence**: SQLite backend with ACID guarantees

**This is systems thinking, not AI hype.** We're treating GPU reasoning as a capital investment—compute once, reuse many times, with structured invalidation when reality changes.

---

**Sources**:
- SQLiteGraph codebase analysis
- Knowledge caching design patterns
- ROCmForge architecture
