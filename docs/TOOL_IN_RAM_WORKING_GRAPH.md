# Tool Proposal: In-RAM Working Graph for Magellan

**Date**: 2026-01-13
**Status**: Design Proposal
**Related**: Magellan, SQLiteGraph, Law-Based Governance

---

## Problem Statement

Magellan + SQLiteGraph provide **truth**, but are optimized for:
- Correctness
- Durability
- Replay
- Audit

Not optimized for:
- Sub-millisecond traversal
- Repeated fan-in/fan-out queries
- Speculative analysis by sub-agents
- "What breaks if I touch this?" loops

This makes certain operations slow:
- Law discovery (requires repeated graph walks)
- Impact analysis (O(seconds) per query)
- Drift detection before edits
- Sub-agent exploration

---

## Proposal: In-RAM Working Graph

Add a **read-through in-RAM graph cache** that sits between SQLiteGraph and tools:

```
┌─────────────────────────────────────────┐
│         Tools / LLM / Sub-agents        │
└─────────────────┬───────────────────────┘
                  │ fast queries
                  ▼
┌─────────────────────────────────────────┐
│      In-RAM Working Graph (Cache)       │
│  • Symbol IDs only (no raw text)        │
│  • Fan-in/fan-out edges                 │
│  • Law tags                             │
│  • Hot metadata                         │
└─────────────────┬───────────────────────┘
                  │ load / invalidate
                  ▼
┌─────────────────────────────────────────┐
│        SQLiteGraph (Authority)          │
│  • Full symbol data                     │
│  • Edges with byte spans                │
│  • Versioned, auditable                 │
└─────────────────────────────────────────┘
```

---

## Data Structure

```rust
use slotmap::SlotMap;

/// Compact symbol node for RAM graph
pub struct RamSymbol {
    /// Original SymbolId from Magellan
    pub id: SymbolId,

    /// Symbol kind (function, struct, etc.)
    pub kind: SymbolKind,

    /// File location
    pub file: FileId,

    /// Module path
    pub module: ModuleId,

    /// Associated laws (from law-based governance)
    pub law_tags: Vec<LawId>,

    /// Precomputed fan-in (how many call this)
    pub fan_in: usize,

    /// Precomputed fan-out (how many this calls)
    pub fan_out: usize,
}

/// Edge between symbols
pub struct RamEdge {
    pub from: SymbolId,
    pub to: SymbolId,
    pub edge_type: EdgeType,  // CALLS, USES, OWNS, REFERENCES
}

/// The RAM graph itself
pub struct RamWorkingGraph {
    /// All symbols (dense, cache-friendly)
    pub symbols: SlotMap<SymbolId, RamSymbol>,

    /// Adjacency list for fast traversal
    pub edges_out: Vec<Vec<EdgeId>>,  // outgoing edges per symbol
    pub edges_in: Vec<Vec<EdgeId>>,   // incoming edges per symbol

    /// Generation counter for invalidation
    pub generation: u64,

    /// What we loaded from
    pub source_generation: u64,
}
```

---

## API Design

```rust
impl RamWorkingGraph {
    /// Load from SQLiteGraph at current generation
    pub fn load_from_sqlite(
        sqlite: &SQLiteGraph,
        scope: &LoadScope,  // what to load
    ) -> HipResult<Self>;

    /// Get symbol with O(1) lookup
    pub fn get_symbol(&self, id: SymbolId) -> Option<&RamSymbol>;

    /// Get all callers (fan-in) - O(k) where k = fan_in
    pub fn get_callers(&self, id: SymbolId) -> Vec<SymbolId>;

    /// Get all callees (fan-out) - O(k) where k = fan_out
    pub fn get_callees(&self, id: SymbolId) -> Vec<SymbolId>;

    /// BFS traversal up to N hops - O(hops × avg_degree)
    pub fn traverse_bfs(
        &self,
        start: SymbolId,
        direction: TraversalDir,
        max_hops: usize,
    ) -> Vec<SymbolId>;

    /// Check if stale
    pub fn is_stale(&self, current_generation: u64) -> bool;

    /// Invalidate and reload
    pub fn refresh(&mut self, sqlite: &SQLiteGraph) -> HipResult<()>;
}
```

---

## Invalidation Strategy

| Event | Action |
|-------|--------|
| File saved | Increment generation |
| Splice applied | Increment generation |
| Index refreshed | Increment generation |
| Query starts | Check `is_stale()`, refresh if needed |

This ensures:
- RAM graph is always fresh enough
- No complex per-symbol invalidation
- Simple generation counter semantics

---

## What to Load

**Full graph** for small projects (< 10K symbols):
- Load everything at startup
- Fast queries forever

**Hot subgraph** for large projects:
- Load files currently being edited
- Load their transitive dependencies
- Load high-centrality symbols (high fan-in)

```rust
pub enum LoadScope {
    Full,
    HotSubgraph {
        files: Vec<FileId>,
        dependency_depth: usize,
        include_central: bool,  // symbols with high fan-in
    }
}
```

---

## Integration with Tools

### For Magellan

```rust
impl MagellanIndex {
    pub fn ram_graph(&self) -> &RamWorkingGraph {
        &self.ram_cache
    }

    pub fn with_fresh_ram<F, R>(&self, f: F) -> HipResult<R>
    where
        F: FnOnce(&RamWorkingGraph) -> HipResult<R>,
    {
        let ram = self.ram_cache.refresh_if_needed(&self.sqlite)?;
        f(&ram)
    }
}
```

### For Splice

```rust
impl Splice {
    /// Before splice: check impact using RAM graph
    pub fn check_impact(&self, op: &Operation) -> ImpactAnalysis {
        let ram = self.magellan.ram_graph()?;

        // Fast impact analysis
        let affected = ram.traverse_bfs(op.target, Up, 3)?;

        // Check law violations
        let violations = affected
            .iter()
            .filter(|s| ram.get_symbol(*s)?.law_tags.contains(&op.law))
            .collect();

        ImpactAnalysis { affected, violations }
    }
}
```

### For LLM

```rust
/// Tool: "What symbols are central in this module?"
pub fn find_central_symbols(ram: &RamWorkingGraph, module: ModuleId) -> Vec<SymbolId> {
    ram.symbols
        .iter()
        .filter(|s| s.module == module)
        .filter(|s| s.fan_in > 10)  // high fan-in = central
        .map(|s| s.id)
        .collect()
}

/// Tool: "What breaks if I touch this?"
pub fn what_breaks_if_i_touch(ram: &RamWorkingGraph, id: SymbolId) -> ImpactInfo {
    let callers = ram.get_callers(id);
    let callers_of_callers = callers
        .iter()
        .flat_map(|c| ram.get_callers(*c))
        .collect::<HashSet<_>>();

    ImpactInfo {
        direct: callers.len(),
        transitive: callers_of_callers.len(),
        has_laws: callers.iter().any(|c| !ram.get_symbol(*c)?.law_tags.is_empty()),
    }
}
```

---

## Performance Expectations

| Operation | SQLiteGraph | RAM Graph | Speedup |
|-----------|-------------|-----------|---------|
| Get symbol | ~1ms (disk round-trip) | ~1μs (memory) | 1000× |
| Get callers | ~10-100ms (query) | ~10μs (vec read) | 1000× |
| BFS 3 hops | ~100-1000ms | ~100μs | 1000× |
| Impact analysis | ~1s | ~1ms | 1000× |

These are rough estimates. Actual speedup depends on:
- Graph size
- Cache locality
- Query patterns

---

## Memory Estimates

For a medium codebase (10K symbols, 50K edges):

| Component | Size |
|-----------|------|
| Symbol nodes (10K × ~64 bytes) | ~640 KB |
| Edge lists (50K × 8 bytes) | ~400 KB |
| Index overhead | ~200 KB |
| **Total** | **~1.2 MB** |

Trivial compared to:
- Semantic embeddings (hundreds of MB)
- Source cache (tens of MB)

---

## Implementation Phases

### Phase 1: Basic RAM Graph
- [ ] `RamSymbol`, `RamEdge` structs
- [ ] `RamWorkingGraph` container
- [ ] Load from SQLiteGraph (full scope)
- [ ] Basic queries: get, callers, callees

### Phase 2: Traversal
- [ ] BFS/DFS traversals
- [ ] k-hop fan-in/out
- [ ] Path finding

### Phase 3: Invalidation
- [ ] Generation counters
- [ ] Stale detection
- [ ] Refresh on demand

### Phase 4: Hot Subgraph
- [ ] `LoadScope::HotSubgraph`
- [ ] Dependency analysis
- [ ] Centrality scoring

### Phase 5: Tool Integration
- [ ] Magellan API
- [ ] Splice impact checking
- [ ] LLM query tools

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Stale cache causing wrong answers | Generation counters, refresh on query |
| Memory growth | Hot subgraph loading, not full graph |
| Cache invalidation complexity | Use generations, not per-symbol tracking |
| Dual source of truth | RAM is explicitly a cache, SQLite is authority |

---

## Alternatives Considered

### petgraph
- Good for algorithms
- No persistence, no schema
- Would replace SQLiteGraph entirely ❌

### indradb
- Key-value backed, heavy
- Overkill for this use case ❌

### DIY over slotmap (chosen)
- Deterministic IDs
- Cache-friendly
- Full control
- Fits "no guessing" philosophy ✅

---

## Status

📋 **Design Phase** - Ready for implementation when needed.

This is a **performance optimization**, not a correctness fix. The current SQLiteGraph-based approach works correctly; this would make certain operations faster.

The **law-based governance** system can work without this, but it becomes more practical with fast graph queries.
