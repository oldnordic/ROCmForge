#![cfg(feature = "cpu-graph")]
//! Step 2.5 validation experiment: structured-recurrence search vs linear CoT.
//!
//! Uses the CPU graph engine (not the 0.5B model) to solve small grid mazes.
//! Each forward step is a deterministic one-hot state transition implemented as
//! a `CpuOpNode::Gemv`.  The structured arm forks branches with
//! `CaptureContext::regress_to()` and shares prefixes; the linear baseline gets
//! the same forward-op budget and tries random action sequences from scratch.
//!
//! This is the cheapest go/no-go gate for the introspection ladder: if
//! branching + rollback does not beat a linear chain under equal compute, the
//! later steps should not be built.

use fastrand::Rng;
use rocmforge::cpu::graph::{
    CpuGraph, CpuGraphArena, CpuOpNode, F32Handle, PersistentSnapshot, Shelf, TemporalWindow,
    U8Handle,
};
use rocmforge::loader::GgmlType;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Action {
    Right,
    Down,
    Left,
    Up,
}

const ACTIONS: [Action; 4] = [Action::Right, Action::Down, Action::Left, Action::Up];

struct Maze {
    rows: usize,
    cols: usize,
    walls: HashSet<usize>,
    start: usize,
    goal: usize,
}

impl Maze {
    fn cell_index(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }

    fn coord(&self, idx: usize) -> (usize, usize) {
        (idx / self.cols, idx % self.cols)
    }

    fn next(&self, pos: usize, action: Action) -> usize {
        let (r, c) = self.coord(pos);
        let (nr, nc) = match action {
            Action::Right if c + 1 < self.cols => (r, c + 1),
            Action::Down if r + 1 < self.rows => (r + 1, c),
            Action::Left if c > 0 => (r, c - 1),
            Action::Up if r > 0 => (r - 1, c),
            _ => (r, c),
        };
        let nidx = self.cell_index(nr, nc);
        if self.walls.contains(&nidx) {
            pos
        } else {
            nidx
        }
    }
}

fn random_connected_maze(rng: &mut Rng, rows: usize, cols: usize) -> Maze {
    let cells = rows * cols;
    let mut walls = HashSet::new();
    // Randomly block ~20% of cells (excluding start and goal), then ensure
    // the goal is still reachable by carving a simple path.
    for i in 1..cells - 1 {
        if rng.f32() < 0.20 {
            walls.insert(i);
        }
    }
    let mut maze = Maze {
        rows,
        cols,
        walls,
        start: 0,
        goal: cells - 1,
    };

    // Carve a guaranteed simple path from start to goal so DFS always has a
    // solution.  Walk toward the goal using only Right/Down; if the chosen
    // cell happens to be a wall, remove it and continue.
    let mut pos = maze.start;
    while pos != maze.goal {
        let (r, c) = maze.coord(pos);
        let can_right = c + 1 < cols;
        let can_down = r + 1 < rows;
        // Prefer the axis that still has progress to make; randomize ties.
        let action = if can_right && can_down {
            if rng.bool() {
                Action::Right
            } else {
                Action::Down
            }
        } else if can_right {
            Action::Right
        } else if can_down {
            Action::Down
        } else {
            break;
        };
        let (nr, nc) = match action {
            Action::Right => (r, c + 1),
            Action::Down => (r + 1, c),
            _ => (r, c),
        };
        let nxt = maze.cell_index(nr, nc);
        maze.walls.remove(&nxt);
        pos = nxt;
    }
    maze
}

fn build_transition_matrix(rows: usize, cols: usize, action: Action, maze: &Maze) -> Vec<f32> {
    let n = rows * cols;
    // Row-major: T[next][current]
    let mut t = vec![0.0f32; n * n];
    for c in 0..n {
        let nxt = maze.next(c, action);
        t[nxt * n + c] = 1.0;
    }
    t
}

fn f32_slice_to_u8(src: &[f32]) -> Vec<u8> {
    src.iter().copied().flat_map(f32::to_le_bytes).collect()
}

struct MazeGraphRunner {
    graph: CpuGraph,
    arena: CpuGraphArena,
    cells: usize,
    state_handle: F32Handle,
    action_weights: [U8Handle; 4],
    snapshots: std::collections::HashMap<u64, PersistentSnapshot>,
}

impl MazeGraphRunner {
    fn new(maze: &Maze) -> Self {
        let cells = maze.rows * maze.cols;
        let mut arena = CpuGraphArena::new();
        let mut state = vec![0.0f32; cells];
        state[maze.start] = 1.0;
        let state_handle = arena.copy_f32(Shelf::Persistent, &state);

        let graph = CpuGraph::new();

        let action_weights = ACTIONS.map(|action| {
            let mat = build_transition_matrix(maze.rows, maze.cols, action, maze);
            arena.copy_u8(Shelf::Constants, &f32_slice_to_u8(&mat))
        });

        Self {
            graph,
            arena,
            cells,
            state_handle,
            action_weights,
            snapshots: std::collections::HashMap::new(),
        }
    }

    fn apply_action(&mut self, timestamp: u64, action: Action) -> F32Handle {
        let out = self.arena.alloc_f32(Shelf::Ephemeral, self.cells);
        let action_idx = ACTIONS
            .iter()
            .position(|&a| a == action)
            .expect("action must be one of ACTIONS");
        let node = CpuOpNode::Gemv {
            weight: self.action_weights[action_idx],
            weight_bytes: self.cells * self.cells * 4,
            input: self.state_handle,
            out,
            scratch: None,
            m: self.cells,
            n: self.cells,
            wtype_code: GgmlType::F32 as u32,
            needs_transpose: false,
        };
        self.graph.add_node(node, 0, 0, timestamp);
        out
    }

    fn commit_ephemeral_state(&mut self, out: F32Handle) {
        // The Gemv wrote into an ephemeral output; copy it back into the
        // persistent state slot so snapshots remain valid at offset 0.
        let src = self.arena.f32(out).to_vec();
        self.arena.f32_mut(self.state_handle).copy_from_slice(&src);
    }

    fn snapshot(&mut self, timestamp: u64) {
        self.snapshots
            .insert(timestamp, self.arena.snapshot_persistent());
    }

    fn regress_to(&mut self, timestamp: u64) {
        self.graph.regress(timestamp);
        if let Some(snap) = self.snapshots.get(&timestamp).cloned() {
            self.arena.restore_persistent(&snap);
        }
    }

    fn execute_at(&mut self, timestamp: u64) {
        self.graph
            .execute_window(
                &mut self.arena,
                TemporalWindow {
                    start: timestamp,
                    end: timestamp + 1,
                },
            )
            .expect("maze replay failed");
    }
}

fn dfs_search(
    runner: &mut MazeGraphRunner,
    maze: &Maze,
    pos: usize,
    depth: usize,
    max_depth: usize,
    nodes: &mut usize,
    timestamp: u64,
) -> Option<Vec<Action>> {
    if pos == maze.goal {
        return Some(Vec::new());
    }
    if depth == max_depth {
        return None;
    }
    runner.snapshot(timestamp);
    for action in ACTIONS {
        let out = runner.apply_action(timestamp + 1, action);
        runner.execute_at(timestamp + 1);
        runner.commit_ephemeral_state(out);
        *nodes += 1;
        let next = maze.next(pos, action);
        if let Some(mut suffix) = dfs_search(
            runner,
            maze,
            next,
            depth + 1,
            max_depth,
            nodes,
            timestamp + 1,
        ) {
            suffix.insert(0, action);
            return Some(suffix);
        }
        runner.regress_to(timestamp);
    }
    None
}

fn linear_random_search(
    maze: &Maze,
    max_depth: usize,
    budget: usize,
    rng: &mut Rng,
) -> (bool, usize) {
    let cells = maze.rows * maze.cols;
    let mut nodes = 0;
    for _ in 0..budget {
        let mut state = vec![0.0f32; cells];
        state[maze.start] = 1.0;
        for _ in 0..max_depth {
            if nodes >= budget {
                break;
            }
            let action = ACTIONS[rng.usize(..ACTIONS.len())];
            let pos = state
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.partial_cmp(b.1)
                        .expect("state probabilities are comparable floats")
                })
                .map(|(i, _)| i)
                .unwrap_or(maze.start);
            let nxt = maze.next(pos, action);
            state.fill(0.0);
            state[nxt] = 1.0;
            nodes += 1;
            if nxt == maze.goal {
                return (true, nodes);
            }
        }
    }
    (false, nodes)
}

#[test]
fn test_cpu_graph_search_vs_linear_baseline() {
    let mut rng = Rng::with_seed(42);
    const TRIALS: usize = 16;
    const ROWS: usize = 3;
    const COLS: usize = 3;
    const MAX_DEPTH: usize = 6;

    let mut search_solved = 0usize;
    let mut search_nodes = 0usize;
    let mut baseline_solved = 0usize;
    let mut baseline_nodes = 0usize;

    for _ in 0..TRIALS {
        let maze = random_connected_maze(&mut rng, ROWS, COLS);
        let mut runner = MazeGraphRunner::new(&maze);
        let mut nodes = 0;
        let path = dfs_search(&mut runner, &maze, maze.start, 0, MAX_DEPTH, &mut nodes, 0);
        assert!(path.is_some(), "DFS search failed to find a reachable goal");
        search_solved += 1;
        search_nodes += nodes;

        let (solved, used) = linear_random_search(&maze, MAX_DEPTH, nodes, &mut rng);
        if solved {
            baseline_solved += 1;
        }
        baseline_nodes += used;
    }

    let search_rate = search_solved as f32 / TRIALS as f32;
    let baseline_rate = baseline_solved as f32 / TRIALS as f32;
    let search_norm = search_rate / search_nodes.max(1) as f32;
    let baseline_norm = baseline_rate / baseline_nodes.max(1) as f32;

    println!(
        "Trials: {}  maze: {}x{}  max_depth: {}",
        TRIALS, ROWS, COLS, MAX_DEPTH
    );
    println!(
        "Search     solved: {}/{}  avg_nodes: {:.2}  norm_acc: {:.6}",
        search_solved,
        TRIALS,
        search_nodes as f32 / TRIALS as f32,
        search_norm
    );
    println!(
        "Linear     solved: {}/{}  avg_nodes: {:.2}  norm_acc: {:.6}",
        baseline_solved,
        TRIALS,
        baseline_nodes as f32 / TRIALS as f32,
        baseline_norm
    );

    assert!(
        search_norm > baseline_norm,
        "Structured-recurrence search must beat linear random baseline in compute-normalized accuracy: search={:.6} baseline={:.6}",
        search_norm,
        baseline_norm
    );
}

#[test]
fn test_dfs_simple_2x2() {
    let maze = Maze {
        rows: 2,
        cols: 2,
        walls: HashSet::new(),
        start: 0,
        goal: 3,
    };
    let mut runner = MazeGraphRunner::new(&maze);
    let mut nodes = 0;
    let path = dfs_search(&mut runner, &maze, maze.start, 0, 4, &mut nodes, 0);
    println!("path: {:?}  nodes: {}", path, nodes);
    assert!(path.is_some());
}
