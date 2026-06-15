//! Trainable linear value head for branch ranking.
//!
//! This is the open-weight experiment in Step 8: a tiny set of differentiable
//! weights sits on top of the frozen 0.5B base model. The head maps a final
//! hidden-state vector to a scalar quality score and is trained on preference
//! pairs derived from traced branch scores.

/// A single training example: hidden-state representation of one branch plus
/// its ground-truth quality score.
#[derive(Debug, Clone)]
pub struct BranchValueExample {
    pub trace_id: String,
    pub timestamp: u64,
    pub hidden: Vec<f32>,
    pub score: f32,
}

/// Linear value head trained with SGD on pairwise hinge loss.
///
/// Input hidden states are standardized using per-dimension mean and standard
/// deviation computed from the training set. Standardization keeps the small
/// SGD updates numerically stable across the natural scale of model
/// activations.
#[derive(Debug, Clone)]
pub struct BranchValueHead {
    weights: Vec<f32>,
    bias: f32,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl BranchValueHead {
    /// Create a new value head for hidden vectors of size `hidden_size`.
    pub fn new(hidden_size: usize) -> Self {
        Self {
            weights: vec![0.0f32; hidden_size],
            bias: 0.0f32,
            mean: vec![0.0f32; hidden_size],
            std: vec![1.0f32; hidden_size],
        }
    }

    /// Return the hidden-size expected by this head.
    pub fn hidden_size(&self) -> usize {
        self.weights.len()
    }

    /// Predict a scalar quality score for `hidden`.
    pub fn predict(&self, hidden: &[f32]) -> f32 {
        if hidden.len() != self.weights.len() {
            return 0.0f32;
        }
        let mut acc = self.bias;
        for (i, &x) in hidden.iter().enumerate() {
            let norm = (x - self.mean[i]) / self.std[i];
            acc += self.weights[i] * norm;
        }
        acc
    }

    /// Train the head on `examples` using pairwise hinge loss.
    ///
    /// For every ordered pair `(worse, better)` where `worse.score <
    /// better.score`, if `f(worse) - f(better) + margin > 0` the weights are
    /// moved so that `f(better)` grows relative to `f(worse)`.
    pub fn fit(&mut self, examples: &[BranchValueExample], epochs: usize, lr: f32, margin: f32) {
        if examples.is_empty() || self.weights.is_empty() {
            return;
        }
        self.compute_statistics(examples);

        let hidden_size = self.weights.len();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for (i, a) in examples.iter().enumerate() {
            for (j, b) in examples.iter().enumerate() {
                if i != j && a.score < b.score {
                    pairs.push((i, j));
                }
            }
        }
        if pairs.is_empty() {
            return;
        }

        for _ in 0..epochs {
            for &(worse_idx, better_idx) in &pairs {
                let worse = normalize(&examples[worse_idx].hidden, &self.mean, &self.std);
                let better = normalize(&examples[better_idx].hidden, &self.mean, &self.std);
                let score_worse = dot(&self.weights, &worse) + self.bias;
                let score_better = dot(&self.weights, &better) + self.bias;
                let violation = score_worse - score_better + margin;
                if violation > 0.0f32 {
                    // Gradient: dL/dw = x_worse - x_better, dL/db = 0.
                    for k in 0..hidden_size {
                        self.weights[k] += lr * (better[k] - worse[k]);
                    }
                }
            }
        }
    }

    /// Train the head on `examples` using MSE against normalized scores.
    ///
    /// This is often easier than pairwise hinge when the prompt already
    /// contains the numeric score: the head only has to map the model's
    /// representation of the prompt back to that score.
    pub fn fit_mse(&mut self, examples: &[BranchValueExample], epochs: usize, lr: f32) {
        if examples.is_empty() || self.weights.is_empty() {
            return;
        }
        self.compute_statistics(examples);

        let hidden_size = self.weights.len();
        let n = examples.len() as f32;
        let min_score = examples
            .iter()
            .map(|ex| ex.score)
            .fold(f32::INFINITY, f32::min);
        let max_score = examples
            .iter()
            .map(|ex| ex.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let score_range = (max_score - min_score).max(1e-6f32);

        for _ in 0..epochs {
            for ex in examples {
                let x = normalize(&ex.hidden, &self.mean, &self.std);
                let target = (ex.score - min_score) / score_range;
                let pred = dot(&self.weights, &x) + self.bias;
                let err = pred - target;
                for (k, &xk) in x.iter().enumerate().take(hidden_size) {
                    self.weights[k] -= lr * 2.0f32 * err * xk / n;
                }
                self.bias -= lr * 2.0f32 * err / n;
            }
        }
    }

    fn compute_statistics(&mut self, examples: &[BranchValueExample]) {
        let hidden_size = self.weights.len();
        self.mean.fill(0.0f32);
        self.std.fill(0.0f32);
        for ex in examples {
            for (i, &x) in ex.hidden.iter().enumerate().take(hidden_size) {
                self.mean[i] += x;
            }
        }
        let n = examples.len() as f32;
        for m in &mut self.mean {
            *m /= n;
        }
        for ex in examples {
            for (i, &x) in ex.hidden.iter().enumerate().take(hidden_size) {
                let diff = x - self.mean[i];
                self.std[i] += diff * diff;
            }
        }
        for s in &mut self.std {
            *s = (*s / n).sqrt().max(1e-6f32);
        }
    }
}

fn normalize(hidden: &[f32], mean: &[f32], std: &[f32]) -> Vec<f32> {
    hidden
        .iter()
        .enumerate()
        .map(|(i, &x)| (x - mean[i]) / std[i])
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}
