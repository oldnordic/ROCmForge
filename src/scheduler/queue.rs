//! Request queue management for continuous batching

/// Statistics about the scheduler's request queues
///
/// Provides a snapshot of the number of requests in each state
/// (pending, processing, completed) at a point in time.
#[derive(Debug, Clone)]
pub struct QueueStats {
    pub pending_requests: usize,
    pub processing_requests: usize,
    pub completed_requests: usize,
}

impl QueueStats {
    /// Total number of requests tracked (pending + processing + completed)
    pub fn total_requests(&self) -> usize {
        self.pending_requests + self.processing_requests + self.completed_requests
    }

    /// Number of active requests (pending + processing)
    pub fn active_requests(&self) -> usize {
        self.pending_requests + self.processing_requests
    }

    /// Check if all queues are empty
    pub fn is_empty(&self) -> bool {
        self.total_requests() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_stats_empty() {
        let stats = QueueStats {
            pending_requests: 0,
            processing_requests: 0,
            completed_requests: 0,
        };

        assert!(stats.is_empty());
        assert_eq!(stats.total_requests(), 0);
        assert_eq!(stats.active_requests(), 0);
    }

    #[test]
    fn test_queue_stats_with_requests() {
        let stats = QueueStats {
            pending_requests: 5,
            processing_requests: 3,
            completed_requests: 10,
        };

        assert!(!stats.is_empty());
        assert_eq!(stats.total_requests(), 18);
        assert_eq!(stats.active_requests(), 8);
    }

    #[test]
    fn test_queue_stats_only_completed() {
        let stats = QueueStats {
            pending_requests: 0,
            processing_requests: 0,
            completed_requests: 5,
        };

        assert!(!stats.is_empty());
        assert_eq!(stats.total_requests(), 5);
        assert_eq!(stats.active_requests(), 0);
    }
}
