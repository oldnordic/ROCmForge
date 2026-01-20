//! Core types for the continuous batching scheduler

use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during scheduling operations
#[derive(Error, Debug)]
pub enum SchedulerError {
    #[error("Request not found: {0}")]
    RequestNotFound(u32),
    #[error("Batch size exceeded maximum: {max}, got {actual}")]
    BatchSizeExceeded { max: usize, actual: usize },
    #[error("Invalid request state transition")]
    InvalidStateTransition,
    #[error("Queue capacity exceeded")]
    QueueCapacityExceeded,
}

/// Result type for scheduler operations
pub type SchedulerResult<T> = Result<T, SchedulerError>;

/// State of a generation request in the scheduler
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RequestState {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

/// A single generation request in the scheduler
///
/// Tracks the complete lifecycle of a request from submission through completion,
/// including prompt tokens, generated tokens, sampling parameters, and state transitions.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub request_id: u32,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub state: RequestState,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub generated_tokens: Vec<u32>,
    pub finish_reason: Option<String>,
}

impl GenerationRequest {
    /// Create a new generation request
    pub fn new(
        request_id: u32,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Self {
        GenerationRequest {
            request_id,
            prompt_tokens,
            max_tokens,
            temperature,
            top_k,
            top_p,
            state: RequestState::Pending,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            generated_tokens: Vec::new(),
            finish_reason: None,
        }
    }

    /// Total tokens (prompt + generated) for this request
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    /// Check if the request is complete
    pub fn is_complete(&self) -> bool {
        match self.state {
            RequestState::Completed | RequestState::Failed | RequestState::Cancelled => true,
            _ => self.generated_tokens.len() >= self.max_tokens,
        }
    }

    /// Transition request to processing state
    pub fn start_processing(&mut self) -> SchedulerResult<()> {
        if self.state != RequestState::Pending {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.state = RequestState::Processing;
        self.started_at = Some(Instant::now());
        Ok(())
    }

    /// Transition request to completed state
    pub fn complete(&mut self, reason: Option<String>) -> SchedulerResult<()> {
        if self.state != RequestState::Processing {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.state = RequestState::Completed;
        self.completed_at = Some(Instant::now());
        if let Some(reason) = reason {
            self.finish_reason = Some(reason);
        } else if self.finish_reason.is_none() {
            self.finish_reason = Some("completed".to_string());
        }
        Ok(())
    }

    /// Transition request to failed state
    pub fn fail(&mut self) -> SchedulerResult<()> {
        if self.state != RequestState::Processing {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.state = RequestState::Failed;
        self.completed_at = Some(Instant::now());
        self.finish_reason
            .get_or_insert_with(|| "failed".to_string());
        Ok(())
    }

    /// Cancel the request
    pub fn cancel(&mut self) -> SchedulerResult<()> {
        if matches!(
            self.state,
            RequestState::Completed | RequestState::Failed | RequestState::Cancelled
        ) {
            return Err(SchedulerError::InvalidStateTransition);
        }

        if self.started_at.is_none() {
            self.started_at = Some(Instant::now());
        }
        self.state = RequestState::Cancelled;
        self.completed_at = Some(Instant::now());
        self.finish_reason = Some("cancelled".to_string());
        Ok(())
    }

    /// Add a generated token to this request
    pub fn add_generated_token(&mut self, token: u32) -> SchedulerResult<()> {
        if self.state != RequestState::Processing {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.generated_tokens.push(token);

        if self.generated_tokens.len() >= self.max_tokens {
            self.complete(Some("length".to_string()))?;
        } else if self.is_complete() {
            self.complete(None)?;
        }

        Ok(())
    }

    /// Get the processing duration for this request
    pub fn processing_time(&self) -> Option<Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            (Some(start), None) => Some(Instant::now().duration_since(start)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_request_creation() {
        let request = GenerationRequest::new(1, vec![1, 2, 3], 10, 0.8, 50, 0.9);

        assert_eq!(request.request_id, 1);
        assert_eq!(request.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.state, RequestState::Pending);
        assert_eq!(request.total_tokens(), 3);
        assert!(!request.is_complete());
    }

    #[test]
    #[serial]
    fn test_request_state_transitions() {
        let mut request = GenerationRequest::new(1, vec![1, 2, 3], 10, 0.8, 50, 0.9);

        // Start processing
        assert!(request.start_processing().is_ok());
        assert_eq!(request.state, RequestState::Processing);
        assert!(request.started_at.is_some());

        // Add tokens
        for i in 0..10 {
            assert!(request.add_generated_token(i).is_ok());
        }

        // Should be complete now
        assert_eq!(request.state, RequestState::Completed);
        assert!(request.completed_at.is_some());
        assert!(request.is_complete());
    }
}
