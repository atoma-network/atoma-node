use atoma_p2p::metrics::RUNNING_REQUESTS;
use dashmap::{DashMap, Entry};
use opentelemetry::KeyValue;
use tracing::error;

/// A thread-safe request counter that tracks the number of requests being processed for each inference service.
#[derive(Clone, Debug)]
pub struct RequestCounter {
    /// A map that holds the count of running requests for each inference service.
    running_num_requests: DashMap<String, usize>,
}

impl Default for RequestCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestCounter {
    /// Creates a new instance of `RequestCounter`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            running_num_requests: DashMap::new(),
        }
    }

    /// Increments the count for the given key or initializes it to 1 if it does not exist.
    pub fn increment(&self, key: &str) {
        let mut entry = self
            .running_num_requests
            .entry(key.to_string())
            .or_insert(0);
        *entry += 1;
        RUNNING_REQUESTS.record(*entry as u64, &[KeyValue::new("service", key.to_string())]);
    }

    /// Decrements the count for the given key. If the count reaches zero, the entry is removed.
    pub fn decrement(&self, key: &str) {
        match self.running_num_requests.entry(key.to_string()) {
            Entry::Occupied(mut entry) => {
                let count = entry.get_mut();
                *count -= 1;
                RUNNING_REQUESTS
                    .record(*count as u64, &[KeyValue::new("service", key.to_string())]);
                if *count == 0 {
                    entry.remove();
                }
            }
            Entry::Vacant(_) => {
                // This should not happen
                error!(
                    target = "atoma-service",
                    level = "info",
                    event = "chat-completions-handler",
                    "Attempted to decrement a non-existent key: {}",
                    key
                );
            }
        }
    }

    /// Retrieves the current count for the given key.
    #[must_use]
    pub fn get_count(&self, key: &str) -> usize {
        self.running_num_requests.get(key).map_or(0, |entry| *entry)
    }
}
