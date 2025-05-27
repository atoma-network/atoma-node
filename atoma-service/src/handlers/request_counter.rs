use std::{
    collections::HashMap,
    sync::{atomic::AtomicUsize, Arc},
};

/// A thread-safe request counter that tracks the number of requests being processed for each inference service.
#[derive(Clone, Debug)]
pub struct RequestCounter {
    /// A map that keeps track of the number of requests currently being processed for each inference service.
    running_num_requests: HashMap<String, Arc<AtomicUsize>>,
}

impl Default for RequestCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestCounter {
    /// Creates a new `RequestCounter`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            running_num_requests: HashMap::new(),
        }
    }

    /// Increments the request count for the given key.
    pub fn increment(&mut self, key: &str) {
        self.running_num_requests
            .entry(key.to_string())
            .and_modify(|count| {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            })
            .or_insert_with(|| Arc::new(AtomicUsize::new(1)));
    }

    /// Decrements the count for the given key.
    pub fn decrement(&mut self, key: &str) {
        self.running_num_requests
            .entry(key.to_string())
            .and_modify(|count| {
                count.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            });
    }

    /// Returns a reference to the `AtomicUsize` for the given key, creating it if it does not exist.
    #[must_use]
    pub fn get_count(&mut self, key: &str) -> Arc<AtomicUsize> {
        Arc::clone(
            self.running_num_requests
                .entry(key.to_string())
                .or_insert_with(|| Arc::new(AtomicUsize::new(0))),
        )
    }
}
