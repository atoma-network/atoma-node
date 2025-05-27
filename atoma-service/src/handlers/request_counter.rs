use dashmap::{DashMap, Entry};

#[derive(Clone, Debug)]
pub struct RequestCounter {
    running_num_requests: DashMap<String, usize>,
}

impl Default for RequestCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestCounter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            running_num_requests: DashMap::new(),
        }
    }

    pub fn increment(&self, key: &str) {
        let mut entry = self
            .running_num_requests
            .entry(key.to_string())
            .or_insert(0);
        *entry += 1;
    }

    pub fn decrement(&self, key: &str) {
        match self.running_num_requests.entry(key.to_string()) {
            Entry::Occupied(mut entry) => {
                let count = entry.get_mut();
                *count -= 1;
                if *count == 0 {
                    entry.remove();
                }
            }
            Entry::Vacant(_) => {
                // This should not happen, but just in case, we remove the entry
                self.running_num_requests.remove(key);
            }
        }
    }

    #[must_use]
    pub fn get_count(&self, key: &str) -> usize {
        self.running_num_requests.get(key).map_or(0, |entry| *entry)
    }
}
