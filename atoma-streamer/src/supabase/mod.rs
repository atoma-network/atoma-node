use atoma_helpers::{Supabase, SupabaseError};
use serde_json::json;
use thiserror::Error;

pub struct Streamer {
    supabase: Supabase,
}

impl Streamer {
    pub fn new(supabase: Supabase) -> Self {
        Self { supabase }
    }

    pub async fn handle_streaming_request(
        &mut self,
        request_id: String,
        data: String,
    ) -> Result<(), StreamerError> {
        self.supabase
            .insert(
                "partial",
                &json!({"transaction_id":request_id, "token":data}),
            )
            .await?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum StreamerError {
    #[error("Supabase error: `{0}`")]
    SupabaseError(#[from] SupabaseError),
}
