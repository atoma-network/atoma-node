use atoma_helpers::Supabase;
use atoma_types::{AtomaOutputMetadata, OutputType};
use serde_json::json;
use tracing::{info, instrument};

use crate::AtomaOutputManagerError;

pub struct SupabaseOutputManager {
    supabase: Supabase,
}

impl SupabaseOutputManager {
    /// Constructor
    pub fn new(supabase: Supabase) -> Self {
        Self { supabase }
    }

    /// Handles a new post request. Encapsulates the logic necessary
    /// to post new requests, using `reqwest::Client`.
    #[instrument(skip_all)]
    pub async fn handle_post_request(
        &self,
        output_metadata: &AtomaOutputMetadata,
        output: Vec<u8>,
    ) -> Result<(), AtomaOutputManagerError> {
        match output_metadata.output_type {
            OutputType::Text => {
                let text = String::from_utf8(output)?;
                self.supabase
                    .insert("text_response", &json!({ "transaction_id": output_metadata.output_destination.request_id(),"text": text, "tokens": output_metadata.tokens }))
                    .await?;
                info!("Text response submitted to Supabase");
            }
            OutputType::Image => {
                unimplemented!();
            }
        }
        Ok(())
    }
}
