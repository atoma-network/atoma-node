use atoma_helpers::{PromptsRow, Supabase, TokensRow};
use atoma_types::{ChatInferenceRequest, ModelInput};
use tokio::sync::mpsc;
use tracing::instrument;

use crate::AtomaInputManagerError;

const NUMBER_OF_REQUESTS_TO_TRY: usize = 10;
const SLEEP_BETWEEN_REQUESTS_SEC: u64 = 1;

/// `SupabaseInputManager` - Responsible for getting the prompt from the user
pub struct SupabaseInputManager {
    supabase: Supabase,
}

impl SupabaseInputManager {
    /// Constructor
    pub fn new(supabase: Supabase) -> Self {
        Self { supabase }
    }

    /// Handles  a new chat request. Encapsulates the logic necessary
    /// to post new requests, using `reqwest::Client`.
    #[instrument(skip_all)]
    pub async fn handle_chat_request(
        &mut self,
        request_id: String,
    ) -> Result<ModelInput, AtomaInputManagerError> {
        for _ in 0..NUMBER_OF_REQUESTS_TO_TRY {
            let response = self
                .supabase
                .select_single::<PromptsRow>(
                    "prompts",
                    &format!("transaction_id=eq.{request_id}&select=previous_tx,prompt"),
                )
                .await?;
            if let Some(response) = response {
                // If the response is non-empty, we can return the prompt
                let prompt = response.prompt.clone();
                let tokens = match response.previous_tx {
                    Some(previous_tx) => {
                        // There is a previous transaction from which we can get the context tokens
                        let row = self
                            .supabase
                            .select_single::<TokensRow>(
                                "text_response",
                                &format!("transaction_id=eq.{previous_tx}&select=tokens"),
                            )
                            .await?;
                        row.map(|row| row.tokens).unwrap_or_default()
                    }
                    None => vec![],
                };
                return Ok(ModelInput::Chat((prompt, tokens)));
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(SLEEP_BETWEEN_REQUESTS_SEC)).await;
        }
        Err(AtomaInputManagerError::TimeoutError)
    }
}
