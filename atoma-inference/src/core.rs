use tokio::sync::oneshot;

use crate::types::{InferenceRequest, InferenceResponse, ModelRequest, ModelResponse};

pub enum CoreThreadCommand {
    RunInference(InferenceRequest, oneshot::Sender<InferenceResponse>),
    FetchModel(ModelRequest, oneshot::Sender<ModelResponse>),
}
