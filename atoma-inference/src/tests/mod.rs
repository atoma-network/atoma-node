use crate::models::{config::ModelConfig, types::ModelType, ModelError, ModelTrait};
use ed25519_consensus::SigningKey as PrivateKey;
use std::{path::PathBuf, time::Duration};

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::mpsc};

    use rand::rngs::OsRng;
    use tokio::sync::oneshot;

    use crate::model_thread::{spawn_model_thread, ModelThreadCommand, ModelThreadDispatcher};

    use super::*;

    const DURATION_1_SECS: Duration = Duration::from_secs(1);
    const DURATION_2_SECS: Duration = Duration::from_secs(2);
    const DURATION_5_SECS: Duration = Duration::from_secs(5);
    const DURATION_10_SECS: Duration = Duration::from_secs(10);

    struct TestModel {
        duration: Duration,
    }

    impl ModelTrait for TestModel {
        type Input = ();
        type Output = ();
        type LoadData = ();

        fn fetch(
            api_key: String,
            cache_dir: PathBuf,
            config: ModelConfig,
        ) -> Result<Self::LoadData, ModelError> {
            Ok(())
        }

        fn load(load_data: Self::LoadData) -> Result<Self, ModelError>
        where
            Self: Sized,
        {
            Ok(Self {
                duration: DURATION_1_SECS,
            })
        }

        fn model_type(&self) -> ModelType {
            todo!()
        }

        fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
            std::thread::sleep(self.duration);
            Ok(())
        }
    }

    impl ModelThreadDispatcher {
        fn test_start() -> Self {
            let mut model_senders = HashMap::with_capacity(4);

            for duration in [
                DURATION_1_SECS,
                DURATION_2_SECS,
                DURATION_5_SECS,
                DURATION_10_SECS,
            ] {
                let model_name = format!("test_model_{:?}", duration);

                let (model_sender, model_receiver) = mpsc::channel::<ModelThreadCommand>();
                model_senders.insert(model_name.clone(), model_sender.clone());

                let api_key = "".to_string();
                let cache_dir = "./".parse().unwrap();
                let model_config =
                    ModelConfig::new(model_name.clone(), "".to_string(), "".to_string(), 0, false);

                let private_key = PrivateKey::new(OsRng);
                let public_key = private_key.verification_key();

                let _join_handle = spawn_model_thread::<TestModel>(
                    model_name,
                    api_key,
                    cache_dir,
                    model_config,
                    public_key,
                    model_receiver,
                );
            }
            Self { model_senders }
        }
    }

    #[tokio::test]
    async fn test_model_thread() {
        let model_thread_dispatcher = ModelThreadDispatcher::test_start();

        for _ in 0..10 {
            for sender in model_thread_dispatcher.model_senders.values() {
                let (response_sender, response_request) = oneshot::channel();
                let command = ModelThreadCommand {
                    request: serde_json::Value::Null,
                    response_sender,
                };
                sender.send(command).expect("Failed to send command");
            }
        }
    }
}
