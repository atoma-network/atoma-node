use std::{
    net::TcpListener,
    path::PathBuf,
    process::Command,
    str::FromStr,
    sync::{Arc, Barrier, Mutex},
    thread,
};

use atoma_types::{AtomaChildMessage, AtomaInferenceMessage, TextModelInput, TextModelOutput};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

use tungstenite::accept;

use crate::models::{config::ModelConfig, types::ModelType, ModelError, ModelTrait};

use super::hub_load_safetensors;

pub struct LlamaMultiModel {
    model_type: ModelType,
    gpu_instances: Arc<Mutex<Vec<Arc<Mutex<tungstenite::WebSocket<std::net::TcpStream>>>>>>,
    wait_for_result: Arc<Barrier>,
    wait_for_send: Arc<Barrier>,
    result: Arc<Mutex<Option<TextModelOutput>>>,
}

pub struct LlamaMultiLoadData {
    file_paths: Vec<PathBuf>,
    dtype: String,
    model_type: ModelType,
    num_shards: usize,
}

impl ModelTrait for LlamaMultiModel {
    type Input = TextModelInput;
    type Output = TextModelOutput;
    type LoadData = LlamaMultiLoadData;

    fn fetch(
        api_key: String,
        cache_dir: PathBuf,
        config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir)
            .build()?;

        let model_type = ModelType::from_str(&config.model_id())?;
        let repo_id = model_type.repo().to_string();
        let revision = model_type.default_revision().to_string();

        let repo = api.repo(Repo::with_revision(
            repo_id.clone(),
            RepoType::Model,
            revision,
        ));
        let config_file_path = repo.get("config.json")?;
        let tokenizer_file_path = repo.get("tokenizer.json")?;

        let model_weights_file_paths = if &repo_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0" {
            vec![repo.get("model.safetensors")?]
        } else {
            hub_load_safetensors(&repo, "model.safetensors.index.json")?
        };

        let mut file_paths = Vec::with_capacity(2 + model_weights_file_paths.len());
        file_paths.extend(vec![config_file_path, tokenizer_file_path]);
        file_paths.extend(model_weights_file_paths);

        Ok(Self::LoadData {
            dtype: config.dtype(),
            file_paths,
            model_type: ModelType::from_str(&config.model_id())?,
            num_shards: config.num_shards(),
        })
    }

    fn model_type(&self) -> ModelType {
        self.model_type.clone()
    }

    fn load(load_data: Self::LoadData) -> Result<Self, ModelError> {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = server.local_addr().unwrap().port();
        println!("Server listening on port {}", port);
        for rank in 0..load_data.num_shards {
            Command::new("cargo")
                .arg("run")
                .arg("--bin")
                .arg("atoma-multi-gpu")
                .arg("--")
                .arg(port.to_string())
                .arg(load_data.num_shards.to_string())
                .arg(rank.to_string())
                .spawn()
                .unwrap();
        }
        let mut num_connections = 0;
        let gpu_instances = Arc::new(Mutex::new(Vec::with_capacity(load_data.num_shards)));
        let init_barrier = Arc::new(Barrier::new(load_data.num_shards));
        let loaded_barrier = Arc::new(Barrier::new(load_data.num_shards + 1));
        let wait_for_send = Arc::new(Barrier::new(load_data.num_shards + 1));
        let wait_for_result = Arc::new(Barrier::new(load_data.num_shards + 1));
        let result = Arc::new(Mutex::new(None));
        for stream in server.incoming() {
            let gpu_instances = Arc::clone(&gpu_instances);
            let init_barrier = Arc::clone(&init_barrier);
            let loaded_barrier = Arc::clone(&loaded_barrier);
            let wait_for_result = Arc::clone(&wait_for_result);
            let wait_for_send = Arc::clone(&wait_for_send);
            let config_file_path = load_data.file_paths[0].clone();
            let tokenizer_filename = load_data.file_paths[1].clone();
            let filenames = load_data.file_paths[2..].to_vec();
            let dtype = load_data.dtype.clone();
            let result = Arc::clone(&result);
            thread::spawn(move || {
                let websocket = Arc::new(Mutex::new(accept(stream.unwrap()).unwrap()));
                let index = {
                    let mut gpu_instances = gpu_instances.lock().unwrap();
                    gpu_instances.push(Arc::clone(&websocket));
                    gpu_instances.len() - 1
                };
                init_barrier.wait();
                loop {
                    let msg = (*websocket.lock().unwrap()).read().unwrap();
                    if msg.is_text() {
                        let message: AtomaChildMessage =
                            serde_json::from_str(msg.to_string().as_str()).unwrap();
                        match message {
                            AtomaChildMessage::Initialized(nccl_id) => {
                                if let Some(nccl_id) = nccl_id {
                                    let mut gpu_instances = gpu_instances.lock().unwrap();
                                    for (i, websocket) in gpu_instances.iter_mut().enumerate() {
                                        if i != index {
                                            (*websocket.lock().unwrap())
                                                .send(tungstenite::Message::Text(
                                                    serde_json::to_string(
                                                        &AtomaInferenceMessage::InitializeComm(
                                                            nccl_id.clone(),
                                                        ),
                                                    )
                                                    .unwrap(),
                                                ))
                                                .unwrap();
                                        }
                                    }
                                    init_barrier.wait();
                                } else {
                                    init_barrier.wait();
                                }
                            }
                            AtomaChildMessage::CommsReady => {
                                (*websocket.lock().unwrap())
                                    .send(tungstenite::Message::Text(
                                        serde_json::to_string(&AtomaInferenceMessage::LoadModel(
                                            config_file_path.clone(),
                                            dtype.clone(),
                                            filenames.clone(),
                                            tokenizer_filename.clone(),
                                        ))
                                        .unwrap(),
                                    ))
                                    .unwrap();
                            }
                            AtomaChildMessage::Loaded => {
                                loaded_barrier.wait(); // This will let the main thread know that the model is loaded.
                                wait_for_send.wait(); // This will prevent to lock the websocket in the read state.
                            }
                            AtomaChildMessage::InferenceResult(output) => {
                                *result.lock().unwrap() = Some(output);
                                wait_for_result.wait(); // This will let the main thread know that the result is filled.
                                wait_for_send.wait(); // This will prevent to lock the websocket in the read state.
                            }
                        }
                    }
                }
            });
            num_connections += 1;
            if num_connections == load_data.num_shards {
                break;
            }
        }
        loaded_barrier.wait();
        Ok(LlamaMultiModel {
            model_type: load_data.model_type,
            gpu_instances,
            wait_for_send,
            wait_for_result,
            result,
        })
    }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        for websocket in self.gpu_instances.lock().unwrap().iter_mut() {
            (*websocket.lock().unwrap())
                .send(tungstenite::Message::Text(
                    serde_json::to_string(&AtomaInferenceMessage::Inference(input.clone()))
                        .unwrap(),
                ))
                .unwrap();
        }
        self.wait_for_send.wait(); // This will let the websocket to process the above send
        self.wait_for_result.wait(); // This will pass once the result is filled.
        Ok(self.result.lock().unwrap().take().unwrap())
    }
}

impl Drop for LlamaMultiModel {
    fn drop(&mut self) {
        for websocket in self.gpu_instances.lock().unwrap().iter_mut() {
            (*websocket.lock().unwrap())
                .send(tungstenite::Message::Text(
                    serde_json::to_string(&AtomaInferenceMessage::Exit).unwrap(),
                ))
                .unwrap();
        }
        self.wait_for_send.wait(); // This will let the websocket to process the above send
    }
}
