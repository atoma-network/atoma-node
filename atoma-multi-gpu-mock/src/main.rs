// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use atoma_types::{AtomaChildMessage, AtomaInferenceMessage, TextModelInput, TextModelOutput};
use candle::DType;
use std::env;
use std::path::PathBuf;
use std::str::FromStr;
use tungstenite::{connect, Message};
use url::Url;

mod models;
mod types;
pub use models::*;
pub use types::*;

struct SingleGpu {
    id: Option<u8>,
    rank: usize,
    num_shards: usize,
    comm: Option<usize>,
    model: Option<models::llama::Model>,
}

impl SingleGpu {
    fn new(rank: usize, num_shards: usize) -> Self {
        Self {
            id: None,
            rank,
            num_shards,
            comm: None,
            model: None,
        }
    }

    // This should be called only on the main gpu.
    fn create_main_id(&mut self) -> Result<(), ModelError> {
        self.id = Some(42);
        self.create_comm()?;
        Ok(())
    }

    fn copy_id_from(&mut self, data: Vec<u8>) -> Result<(), ModelError> {
        self.id = Some(data[0]);
        self.create_comm()?;
        Ok(())
    }

    fn create_comm(&mut self) -> Result<(), ModelError> {
        // Don't mistake nccl device with cuda device.
        self.comm = Some(43);
        println!("Rank {} spawned", self.rank);
        Ok(())
    }

    fn load_model(
        &mut self,
        config_filename: PathBuf,
        dtype: DType,
        filenames: Vec<PathBuf>,
        tokenizer_filename: PathBuf,
    ) -> Result<(), ModelError> {
        if self.comm.is_none() {
            panic!("Comm not initialized");
        }
        if self.model.is_some() {
            panic!("Model already loaded");
        }
        self.model = Some(llama::Model::load(
            config_filename,
            dtype,
            filenames,
            tokenizer_filename,
        )?);
        Ok(())
    }

    fn inference(&self, input: TextModelInput) -> Result<TextModelOutput, ModelError> {
        // self.model.as_ref().unwrap().inference(input)
        Ok(TextModelOutput {
            text: input.prompt,
            time: 1.0,
            tokens_count: 1,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), ()> {
    let port = env::args().nth(1).expect("Expected a port number");
    let num_shards = env::args()
        .nth(2)
        .expect("Expected the number of shards")
        .parse()
        .expect("Expected a number");
    let rank = env::args()
        .nth(3)
        .expect("Expected the rank")
        .parse()
        .expect("Expected a number");

    let url = format!("ws://127.0.0.1:{}/socket", port);
    let (mut socket, response) = connect(Url::parse(url.as_str()).unwrap()).expect("Can't connect");
    println!("Connected to the server: {}", response.status());
    let mut gpu_instance = SingleGpu::new(rank, num_shards);
    if rank == 0 {
        gpu_instance.create_main_id().unwrap();
    }
    let id = gpu_instance.id.map(|x| vec![x]);

    socket
        .send(Message::Text(
            serde_json::to_string(&AtomaChildMessage::Initialized(id)).unwrap(),
        ))
        .unwrap();
    if rank == 0 {
        socket
            .send(Message::Text(
                serde_json::to_string(&AtomaChildMessage::CommsReady).unwrap(),
            ))
            .unwrap();
    }
    loop {
        let msg = socket.read().unwrap();
        let msg: AtomaInferenceMessage = serde_json::from_str(msg.to_string().as_str()).unwrap();
        match msg {
            AtomaInferenceMessage::InitializeComm(data) => {
                if gpu_instance.id.is_some() {
                    panic!("Id already initialized");
                }
                gpu_instance.copy_id_from(data).unwrap();
                socket
                    .send(Message::Text(
                        serde_json::to_string(&AtomaChildMessage::CommsReady).unwrap(),
                    ))
                    .unwrap();
            }
            AtomaInferenceMessage::LoadModel(
                config_filename,
                dtype,
                filenames,
                tokenizer_filename,
            ) => {
                let dtype = DType::from_str(dtype.as_str()).unwrap();
                gpu_instance
                    .load_model(config_filename, dtype, filenames, tokenizer_filename)
                    .unwrap();
                socket
                    .send(Message::Text(
                        serde_json::to_string(&AtomaChildMessage::Loaded).unwrap(),
                    ))
                    .unwrap();
            }
            AtomaInferenceMessage::Inference(input) => {
                let result = gpu_instance.inference(input).unwrap();
                socket
                    .send(Message::Text(
                        serde_json::to_string(&AtomaChildMessage::InferenceResult(result)).unwrap(),
                    ))
                    .unwrap();
            }
            AtomaInferenceMessage::Exit => break,
        }
    }
    Ok(())
}
