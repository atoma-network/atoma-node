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
use candle::{DType, Device};
use cudarc::driver::safe::CudaDevice;
use cudarc::nccl::safe::{Comm, Id};
use std::env;
use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr;
use tungstenite::{connect, Message};
use url::Url;

mod models;
mod token_output_stream;
mod types;
pub use models::*;
pub use types::*;

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<PathBuf>, ModelError> {
    let json_file = repo.get(json_file)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value = serde_json::from_reader(&json_file)?;
    let weight_map = match json.get("weight_map") {
        None => bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle::Error::wrap))
        .collect::<candle::Result<Vec<_>>>()?;
    Ok(safetensors_files)
}

struct SingleGpu {
    id: Option<Id>,
    rank: usize,
    num_shards: usize,
    comm: Option<Rc<Comm>>,
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
        self.id = Some(Id::new().map_err(|err| ModelError::NcclError(err))?);
        Ok(())
    }

    fn copy_id_from(&mut self, data: Vec<u8>) -> Result<(), ModelError> {
        self.id = Some(Id::uninit(
            data.into_iter()
                .map(|i| i as i8)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        ));
        self.create_comm()?;
        Ok(())
    }

    fn create_comm(&mut self) -> Result<(), ModelError> {
        // Don't mistake nccl device with cuda device.
        let device = CudaDevice::new(self.rank).unwrap();
        self.comm = Some(
            match Comm::from_rank(device, self.rank, self.num_shards, self.id.unwrap()) {
                Ok(comm) => Rc::new(comm),
                Err(err) => panic!("nccl error {:?}", err.0),
            },
        );
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
            Device::new_cuda(self.rank).unwrap(),
            dtype,
            filenames,
            tokenizer_filename,
            Rc::clone(self.comm.as_ref().unwrap()),
        )?);
        Ok(())
    }

    fn inference(&self, input: TextModelInput) -> Result<TextModelOutput, ModelError> {
        self.model.as_ref().unwrap().inference(input)
    }
}

fn main() -> Result<(), ()> {
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
    let id = gpu_instance
        .id
        .map(|id| id.internal().iter().map(|&i| i as u8).collect::<Vec<_>>());

    socket
        .send(Message::Text(
            serde_json::to_string(&AtomaChildMessage::Initialized(id)).unwrap(),
        ))
        .unwrap();
    if rank == 0 {
        gpu_instance.create_comm().unwrap();
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
