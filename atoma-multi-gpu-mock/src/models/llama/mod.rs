use std::path::PathBuf;

use atoma_types::{TextModelInput, TextModelOutput};
use candle::DType;

use crate::ModelError;

pub struct Model {}

impl Model {
    pub fn load(
        config_filename: PathBuf,
        dtype: DType,
        filenames: Vec<PathBuf>,
        tokenizer_filename: PathBuf,
    ) -> Result<Self, ModelError> {
        println!("Loading model...");
        println!("Config file: {:?}", config_filename);
        println!("tokenizer file: {:?}", tokenizer_filename);
        println!("filenames: {:?}", filenames);
        println!("Dtype: {:?}", dtype);
        Ok(Self {})
    }

    pub fn inference(&self, input: TextModelInput) -> Result<TextModelOutput, ModelError> {
        Ok(TextModelOutput {
            text: input.prompt,
            time: 1.0,
            tokens_count: 1,
        })
    }
}
