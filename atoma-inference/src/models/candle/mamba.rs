use std::path::PathBuf;

use candle_nn::VarBuilder;
use candle_transformers::models::mamba::Model as MambaModel;

use crate::models::{ModelError, ModelId, ModelTrait};

impl ModelTrait for MambaModel { 
    type Input = String;
    type Output = String;

    fn load(filenames: Vec<PathBuf>) -> Result<Self, ModelError>
        where
            Self: Sized {
        todo!()
    }

    fn model_id(&self) -> ModelId {
        todo!()
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output, ModelError> {
        todo!()
    }
}

