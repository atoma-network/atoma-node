use std::{fs::File, io::Write, path::PathBuf};

use candle::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device, Tensor,
};
use tracing::info;

use crate::bail;

use super::ModelError;

pub mod falcon;
pub mod mamba;
pub mod stable_diffusion;

pub fn device(device_id: usize) -> Result<Device, candle::Error> {
    if cuda_is_available() {
        info!("Using CUDA");
        Device::new_cuda(device_id)
    } else if metal_is_available() {
        info!("Using Metal");
        Device::new_metal(device_id)
    } else {
        info!("Using Cpu");
        Ok(Device::Cpu)
    }
}

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>, ModelError> {
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

pub fn save_image<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> Result<(), ModelError> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        bail!("save_image expects an input of shape (3, height, width)")
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => bail!("error saving image {p:?}"),
        };
    image.save(p).map_err(candle::Error::wrap)?;
    Ok(())
}

pub fn save_tensor_to_file(tensor: &Tensor, filename: &str) -> Result<(), candle::Error> {
    let json_output = serde_json::to_string(
        &tensor
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_dtype(DType::F64)?
            .to_vec1::<f64>()?,
    )
    .unwrap();
    let mut file = File::create(PathBuf::from(filename))?;
    file.write_all(json_output.as_bytes())?;
    Ok(())
}
