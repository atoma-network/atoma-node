#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::path::PathBuf;

use candle_transformers::models::stable_diffusion::{self};

use candle::{DType, Device, IndexOp, Module, Tensor, D};
use hf_hub::api::sync::ApiBuilder;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::models::{config::ModelConfig, types::{ModelType, PrecisionBits}, ModelError, ModelId, ModelTrait};

use super::{convert_to_image, device, save_tensor_to_file};

#[derive(Deserialize)]
pub struct Input {
    prompt: String,
    uncond_prompt: String,

    height: Option<usize>,
    width: Option<usize>,

    /// The UNet weight file, in .safetensors format.
    unet_weights: Option<String>,

    /// The CLIP weight file, in .safetensors format.
    clip_weights: Option<String>,

    /// The VAE weight file, in .safetensors format.
    vae_weights: Option<String>,

    /// The file specifying the tokenizer to used for tokenization.
    tokenizer: Option<String>,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    sliced_attention_size: Option<usize>,

    /// The number of steps to run the diffusion for.
    n_steps: Option<usize>,

    /// The number of samples to generate.
    num_samples: i64,

    sd_version: StableDiffusionVersion,

    use_flash_attn: bool,

    use_f16: bool,

    guidance_scale: Option<f64>,

    img2img: Option<String>,

    /// The strength, indicates how much to transform the initial image. The
    /// value must be between 0 and 1, a value of 1 discards the initial image
    /// information.
    img2img_strength: f64,

    /// The seed to use when generating random samples.
    seed: Option<u64>,
}

impl Input {
    pub fn default_prompt(prompt: String) -> Self {
        Self {
            prompt,
            uncond_prompt: "".to_string(),
            height: Some(256),
            width: Some(256),
            unet_weights: None,
            clip_weights: None,
            vae_weights: None,
            tokenizer: None,
            sliced_attention_size: None,
            n_steps: Some(20),
            num_samples: 1,
            sd_version: StableDiffusionVersion::V1_5,
            use_flash_attn: false,
            use_f16: true,
            guidance_scale: None,
            img2img: None,
            img2img_strength: 0.8,
            seed: Some(0),
        }
    }
}

impl From<&Input> for Fetch {
    fn from(input: &Input) -> Self {
        Self {
            tokenizer: input.tokenizer.clone(),
            sd_version: input.sd_version,
            use_f16: input.use_f16,
            clip_weights: input.clip_weights.clone(),
            vae_weights: input.vae_weights.clone(),
            unet_weights: input.unet_weights.clone(),
        }
    }
}

pub struct StableDiffusion {
    device: Device,
    dtype: DType,
}

impl ModelTrait for StableDiffusion {
    type Input = Input;
    type Output = Vec<(Vec<u8>, usize, usize)>;
    type LoadData = Self;

    fn load(
        load_data: Self::LoadData
    ) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        Ok(load_data)
    }

    fn fetch(config: ModelConfig) -> Result<Self::LoadData, ModelError> {
        let model_type = ModelType::from_str(&config.model_id())?;
        let which = match model_type {
            ModelType::StableDiffusionXl | ModelType::StableDiffusionTurbo => vec![true, false],
            _ => vec![true],
        };
        for first in which {
            let (clip_weights_file, tokenizer_file) = if first {
                (ModelFile::Clip, ModelFile::Tokenizer)
            } else {
                (ModelFile::Clip2, ModelFile::Tokenizer2)
            };

            let api_key = config.api_key();
            let cache_dir = config.cache_dir().into();
            let use_f16 = config.dtype() == "f16";

            clip_weights_file.get(api_key, cache_dir,model_type, false)?;
            ModelFile::Vae.get(api_key, cache_dir, model_type, use_f16)?;
            tokenizer_file.get(api_key, cache_dir, model_type, use_f16)?;
            ModelFile::Unet.get(api_key, cache_dir, model_type, use_f16)?;
        }
        Ok(Load)
    }

    fn model_type(&self) -> ModelType {
        self.model_type
    }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        if !(0. ..=1.).contains(&input.img2img_strength) {
            Err(ModelError::Config(format!(
                "img2img_strength must be between 0 and 1, got {}",
                input.img2img_strength,
            )))?
        }

        let guidance_scale = match input.guidance_scale {
            Some(guidance_scale) => guidance_scale,
            None => match input.sd_version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::Xl => 7.5,
                StableDiffusionVersion::Turbo => 0.,
            },
        };
        let n_steps = match input.n_steps {
            Some(n_steps) => n_steps,
            None => match input.sd_version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::Xl => 30,
                StableDiffusionVersion::Turbo => 1,
            },
        };
        let dtype = if input.use_f16 {
            DType::F16
        } else {
            DType::F32
        };
        let sd_config = match input.sd_version {
            StableDiffusionVersion::V1_5 => stable_diffusion::StableDiffusionConfig::v1_5(
                input.sliced_attention_size,
                input.height,
                input.width,
            ),
            StableDiffusionVersion::V2_1 => stable_diffusion::StableDiffusionConfig::v2_1(
                input.sliced_attention_size,
                input.height,
                input.width,
            ),
            StableDiffusionVersion::Xl => stable_diffusion::StableDiffusionConfig::sdxl(
                input.sliced_attention_size,
                input.height,
                input.width,
            ),
            StableDiffusionVersion::Turbo => stable_diffusion::StableDiffusionConfig::sdxl_turbo(
                input.sliced_attention_size,
                input.height,
                input.width,
            ),
        };

        let scheduler = sd_config.build_scheduler(n_steps)?;
        let device = device(self.device_id)?;
        if let Some(seed) = input.seed {
            device.set_seed(seed)?;
        }
        let use_guide_scale = guidance_scale > 1.0;

        let which = match input.sd_version {
            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => vec![true, false],
            _ => vec![true],
        };
        let text_embeddings = which
            .iter()
            .map(|first| {
                Self::text_embeddings(
                    &input.prompt,
                    &input.uncond_prompt,
                    input.tokenizer.clone(),
                    input.clip_weights.clone(),
                    input.sd_version,
                    &sd_config,
                    input.use_f16,
                    &device,
                    dtype,
                    use_guide_scale,
                    *first,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;

        let vae_weights = ModelFile::Vae.get(input.vae_weights, input.sd_version, input.use_f16)?;
        let vae = sd_config.build_vae(vae_weights, &device, dtype)?;
        let init_latent_dist = match &input.img2img {
            None => None,
            Some(image) => {
                let image = Self::image_preprocess(image)?.to_device(&device)?;
                Some(vae.encode(&image)?)
            }
        };
        let unet_weights =
            ModelFile::Unet.get(input.unet_weights, input.sd_version, input.use_f16)?;
        let unet = sd_config.build_unet(unet_weights, &device, 4, input.use_flash_attn, dtype)?;

        let t_start = if input.img2img.is_some() {
            n_steps - (n_steps as f64 * input.img2img_strength) as usize
        } else {
            0
        };
        let bsize = 1;

        let vae_scale = match input.sd_version {
            StableDiffusionVersion::V1_5
            | StableDiffusionVersion::V2_1
            | StableDiffusionVersion::Xl => 0.18215,
            StableDiffusionVersion::Turbo => 0.13025,
        };
        let mut res = Vec::new();

        for _ in 0..input.num_samples {
            let timesteps = scheduler.timesteps();
            let latents = match &init_latent_dist {
                Some(init_latent_dist) => {
                    let latents = (init_latent_dist.sample()? * vae_scale)?.to_device(&device)?;
                    if t_start < timesteps.len() {
                        let noise = latents.randn_like(0f64, 1f64)?;
                        scheduler.add_noise(&latents, noise, timesteps[t_start])?
                    } else {
                        latents
                    }
                }
                None => {
                    let latents = Tensor::randn(
                        0f32,
                        1f32,
                        (bsize, 4, sd_config.height / 8, sd_config.width / 8),
                        &device,
                    )?;
                    // scale the initial noise by the standard deviation required by the scheduler
                    (latents * scheduler.init_noise_sigma())?
                }
            };
            let mut latents = latents.to_dtype(dtype)?;

            for (timestep_index, &timestep) in timesteps.iter().enumerate() {
                if timestep_index < t_start {
                    continue;
                }
                let latent_model_input = if use_guide_scale {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };

                let latent_model_input =
                    scheduler.scale_model_input(latent_model_input, timestep)?;
                let noise_pred =
                    unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

                let noise_pred = if use_guide_scale {
                    let noise_pred = noise_pred.chunk(2, 0)?;
                    let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                    (noise_pred_uncond
                        + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
                } else {
                    noise_pred
                };

                latents = scheduler.step(&noise_pred, timestep, &latents)?;
            }
            save_tensor_to_file(&latents, "tensor1")?;
            let image = vae.decode(&(&latents / vae_scale)?)?;
            save_tensor_to_file(&image, "tensor2")?;
            let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
            save_tensor_to_file(&image, "tensor3")?;
            let image = (image.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?.i(0)?;
            save_tensor_to_file(&image, "tensor4")?;
            res.push(convert_to_image(&image)?);
        }
        Ok(res)
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Deserialize)]
enum StableDiffusionVersion {
    V1_5,
    V2_1,
    Xl,
    Turbo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}

impl ModelType {
    fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5 | Self::StableDiffusionV2_1 | Self::StableDiffusionXl | Self::StableDiffusionTurbo => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type")
        }
    }

    fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5 | Self::StableDiffusionV2_1 | Self::StableDiffusionXl | Self::StableDiffusionTurbo => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type")
        }
    }

    fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5 | Self::StableDiffusionV2_1 | Self::StableDiffusionXl | Self::StableDiffusionTurbo => { 
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type")
        }
    }

    fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5 | Self::StableDiffusionV2_1 | Self::StableDiffusionXl | Self::StableDiffusionTurbo => { 
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type")
        }
    }
}

impl ModelFile {
    fn get(
        &self,
        api_key: String,
        cache_dir: PathBuf,

        model_type: ModelType,
        use_f16: bool,
    ) -> Result<std::path::PathBuf, ModelError> {
                let (repo, path) = match self {
                    Self::Tokenizer => {
                        let tokenizer_repo = match model_type {
                            StableDiffusionVersion::V1_5 | StableDiffusionVersion::V2_1 => {
                                "openai/clip-vit-base-patch32"
                            }
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => {
                                // This seems similar to the patch32 version except some very small
                                // difference in the split regex.
                                "openai/clip-vit-large-patch14"
                            }
                        };
                        (tokenizer_repo, "tokenizer.json")
                    }
                    Self::Tokenizer2 => {
                        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")
                    }
                    Self::Clip => (model_type.repo(), model_type.clip_file(use_f16)),
                    Self::Clip2 => (model_type.repo(), model_type.clip2_file(use_f16)),
                    Self::Unet => (model_type.repo(), model_type.unet_file(use_f16)),
                    Self::Vae => {
                        // Override for SDXL when using f16 weights.
                        // See https://github.com/huggingface/candle/issues/1060
                        if matches!(
                            model_type,
                            ModelType::StableDiffusionXl | ModelType::StableDiffusionTurbo,
                        ) && use_f16
                        {
                            (
                                "madebyollin/sdxl-vae-fp16-fix",
                                "diffusion_pytorch_model.safetensors",
                            )
                        } else {
                            (model_type.repo(), model_type.vae_file(use_f16))
                        }
                    }
                };
                let filename = ApiBuilder::new()
                .with_progress(true)
                .with_token(Some(api_key))
                .with_cache_dir(cache_dir)
                .build()?;
                Ok(filename)
    }
}

impl StableDiffusion {
    #[allow(clippy::too_many_arguments)]
    fn text_embeddings(
        prompt: &str,
        uncond_prompt: &str,
        tokenizer: Option<String>,
        clip_weights: Option<String>,
        sd_version: StableDiffusionVersion,
        sd_config: &stable_diffusion::StableDiffusionConfig,
        use_f16: bool,
        device: &Device,
        dtype: DType,
        use_guide_scale: bool,
        first: bool,
    ) -> Result<Tensor, ModelError> {
        let (clip_weights_file, tokenizer_file) = if first {
            (ModelFile::Clip, ModelFile::Tokenizer)
        } else {
            (ModelFile::Clip2, ModelFile::Tokenizer2)
        };
        let tokenizer = tokenizer_file.get(tokenizer, sd_version, use_f16)?;
        let tokenizer = Tokenizer::from_file(tokenizer)?;
        let pad_id = match &sd_config.clip.pad_with {
            Some(padding) => {
                *tokenizer
                    .get_vocab(true)
                    .get(padding.as_str())
                    .ok_or(ModelError::Msg(format!(
                        "Padding token {padding} not found in the tokenizer vocabulary"
                    )))?
            }
            None => *tokenizer
                .get_vocab(true)
                .get("<|endoftext|>")
                .ok_or(ModelError::Msg("".to_string()))?,
        };
        let mut tokens = tokenizer.encode(prompt, true)?.get_ids().to_vec();
        while tokens.len() < sd_config.clip.max_position_embeddings {
            tokens.push(pad_id)
        }
        let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

        let clip_weights = clip_weights_file.get(clip_weights, sd_version, false)?;
        let clip_config = if first {
            &sd_config.clip
        } else {
            sd_config.clip2.as_ref().unwrap()
        };
        let text_model = stable_diffusion::build_clip_transformer(
            clip_config,
            clip_weights,
            device,
            DType::F64,
        )?;
        let text_embeddings = text_model.forward(&tokens)?;

        let text_embeddings = if use_guide_scale {
            let mut uncond_tokens = tokenizer.encode(uncond_prompt, true)?.get_ids().to_vec();
            while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
                uncond_tokens.push(pad_id)
            }

            let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;
            let uncond_embeddings = text_model.forward(&uncond_tokens)?;

            Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
        } else {
            text_embeddings.to_dtype(dtype)?
        };
        Ok(text_embeddings)
    }

    fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> Result<Tensor, ModelError> {
        let img = image::io::Reader::open(path)?.decode()?;
        let (height, width) = (img.height() as usize, img.width() as usize);
        let height = height - height % 32;
        let width = width - width % 32;
        let img = img.resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let img = img.to_rgb8();
        let img = img.into_raw();
        let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?
            .affine(2. / 255., -1.)?
            .unsqueeze(0)?;
        Ok(img)
    }
}
