use std::{path::PathBuf, str::FromStr};

use candle_transformers::models::stable_diffusion::{
    self, clip::ClipTextTransformer, unet_2d::UNet2DConditionModel, vae::AutoEncoderKL,
    StableDiffusionConfig,
};

use candle::{DType, Device, IndexOp, Module, Tensor, D};
use hf_hub::api::sync::ApiBuilder;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    bail,
    models::{config::ModelConfig, types::ModelType, ModelError, ModelTrait},
};

use super::{convert_to_image, device, save_tensor_to_file};

#[derive(Deserialize)]
pub struct Input {
    prompt: String,
    uncond_prompt: String,

    height: Option<usize>,
    width: Option<usize>,

    /// The number of steps to run the diffusion for.
    n_steps: Option<usize>,

    /// The number of samples to generate.
    num_samples: i64,

    sd_version: StableDiffusionVersion,

    guidance_scale: Option<f64>,

    img2img: Option<String>,

    /// The strength, indicates how much to transform the initial image. The
    /// value must be between 0 and 1, a value of 1 discards the initial image
    /// information.
    img2img_strength: f64,

    /// The seed to use when generating random samples.
    seed: Option<u64>,
}

pub struct StableDiffusionLoadData {
    device: Device,
    dtype: DType,
    model_type: ModelType,
    sliced_attention_size: Option<usize>,
    clip_weights_file_paths: Vec<PathBuf>,
    tokenizer_file_paths: Vec<PathBuf>,
    vae_weights_file_path: PathBuf,
    unet_weights_file_path: PathBuf,
    use_flash_attention: bool,
}

pub struct StableDiffusion {
    config: StableDiffusionConfig,
    device: Device,
    dtype: DType,
    model_type: ModelType,
    text_model: ClipTextTransformer,
    text_model_2: Option<ClipTextTransformer>,
    tokenizer: Tokenizer,
    tokenizer_2: Option<Tokenizer>,
    unet: UNet2DConditionModel,
    vae: AutoEncoderKL,
}

impl ModelTrait for StableDiffusion {
    type Input = Input;
    type Output = Vec<(Vec<u8>, usize, usize)>;
    type LoadData = StableDiffusionLoadData;

    fn fetch(cache_dir: PathBuf, config: ModelConfig) -> Result<Self::LoadData, ModelError> {
        let device = device(config.device_id())?;
        let dtype = DType::from_str(&config.dtype())?;
        let model_type = ModelType::from_str(&config.model_id())?;
        let which = match model_type {
            ModelType::StableDiffusionXl | ModelType::StableDiffusionTurbo => vec![true, false],
            _ => vec![true],
        };

        let api_key = config.api_key();
        let use_f16 = config.dtype() == "f16";

        let vae_weights_file_path = ModelFile::Vae.get(
            api_key.clone(),
            cache_dir.clone(),
            model_type.clone(),
            use_f16,
        )?;
        let unet_weights_file_path = ModelFile::Unet.get(
            api_key.clone(),
            cache_dir.clone(),
            model_type.clone(),
            use_f16,
        )?;

        let mut clip_weights_file_paths = vec![];
        let mut tokenizer_file_paths = vec![];

        for first in which {
            let (clip_weights_file, tokenizer_file) = if first {
                (ModelFile::Clip, ModelFile::Tokenizer)
            } else {
                (ModelFile::Clip2, ModelFile::Tokenizer2)
            };

            let clip_weights_file_path = clip_weights_file.get(
                api_key.clone(),
                cache_dir.clone(),
                model_type.clone(),
                false,
            )?;
            let tokenizer_file_path = tokenizer_file.get(
                api_key.clone(),
                cache_dir.clone(),
                model_type.clone(),
                use_f16,
            )?;

            clip_weights_file_paths.push(clip_weights_file_path);
            tokenizer_file_paths.push(tokenizer_file_path);
        }

        Ok(Self::LoadData {
            device,
            dtype,
            model_type,
            sliced_attention_size: config.sliced_attention_size(),
            clip_weights_file_paths,
            tokenizer_file_paths,
            vae_weights_file_path,
            unet_weights_file_path,
            use_flash_attention: config.use_flash_attention(),
        })
    }

    fn load(load_data: Self::LoadData) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let sliced_attention_size = load_data.sliced_attention_size;
        let config = match load_data.model_type {
            ModelType::StableDiffusionV1_5 => {
                stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, None, None)
            }
            ModelType::StableDiffusionV2_1 => {
                stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, None, None)
            }
            ModelType::StableDiffusionXl => {
                stable_diffusion::StableDiffusionConfig::sdxl(sliced_attention_size, None, None)
            }
            ModelType::StableDiffusionTurbo => stable_diffusion::StableDiffusionConfig::sdxl_turbo(
                sliced_attention_size,
                None,
                None,
            ),
            _ => bail!("Invalid stable diffusion model type"),
        };

        let (tokenizer, tokenizer_2) = match load_data.model_type {
            ModelType::StableDiffusionXl | ModelType::StableDiffusionTurbo => (
                Tokenizer::from_file(load_data.tokenizer_file_paths[0].clone())?,
                Some(Tokenizer::from_file(
                    load_data.tokenizer_file_paths[1].clone(),
                )?),
            ),
            _ => (
                Tokenizer::from_file(load_data.tokenizer_file_paths[0].clone())?,
                None,
            ), // INTEGRITY: we have checked previously if the model type is valid for the family of stable diffusion models
        };

        let text_model = stable_diffusion::build_clip_transformer(
            &config.clip,
            load_data.clip_weights_file_paths[0].clone(),
            &load_data.device,
            load_data.dtype,
        )?;
        let text_model_2 = if let Some(clip_config_2) = &config.clip2 {
            Some(stable_diffusion::build_clip_transformer(
                clip_config_2,
                load_data.clip_weights_file_paths[1].clone(),
                &load_data.device,
                load_data.dtype,
            )?)
        } else {
            None
        };

        let vae = config.build_vae(
            load_data.vae_weights_file_path,
            &load_data.device,
            load_data.dtype,
        )?;
        let unet = config.build_unet(
            load_data.unet_weights_file_path,
            &load_data.device,
            4, // see https://github.com/huggingface/candle/blob/main/candle-examples/examples/stable-diffusion/main.rs#L492
            load_data.use_flash_attention,
            load_data.dtype,
        )?;

        Ok(Self {
            config,
            device: load_data.device,
            dtype: load_data.dtype,
            model_type: load_data.model_type,
            tokenizer,
            tokenizer_2,
            text_model,
            text_model_2,
            vae,
            unet,
        })
    }

    fn model_type(&self) -> ModelType {
        self.model_type.clone()
    }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        if !(0. ..=1.).contains(&input.img2img_strength) {
            Err(ModelError::Config(format!(
                "img2img_strength must be between 0 and 1, got {}",
                input.img2img_strength,
            )))?
        }

        // self.config.height = input.height;
        // self.config.width = input.width;

        let guidance_scale = match input.guidance_scale {
            Some(guidance_scale) => guidance_scale,
            None => match self.model_type {
                ModelType::StableDiffusionV1_5
                | ModelType::StableDiffusionV2_1
                | ModelType::StableDiffusionXl => 7.5,
                ModelType::StableDiffusionTurbo => 0.,
                _ => bail!("Invalid stable diffusion model type"),
            },
        };
        let n_steps = match input.n_steps {
            Some(n_steps) => n_steps,
            None => match self.model_type {
                ModelType::StableDiffusionV1_5
                | ModelType::StableDiffusionV2_1
                | ModelType::StableDiffusionXl => 30,
                ModelType::StableDiffusionTurbo => 1,
                _ => bail!("Invalid stable diffusion model type"),
            },
        };

        let scheduler = self.config.build_scheduler(n_steps)?;
        if let Some(seed) = input.seed {
            self.device.set_seed(seed)?;
        }
        let use_guide_scale = guidance_scale > 1.0;

        let which = match self.model_type {
            ModelType::StableDiffusionXl | ModelType::StableDiffusionTurbo => vec![true, false],
            _ => vec![true], // INTEGRITY: we have checked previously if the model type is valid for the family of stable diffusion models
        };
        let text_embeddings = which
            .iter()
            .map(|first| {
                Self::text_embeddings(
                    &input.prompt,
                    &input.uncond_prompt,
                    &self.tokenizer,
                    self.tokenizer_2.as_ref(),
                    &self.text_model,
                    self.text_model_2.as_ref(),
                    &self.config,
                    &self.device,
                    self.dtype,
                    use_guide_scale,
                    *first,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;

        let init_latent_dist = match &input.img2img {
            None => None,
            Some(image) => {
                let image = Self::image_preprocess(image)?.to_device(&self.device)?;
                Some(self.vae.encode(&image)?)
            }
        };

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
                    let latents =
                        (init_latent_dist.sample()? * vae_scale)?.to_device(&self.device)?;
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
                        (
                            bsize,
                            4,
                            input.height.unwrap_or(512) / 8,
                            input.width.unwrap_or(512) / 8,
                        ),
                        &self.device,
                    )?;
                    // scale the initial noise by the standard deviation required by the scheduler
                    (latents * scheduler.init_noise_sigma())?
                }
            };
            let mut latents = latents.to_dtype(self.dtype)?;

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
                    self.unet
                        .forward(&latent_model_input, timestep as f64, &text_embeddings)?;

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
            let image = self.vae.decode(&(&latents / vae_scale)?)?;
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

impl ModelType {
    fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5
            | Self::StableDiffusionV2_1
            | Self::StableDiffusionXl
            | Self::StableDiffusionTurbo => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type"),
        }
    }

    fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5
            | Self::StableDiffusionV2_1
            | Self::StableDiffusionXl
            | Self::StableDiffusionTurbo => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type"),
        }
    }

    fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5
            | Self::StableDiffusionV2_1
            | Self::StableDiffusionXl
            | Self::StableDiffusionTurbo => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type"),
        }
    }

    fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5
            | Self::StableDiffusionV2_1
            | Self::StableDiffusionXl
            | Self::StableDiffusionTurbo => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type"),
        }
    }
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

impl ModelFile {
    fn get(
        &self,
        api_key: String,
        cache_dir: PathBuf,

        model_type: ModelType,
        use_f16: bool,
    ) -> Result<PathBuf, ModelError> {
        let (repo, path) = match self {
            Self::Tokenizer => {
                let tokenizer_repo = match model_type {
                    ModelType::StableDiffusionV1_5 | ModelType::StableDiffusionV2_1 => {
                        "openai/clip-vit-base-patch32"
                    }
                    ModelType::StableDiffusionXl | ModelType::StableDiffusionTurbo => {
                        // This seems similar to the patch32 version except some very small
                        // difference in the split regex.
                        "openai/clip-vit-large-patch14"
                    }
                    _ => bail!("Invalid stable diffusion model type"),
                };
                (tokenizer_repo, "tokenizer.json")
            }
            Self::Tokenizer2 => ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json"),
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
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir)
            .build()?;
        let filename = api.model(repo.to_string()).get(path)?;
        Ok(filename)
    }
}

impl StableDiffusion {
    #[allow(clippy::too_many_arguments)]
    fn text_embeddings(
        prompt: &str,
        uncond_prompt: &str,
        tokenizer: &Tokenizer,
        tokenizer_2: Option<&Tokenizer>,
        text_model: &ClipTextTransformer,
        text_model_2: Option<&ClipTextTransformer>,
        sd_config: &StableDiffusionConfig,
        device: &Device,
        dtype: DType,
        use_guide_scale: bool,
        first: bool,
    ) -> Result<Tensor, ModelError> {
        let (tokenizer, text_model) = if first {
            (tokenizer, text_model)
        } else {
            (tokenizer_2.unwrap(), text_model_2.unwrap())
        };
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
