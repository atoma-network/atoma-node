use std::{path::PathBuf, str::FromStr, time::Instant};

use atoma_types::AtomaStreamingData;
use candle_transformers::models::stable_diffusion::{
    self, clip::ClipTextTransformer, unet_2d::UNet2DConditionModel, vae::AutoEncoderKL,
    StableDiffusionConfig,
};

use candle::{DType, Device, IndexOp, Module, Tensor, D};
use hf_hub::api::sync::ApiBuilder;
use serde::Serialize;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{debug, info, instrument};

use crate::{
    bail,
    models::{
        candle::save_image,
        config::ModelConfig,
        types::{LlmOutput, ModelType, StableDiffusionInput},
        ModelError, ModelTrait,
    },
};

use super::{convert_to_image, device, save_tensor_to_file};

/// Stable diffusion load data
pub struct StableDiffusionLoadData {
    /// Device
    device: Device,
    /// DType, for the decimal precision which the
    /// model should run on
    dtype: DType,
    /// The model's unique identifier
    model_type: ModelType,
    /// Size of sliced attention, if applicable
    sliced_attention_size: Option<usize>,
    /// The image clip file weights paths
    clip_weights_file_paths: Vec<PathBuf>,
    /// Tokenizer file paths
    tokenizer_file_paths: Vec<PathBuf>,
    /// Variational auto encoder file weights paths
    vae_weights_file_path: PathBuf,
    /// Unet file weights paths
    unet_weights_file_path: PathBuf,
    /// To use or not flash attention
    use_flash_attention: bool,
}

/// `StableDiffusion` - encapsulates a
/// Stable diffusion model, together with further metadata
/// necessary to run inference
pub struct StableDiffusion {
    /// Stable diffusion configuration
    config: StableDiffusionConfig,
    /// Device hosting the model
    device: Device,
    /// DType, to control model prevision
    /// while running inference
    dtype: DType,
    /// The model's unique identifier
    model_type: ModelType,
    /// The text model, to parse the initial text
    text_model: ClipTextTransformer,
    /// Optional second text model, for more detailed
    /// expressivity
    text_model_2: Option<ClipTextTransformer>,
    /// Tokenizer
    tokenizer: Tokenizer,
    /// Optional tokenizer
    tokenizer_2: Option<Tokenizer>,
    /// Unet 2-dimensional model
    unet: UNet2DConditionModel,
    /// Variational auto-encoder model
    vae: AutoEncoderKL,
}

/// Stable diffusion output
#[derive(Serialize)]
pub struct StableDiffusionOutput {
    /// Data buffer of the image encoding
    pub image_data: Vec<u8>,
    /// Height of the image
    pub height: usize,
    /// Width of the image
    pub width: usize,
    /// Number of input tokens
    input_tokens: usize,
    /// Time to generate output
    time_to_generate: f64,
}

impl LlmOutput for StableDiffusionOutput {
    fn num_input_tokens(&self) -> usize {
        self.input_tokens
    }

    fn num_output_tokens(&self) -> Option<usize> {
        None
    }

    fn time_to_generate(&self) -> f64 {
        self.time_to_generate
    }
}

impl ModelTrait for StableDiffusion {
    type Input = StableDiffusionInput;
    type Output = StableDiffusionOutput;
    type LoadData = StableDiffusionLoadData;

    #[instrument(skip_all)]
    fn fetch(
        api_key: String,
        cache_dir: PathBuf,
        config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError> {
        let device = device(config.device_first_id())?;
        let dtype = DType::from_str(&config.dtype())?;
        let model_type = ModelType::from_str(&config.model_id())?;
        let which = match model_type {
            ModelType::StableDiffusionXl | ModelType::StableDiffusionTurbo => vec![true, false],
            _ => vec![true],
        };

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
            sliced_attention_size: None,
            clip_weights_file_paths,
            tokenizer_file_paths,
            vae_weights_file_path,
            unet_weights_file_path,
            use_flash_attention: config.use_flash_attention(),
        })
    }

    #[instrument(skip_all)]
    fn load(
        load_data: Self::LoadData,
        _: mpsc::Sender<AtomaStreamingData>,
    ) -> Result<Self, ModelError>
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

        info!("Loading text model...");
        let text_model = stable_diffusion::build_clip_transformer(
            &config.clip,
            load_data.clip_weights_file_paths[0].clone(),
            &load_data.device,
            load_data.dtype,
        )?;
        let text_model_2 = if let Some(clip_config_2) = &config.clip2 {
            info!("Loading second text model...");
            Some(stable_diffusion::build_clip_transformer(
                clip_config_2,
                load_data.clip_weights_file_paths[1].clone(),
                &load_data.device,
                load_data.dtype,
            )?)
        } else {
            None
        };

        info!("Loading variational auto encoder model...");
        let vae = config.build_vae(
            load_data.vae_weights_file_path,
            &load_data.device,
            load_data.dtype,
        )?;
        info!("Loading unet model...");
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

    #[instrument(skip_all)]
    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        if !(0. ..=1.).contains(&input.img2img_strength) {
            Err(ModelError::Config(format!(
                "img2img_strength must be between 0 and 1, got {}",
                input.img2img_strength,
            )))?
        }

        let start_gen = Instant::now();

        let height = input.height.unwrap_or(512);
        let width = input.width.unwrap_or(512);

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
        if let Some(seed) = input.random_seed {
            self.device.set_seed(seed as u64)?;
        }
        let use_guide_scale = guidance_scale > 1.0;

        let which = match self.model_type {
            ModelType::StableDiffusionXl | ModelType::StableDiffusionTurbo => vec![true, false],
            _ => vec![true], // INTEGRITY: we have checked previously if the model type is valid for the family of stable diffusion models
        };

        debug!("Computing text embeddings...");
        let text_embeddings_data = which
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

        // We sum all input tokens from each tokenizer run
        let num_input_tokens = text_embeddings_data
            .iter()
            .map(|(_, input_tokens)| input_tokens)
            .sum();
        let text_embeddings = text_embeddings_data
            .iter()
            .map(|(t, _)| t)
            .collect::<Vec<_>>();

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

        let model_type = input.model;
        let vae_scale = match model_type {
            ModelType::StableDiffusionV1_5
            | ModelType::StableDiffusionV2_1
            | ModelType::StableDiffusionXl => 0.18215,
            ModelType::StableDiffusionTurbo => 0.13025,
            _ => bail!("Invalid stable diffusion model type"),
        };
        let mut res = (vec![], 0, 0);

        for idx in 0..input.num_samples {
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
                    let latents =
                        Tensor::randn(0f32, 1f32, (bsize, 4, height / 8, width / 8), &self.device)?;
                    // scale the initial noise by the standard deviation required by the scheduler
                    (latents * scheduler.init_noise_sigma())?
                }
            };
            let mut latents = latents.to_dtype(self.dtype)?;

            for (timestep_index, &timestep) in timesteps.iter().enumerate() {
                if timestep_index < t_start {
                    continue;
                }
                let start_time = std::time::Instant::now();
                let latent_model_input = if use_guide_scale {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };

                let latent_model_input =
                    scheduler.scale_model_input(latent_model_input, timestep)?;
                debug!("Computing noise prediction...");
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
                let dt = start_time.elapsed().as_secs_f32();
                debug!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);
            }

            let dt = start_gen.elapsed();
            info!("Generated response in {:?}", dt);
            debug!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                input.num_samples
            );

            save_tensor_to_file(&latents, "tensor1")?;
            let image = self.vae.decode(&(&latents / vae_scale)?)?;
            save_tensor_to_file(&image, "tensor2")?;
            let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
            save_tensor_to_file(&image, "tensor3")?;
            let image = (image.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?.i(0)?;
            if idx == input.num_samples - 1 {
                save_image(&image, "./image.png").unwrap();
            }
            save_tensor_to_file(&image, "tensor4")?;

            res = convert_to_image(&image)?;
        }

        let time_to_generate = start_gen.elapsed().as_secs_f64();

        Ok(StableDiffusionOutput {
            image_data: res.0,
            height: res.1,
            width: res.2,
            input_tokens: num_input_tokens,
            time_to_generate,
        })
    }
}

impl ModelType {
    /// The unet file specifier
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

    /// The actual vae file specifier
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

    /// The actual clip model file specifier
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

    /// The actual clip2 model file specifier
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
/// Model's file
enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}

impl ModelFile {
    /// Get the actual file path, after downloading
    /// the required model weights/tokenizer
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
    /// Performs text embeddings
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
    ) -> Result<(Tensor, usize), ModelError> {
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
        let num_input_tokens = tokens.len();
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
        Ok((text_embeddings, num_input_tokens))
    }

    /// Pre-processes image
    fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> Result<Tensor, ModelError> {
        let img = image::ImageReader::open(path)?.decode()?;
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn test_stable_diffusion_model_interface() {
        let api_key = "".to_string();
        let cache_dir: PathBuf = "./test_sd_cache_dir/".into();
        let model_id = "stable_diffusion_v1-5".to_string();
        let dtype = "f32".to_string();
        let revision = "".to_string();
        let device_id = 0;
        let use_flash_attention = false;
        let config = ModelConfig::new(
            model_id,
            dtype.clone(),
            revision,
            vec![device_id],
            use_flash_attention,
        );
        let load_data = StableDiffusion::fetch(api_key, cache_dir.clone(), config)
            .expect("Failed to fetch mamba model");

        println!("model device = {:?}", load_data.device);
        let should_be_device = device(device_id).unwrap();
        if should_be_device.is_cpu() {
            assert!(load_data.device.is_cpu());
        } else if should_be_device.is_cuda() {
            assert!(load_data.device.is_cuda());
        } else if should_be_device.is_metal() {
            assert!(load_data.device.is_metal());
        } else {
            panic!("Invalid device")
        }

        assert_eq!(load_data.use_flash_attention, use_flash_attention);
        assert_eq!(load_data.model_type, ModelType::StableDiffusionV1_5);

        let should_be_dtype = DType::from_str(&dtype).unwrap();
        assert_eq!(load_data.dtype, should_be_dtype);

        let (stream_tx, _) = mpsc::channel(1);
        let mut model = StableDiffusion::load(load_data, stream_tx).expect("Failed to load model");

        if should_be_device.is_cpu() {
            assert!(model.device.is_cpu());
        } else if should_be_device.is_cuda() {
            assert!(model.device.is_cuda());
        } else if should_be_device.is_metal() {
            assert!(model.device.is_metal());
        } else {
            panic!("Invalid device")
        }

        assert_eq!(model.dtype, should_be_dtype);
        assert_eq!(model.model_type, ModelType::StableDiffusionV1_5);

        let prompt = "A portrait of a flying cat: ".to_string();
        let uncond_prompt = "".to_string();
        let random_seed = 42;

        let input = StableDiffusionInput {
            prompt: prompt.clone(),
            uncond_prompt,
            height: None,
            width: None,
            random_seed: Some(random_seed),
            n_steps: None,
            num_samples: 1,
            model: ModelType::StableDiffusionV1_5,
            guidance_scale: None,
            img2img: None,
            img2img_strength: 1.0,
        };
        println!("Running inference on input: {:?}", input);
        let output = model.run(input).expect("Failed to run inference");
        println!("{:?}", output.image_data);

        assert_eq!(output.height, 512);
        assert_eq!(output.width, 512);

        std::fs::remove_dir_all(cache_dir).unwrap();
        std::fs::remove_file("tensor1").unwrap();
        std::fs::remove_file("tensor2").unwrap();
        std::fs::remove_file("tensor3").unwrap();
        std::fs::remove_file("tensor4").unwrap();

        tokio::time::sleep(Duration::from_secs(5)).await; // give 5 seconds to look at the generated image

        std::fs::remove_file("./image.png").unwrap();
    }
}
