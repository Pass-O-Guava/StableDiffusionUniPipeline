# -----------------------------------
# Copyright 2023
# 版本：diffusers==0.20.2
# 优化：pipeline合二为一：StableDiffusionPipeline、StableDiffusionImg2ImgPipeline
# 优化：CFG并行加速，正、反prompt文本编码和Unet预测过程改成并行推理，支持单/双卡模式
# -----------------------------------

from concurrent.futures import ThreadPoolExecutor
# from diffusers import LMSDiscreteScheduler

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers, LMSDiscreteScheduler
from ...utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from ..pipeline_utils import DiffusionPipeline
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> from diffusers import StableDiffusionUniParallelPipeline

        # Double-GPUs parallel
        >>> pipe = StableDiffusionUniParallelPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        
        # Single-GPU parallel
        >>> pipe = StableDiffusionUniParallelPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", single_gpu_parallel=True)
        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

# copy from img2img
def preprocess(image):
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class StableDiffusionUniParallelPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    # 重写from_pretrained，模型加载实际代码逻辑在__init__，返回pipe实例
    def from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        single_gpu_parallel=False, #强制单GPU并行（默认关闭，有多卡优先双GPU并行）
    ):
        model = StableDiffusionUniParallelPipeline(model_path, torch_dtype, single_gpu_parallel)
        return model
        
    def __init__(
        self, 
        model_path, 
        torch_dtype=torch.float16, 
        single_gpu_parallel=False, #强制单GPU并行（默认关闭，有多卡优先双GPU并行）
    ):

        logger.warning("==> 🤔 The current version does not support safety_checker.")
        # super().__init__()
                
        # 无GPU直接抛错返回
        if torch.cuda.is_available() == False:
            raise Exception("GPU is not available!")
                
        # 加载到cpu的模型
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=torch_dtype)
        self.scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch_dtype)
        
        # vae模型加载到cuda:0
        self.output_device = "cuda:0"
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch_dtype)
        self.vae.to(self.output_device)
        self.dtype = torch_dtype
        
        # text_encoder、unet按不同模式加载
        def model_load(text_encoders, unets, torch_device):
            text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype)
            text_encoders.append(text_encoder.to(torch_device))
            unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch_dtype)
            unets.append(unet.to(torch_device))
            return text_encoders, unets
        
        # ------------------------------------------------------------------------------------------------
        # single_gpu：单GPU并行，text_encoder和unet2个副本部署在同一个GPU上，GPU显存大小需满足要求，否则OutOfMemory
        # double_gpu：双GPU并行，text_encoder和unet在每个GPU部署一个副本，至少单机2个GPU，否则自动降级为single_gpu
        # ------------------------------------------------------------------------------------------------
        
        text_encoders, unets = [], []
        for gpu_id in range(2):
            text_encoders, unets = model_load(
                text_encoders, 
                unets, 
                f"cuda:{gpu_id}" if (torch.cuda.device_count() > 1 and single_gpu_parallel == False) else "cuda:0"
            )
        
        print(f"text_encoders: {[en.device for en in text_encoders]}")
        print(f"unets: {[un.device for un in unets]}")
        
        self.text_encoders = text_encoders
        self.unets = unets
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=False)
        

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def encode_prompt(self, prompt, gpu_id: int = None,):

        # 线程调度，动态指定设备
        device =  self.text_encoders[gpu_id].device
            
        # textual inversion: procecss multi-vector tokens if necessary
        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        # 输入文本通过tokenizer转为令牌输入格式
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        # attention_mask ?
        if hasattr(self.text_encoders[gpu_id].config, "use_attention_mask") and self.text_encoders[gpu_id].config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        # text_encoder推理
        prompt_embeds = self.text_encoders[gpu_id](
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        if self.text_encoders[gpu_id] is not None:
            prompt_embeds_dtype = self.text_encoders[gpu_id].dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, self.num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * self.num_images_per_prompt, seq_len, -1)
        
        return prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # add from img2img, add 'strength'
    def check_inputs(
        self, prompt, height, width, callback_steps, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None,
        strength=None,          #+
        FLAG_IMG2IMG=False,     #+
    ):
            
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if FLAG_IMG2IMG:
            if strength < 0 or strength > 1:
                raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # copy from img2img
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    # add from img2img
    def prepare_latents(
        self, 
        batch_size, 
        num_channels_latents, 
        height, 
        width, 
        dtype, 
        device,
        generator=None, 
        latents=None,
        image=None,                 #+
        timestep=None,              #+
        num_images_per_prompt=None, #+
        FLAG_IMG2IMG=False,         #+
        ):
        if FLAG_IMG2IMG:
            if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
                raise ValueError(
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
                )

            image = image.to(device=device, dtype=dtype)
            
            batch_size = batch_size * num_images_per_prompt
            
            if image.shape[1] == 4:
                init_latents = image

            else:
                if isinstance(generator, list) and len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )

                elif isinstance(generator, list):
                    init_latents = [
                        self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                    ]
                    init_latents = torch.cat(init_latents, dim=0)
                else:
                    init_latents = self.vae.encode(image).latent_dist.sample(generator)
                    
                init_latents = self.vae.config.scaling_factor * init_latents
            
            if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
                # expand init_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                additional_image_per_prompt = batch_size // init_latents.shape[0]
                init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                init_latents = torch.cat([init_latents], dim=0)

            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            # get latents
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
            latents = init_latents
        else:
            shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    # gpu_id上的unet推理计算第t步的噪声预测结果
    def unet_pred(self, latent_model_input, t, text_embeddings, gpu_id):
        
        # scheduler.scale_model_input
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        
        # unet推理（输入为：1、潜变量转换后，2、时间步t，3、文本编码的sample采样）
        with torch.no_grad():
            noise_pred = self.unets[gpu_id](
                latent_model_input.to(self.unets[gpu_id].device), 
                t,
                encoder_hidden_states=text_embeddings.to(self.unets[gpu_id].device)
            ).sample #注意有个sample
        
        # 得到该步噪声预测结果
        return noise_pred
    
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,                           #+
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        strength: float = 0.8,              #+
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        
        FLAG_IMG2IMG = True if image else False

        # 0. Default height and width to unet
        height = height or self.unets[0].config.sample_size * self.vae_scale_factor
        width = width or self.unets[0].config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        if FLAG_IMG2IMG:
            self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, strength, FLAG_IMG2IMG=True)
        else:
            self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, strength=None, FLAG_IMG2IMG=False)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = self._execution_device
        
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        if do_classifier_free_guidance:
            # 对CFG输入[pos,neg]两分支并行计算
            inputs = [prompt, negative_prompt] if negative_prompt else [prompt, ""]
        else:
            inputs = [prompt]
        
        self.num_images_per_prompt = num_images_per_prompt
        
        # 开启两个线程计算有/无条件约束下的文本编码
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_embeddings = [executor.submit(self.encode_prompt, inputs[i], i) for i in range(len(inputs))]
        prompt_embeds = [f.result() for f in future_embeddings]
        
        # 4. Prepare timesteps
        if FLAG_IMG2IMG:
            # Preprocess image
            image = self.image_processor.preprocess(image)

            self.scheduler.set_timesteps(num_inference_steps)
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        else:
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        if FLAG_IMG2IMG:
            latents = self.prepare_latents(
                batch_size=batch_size,                          #change
                num_channels_latents=None,
                height=None,
                width=None,
                dtype=prompt_embeds[0].dtype,
                device=self.vae.device,
                generator=generator,
                latents=None,
                image=image,                                    #+
                timestep=latent_timestep,                       #+
                num_images_per_prompt=num_images_per_prompt,    #+
                FLAG_IMG2IMG=True,                              #+
            )
        else:
            num_channels_latents = self.unets[0].config.in_channels
            latents = self.prepare_latents(
                batch_size=batch_size * num_images_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds[0].dtype,
                device=self.vae.device,
                generator=generator,
                latents=latents,
            )
            

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                if do_classifier_free_guidance:
                                        
                    # 多线程[有/无条件或正反prompt两个并行推理分支]
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        noise_pred = [executor.submit(
                                            self.unet_pred, 
                                            torch.cat([latents] * 1),  # latents: torch.Size([1, 4, 64, 64])
                                            t, 
                                            prompt_embeds[i], 
                                            i) for i in range(len(prompt_embeds))
                                        ]
                    
                    # 推理得到有/无条件约束下的噪声预测结果，并统一移动到vae所在cuda设备
                    [noise_pred_text, noise_pred_uncond] = [f.result().to(self.output_device) for f in noise_pred]
                    
                    # 将有条件和无条件约束下的噪声预测结果融合
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    if guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                    
                else:
                    # 单线程[只有prompt推理分支]    
                    noise_pred = self.unet_pred(latents, t, prompt_embeds[0], 0)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents.to(self.output_device), **extra_step_kwargs, return_dict=False)[0]
                
                # 更新潜变量（原潜变量+噪声预测变量，通过scheduler计算更新潜变量, sample x_t -> x_t-1）
                # latents = self.scheduler.step(noise_pred, t, latents.to(self.output_device)).prev_sample
      
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # vae解码到图片
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        has_nsfw_concept = None
        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
