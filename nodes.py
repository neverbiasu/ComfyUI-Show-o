import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

# ComfyUI models directory
try:
    from folder_paths import folder_names_and_paths, models_dir as comfy_models_dir

    models_dir = comfy_models_dir
except ImportError:
    # Fallback if folder_paths not available
    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    folder_names_and_paths = {}
    comfy_models_dir = models_dir

# Register the Show-o model folder
if "showo" not in folder_names_and_paths:
    folder_names_and_paths["showo"] = (
        [os.path.join(comfy_models_dir, "show_o")],
        [".json", ".safetensors", ".pt", ".pth", ".bin"],
    )

# Set cache directory for transformers models - all in show_o folder
showo_cache_dir = os.path.join(comfy_models_dir, "show_o")
os.makedirs(showo_cache_dir, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = os.path.join(showo_cache_dir, "transformers")
os.environ["HF_HOME"] = os.path.join(showo_cache_dir, "huggingface")

# Import transformers modules (required)
try:
    from transformers import AutoTokenizer, CLIPImageProcessor
except ImportError as e:
    raise ImportError(
        f"transformers library is required but not installed: {e}. Please install it with: pip install transformers"
    )

# Import models
from .models import Showo, MAGVITv2, CLIPVisionTower, get_mask_chedule

# Import training modules
from .training.prompting_utils import (
    UniversalPrompting,
    create_attention_mask_predict_next,
    create_attention_mask_for_mmu,
    create_attention_mask_for_mmu_vit,
)
from .training.utils import image_transform, get_config

# Import omegaconf for config handling
try:
    from omegaconf import OmegaConf
except ImportError as e:
    raise ImportError(
        f"omegaconf library is required but not installed: {e}. Please install it with: pip install omegaconf"
    )

# Import llava
from .llava.llava import conversation as conversation_lib

# Set flag to indicate modules are available
SHOWO_MODULES_AVAILABLE = True

# Global constants
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
SYSTEM_PROMPT_LEN = 28


def get_config_for_comfyui(config_path=None, **cli_overrides):
    """
    Simulate the behavior of get_config() function for ComfyUI environment
    Reference: get_config() implementation in training/utils.py
    """
    # Use default demo config if no config file path is specified
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "configs", "showo_demo.yaml"
        )

    # Simulate cli_conf = OmegaConf.from_cli()
    # In ComfyUI environment, manually create cli_conf with config path and other override parameters
    cli_conf = OmegaConf.create({"config": config_path, **cli_overrides})

    # Load YAML configuration
    yaml_conf = OmegaConf.load(cli_conf.config)

    # Merge configurations
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def get_vq_model_class(model_type: str):
    """Get VQ model class by type"""
    if not SHOWO_MODULES_AVAILABLE:
        raise ImportError(
            "Show-o modules are not available. Please check installation."
        )
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def extract_model_components(showo_model):
    """Extract model components from bundle for convenience"""
    return (
        showo_model["showo_model"],
        showo_model["vq_model"],
        showo_model["tokenizer"],
        showo_model["uni_prompting"],
        showo_model["clip_vision"],
        torch.device(showo_model["device"]),
        showo_model["dtype"],
    )


class ShowoModelLoader:
    """
    Load Show-o model, VQ model, and related components
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["show-o", "show-o2"], {"default": "show-o"}),
                "model_path": ("STRING", {"default": "showlab/show-o"}),
                "vq_model_path": ("STRING", {"default": "showlab/magvitv2"}),
                "llm_model_path": ("STRING", {"default": "microsoft/phi-1_5"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION",),
            },
        }

    RETURN_TYPES = ("SHOWO_MODEL",)
    RETURN_NAMES = ("showo_model",)
    FUNCTION = "load_model"
    CATEGORY = "Show-o"

    @classmethod
    def VALIDATE_INPUTS(cls, model_path, vq_model_path, llm_model_path, **kwargs):
        if not model_path or len(model_path.strip()) == 0:
            return "Model path cannot be empty"
        if not vq_model_path or len(vq_model_path.strip()) == 0:
            return "VQ model path cannot be empty"
        if not llm_model_path or len(llm_model_path.strip()) == 0:
            return "LLM model path cannot be empty"
        return True

    def load_model(
        self,
        model_version: str,
        model_path: str,
        vq_model_path: str,
        llm_model_path: str,
        device: str,
        precision: str,
        clip_vision=None,
    ):
        """Load Show-o model and components"""  # Device selection
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        device_obj = torch.device(device)

        # Set precision
        if precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        try:
            # Set cache directory for this specific load - all under show_o folder
            cache_dir = os.path.join(comfy_models_dir, "show_o", model_version)
            os.makedirs(cache_dir, exist_ok=True)

            # Load tokenizer with custom cache directory
            tokenizer = AutoTokenizer.from_pretrained(
                llm_model_path, padding_side="left", cache_dir=cache_dir
            )

            # Initialize universal prompting
            uni_prompting = UniversalPrompting(
                tokenizer,
                max_text_len=128,
                special_tokens=(
                    "<|soi|>",
                    "<|eoi|>",
                    "<|sov|>",
                    "<|eov|>",
                    "<|t2i|>",
                    "<|mmu|>",
                    "<|t2v|>",
                    "<|v2v|>",
                    "<|lvg|>",
                ),
                ignore_id=-100,
                cond_dropout_prob=0.1,
            )

            # Load VQ model with custom cache directory
            vq_model = (
                get_vq_model_class("magvitv2")
                .from_pretrained(vq_model_path, cache_dir=cache_dir)
                .to(device_obj)
            )
            vq_model.requires_grad_(False)
            vq_model.eval()

            if precision != "fp32" and device == "cuda":
                vq_model = vq_model.to(
                    dtype
                )  # Load Show-o main model with custom cache directory
            # Use torch_dtype to avoid meta tensor issues
            print(f"Loading Show-o model from {model_path}...")
            model_kwargs = {
                "cache_dir": cache_dir,
                "torch_dtype": dtype if device == "cuda" else torch.float32,
                "device_map": None,  # Don't use device_map to avoid meta tensors
                "low_cpu_mem_usage": False,  # Disable to avoid meta tensors
            }

            try:
                showo_model = Showo.from_pretrained(model_path, **model_kwargs)
                print("✅ Show-o model loaded successfully")
            except Exception as model_load_error:
                print(
                    f"⚠️ Failed to load with torch_dtype, trying without: {model_load_error}"
                )
                # Fallback: load without torch_dtype
                model_kwargs.pop("torch_dtype", None)
                showo_model = Showo.from_pretrained(model_path, **model_kwargs)
                print("✅ Show-o model loaded with fallback method")

            showo_model = showo_model.to(device_obj)
            showo_model.eval()

            # Ensure model precision is consistent - this is crucial for attention mechanisms
            if precision != "fp32" and device == "cuda":
                # Convert the entire model to the specified dtype
                showo_model = showo_model.to(dtype)
                # Also ensure all parameters and buffers are in the correct dtype
                for param in showo_model.parameters():
                    if (
                        param.dtype != torch.long and param.dtype != torch.int
                    ):  # Don't convert integer tensors
                        param.data = param.data.to(dtype)
                for buffer in showo_model.buffers():
                    if (
                        buffer.dtype != torch.long and buffer.dtype != torch.int
                    ):  # Don't convert integer tensors
                        buffer.data = buffer.data.to(dtype)

            # Create model bundle
            model_bundle = {
                "version": model_version,
                "showo_model": showo_model,
                "vq_model": vq_model,
                "tokenizer": tokenizer,
                "uni_prompting": uni_prompting,
                "clip_vision": clip_vision,
                "device": device,
                "dtype": dtype,
                "cache_dir": cache_dir,
            }

            print(
                f"Show-o {model_version} models loaded successfully on {device} with {precision} precision"
            )
            print(f"Models cached in: {cache_dir}")

            return (model_bundle,)

        except Exception as e:
            raise RuntimeError(f"Failed to load Show-o models: {str(e)}")


class ShowoTextToImage:
    """
    Generate images from text using Show-o model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "showo_model": ("SHOWO_MODEL",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "placeholder": "Enter your prompt here..."},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.5},
                ),
                "generation_timesteps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 100, "step": 1},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "resolution": ([256, 512], {"default": 256}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1},
                ),
                "mask_schedule": (
                    ["cosine", "linear", "sigmoid"],
                    {"default": "cosine"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "Show-o"

    @classmethod
    def VALIDATE_INPUTS(
        cls, prompt, guidance_scale, generation_timesteps, batch_size, **kwargs
    ):
        if not prompt or len(prompt.strip()) == 0:
            return "Prompt cannot be empty"
        if not (1.0 <= guidance_scale <= 20.0):
            return "Guidance scale must be between 1.0 and 20.0"
        if not (1 <= generation_timesteps <= 100):
            return "Generation timesteps must be between 1 and 100"
        if not (1 <= batch_size <= 8):
            return "Batch size must be between 1 and 8"
        return True

    def generate(
        self,
        showo_model,
        prompt: str,
        guidance_scale: float,
        generation_timesteps: int,
        batch_size: int,
        resolution: int,
        seed: int = -1,
        temperature: float = 1.0,
        mask_schedule: str = "cosine",
    ):
        """Generate images from text prompt"""

        # Extract components from pipeline
        (
            showo_model_components,
            vq_model,
            tokenizer,
            uni_prompting,
            clip_vision,
            device_obj,
            dtype,
        ) = extract_model_components(showo_model)
        config = get_config_for_comfyui(
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            generation_timesteps=generation_timesteps,
            mode="t2i",
        )
        config.training.batch_size = batch_size
        config.training.guidance_scale = guidance_scale
        config.training.generation_timesteps = generation_timesteps
        config.dataset.params.resolution = resolution  # Set generation resolution

        # Calculate VQ token count based on resolution
        # Show-o uses 16x downsampling, so token count = (resolution/16)^2
        vq_resolution = resolution // 16
        num_vq_tokens = vq_resolution * vq_resolution
        config.model.showo.num_vq_tokens = num_vq_tokens

        # Set seed if provided
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.cuda.manual_seed_all(seed)

        mask_token_id = showo_model_components.config.mask_token_id

        try:
            # Calculate correct token count based on resolution
            vq_resolution = resolution // 16
            num_vq_tokens = vq_resolution * vq_resolution  # Prepare prompts
            prompts = [prompt.strip()] * batch_size

            # Initialize image tokens as mask tokens
            image_tokens = (
                torch.ones(
                    (batch_size, num_vq_tokens), dtype=torch.long, device=device_obj
                )
                * mask_token_id
            )

            # Build input sequence
            input_ids, _ = uni_prompting((prompts, image_tokens), "t2i_gen")

            # Ensure input_ids are on the correct device and dtype (long for token IDs)
            input_ids = input_ids.to(device_obj, dtype=torch.long)

            # Build attention mask
            if guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(
                    ([""] * len(prompts), image_tokens), "t2i_gen"
                )
                # Ensure uncond_input_ids are on the correct device
                uncond_input_ids = uncond_input_ids.to(device_obj, dtype=torch.long)

                attention_mask = create_attention_mask_predict_next(
                    torch.cat([input_ids, uncond_input_ids], dim=0),
                    pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                    soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                    eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                    rm_pad_in_image=True,
                )
                # Ensure attention mask is on correct device and dtype (bool or float depending on model)
                attention_mask = attention_mask.to(device_obj)
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    attention_mask = attention_mask.to(dtype)
            else:
                attention_mask = create_attention_mask_predict_next(
                    input_ids,
                    pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                    soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                    eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                    rm_pad_in_image=True,
                )  # Ensure attention mask is on correct device and dtype
                attention_mask = attention_mask.to(device_obj)
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    attention_mask = attention_mask.to(dtype)
                uncond_input_ids = None
            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule_fn = get_mask_chedule(schedule, **args)
            else:
                mask_schedule_fn = get_mask_chedule(
                    config.training.get("mask_schedule", "cosine")
                )

            # Generate tokens
            with torch.no_grad():
                gen_token_ids = showo_model_components.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule_fn,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )
            gen_token_ids = torch.clamp(
                gen_token_ids, max=config.model.showo.codebook_size - 1, min=0
            )

            # VQ decoding
            vq_resolution = resolution // 16
            images = vq_model.decode_code(
                gen_token_ids, shape=(vq_resolution, vq_resolution)
            )

            # Convert to ComfyUI format [B, H, W, C] and normalize to [0, 1]
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images = images.permute(0, 2, 3, 1).cpu().float()

            return (images,)

        except Exception as e:
            raise RuntimeError(f"Image generation failed: {str(e)}")


class ShowoImageCaptioning:
    """
    Generate captions for images using Show-o model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "showo_model": ("SHOWO_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "question": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Ask a question about the image (leave empty for automatic captioning)",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 128, "min": 1, "max": 512, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1},
                ),
                "top_k": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption_image"
    CATEGORY = "Show-o"

    @classmethod
    def VALIDATE_INPUTS(cls, image, max_new_tokens, **kwargs):
        if image is None:
            return "Image input is required"
        if not (1 <= max_new_tokens <= 512):
            return "Max new tokens must be between 1 and 512"
        return True

    def caption_image(
        self,
        showo_model,
        image,
        question: str = "",
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 1,
    ):
        """Generate caption or answer question about image"""  # Extract components from bundle
        (
            showo_model_components,
            vq_model,
            tokenizer,
            uni_prompting,
            clip_vision,
            device_obj,
            dtype,
        ) = extract_model_components(showo_model)

        try:
            # Convert ComfyUI image format [B, H, W, C] to [B, C, H, W]
            if len(image.shape) == 4:
                image_tensor = image[0]  # Take first image if batch
            else:
                image_tensor = image

            # Convert to PIL for processing
            image_pil = Image.fromarray(
                (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            )  # Transform image to model format
            image_transformed = (
                image_transform(image_pil, resolution=256).to(device_obj).unsqueeze(0)
            )

            # VQ encode image
            image_tokens = vq_model.get_code(image_transformed) + len(
                uni_prompting.text_tokenizer
            )

            # Prepare question
            if not question.strip():
                question = "Describe this image in detail."

            # Build MMU input sequence (without CLIP for now)
            input_ids = uni_prompting.text_tokenizer(
                ["USER: \n" + question + " ASSISTANT:"]
            )["input_ids"]
            input_ids = torch.tensor(input_ids).to(device_obj)

            input_ids = torch.cat(
                [
                    (
                        torch.ones(input_ids.shape[0], 1)
                        * uni_prompting.sptids_dict["<|mmu|>"]
                    ).to(device_obj),
                    (
                        torch.ones(input_ids.shape[0], 1)
                        * uni_prompting.sptids_dict["<|soi|>"]
                    ).to(device_obj),
                    image_tokens,
                    (
                        torch.ones(input_ids.shape[0], 1)
                        * uni_prompting.sptids_dict["<|eoi|>"]
                    ).to(device_obj),
                    (
                        torch.ones(input_ids.shape[0], 1)
                        * uni_prompting.sptids_dict["<|sot|>"]
                    ).to(device_obj),
                    input_ids,
                ],
                dim=1,
            ).long()

            # Create attention mask
            attention_mask = create_attention_mask_for_mmu(
                input_ids.to(device_obj),
                eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
            )
            # Generate response
            with torch.no_grad():
                cont_toks_list = showo_model_components.mmu_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    eot_token=uni_prompting.sptids_dict["<|eot|>"],
                )

            # Decode response
            cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
            text = uni_prompting.text_tokenizer.batch_decode(
                cont_toks_list, skip_special_tokens=True
            )

            return (text[0],)

        except Exception as e:
            raise RuntimeError(f"Image captioning failed: {str(e)}")


class ShowoImageInpainting:
    """
    Inpaint images using Show-o model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "showo_model": ("SHOWO_MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Describe what should be in the masked area...",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.5},
                ),
                "generation_timesteps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 100, "step": 1},
                ),
                "resolution": ([256, 512], {"default": 256}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("inpainted_image",)
    FUNCTION = "inpaint"
    CATEGORY = "Show-o"

    @classmethod
    def VALIDATE_INPUTS(cls, image, mask, prompt, **kwargs):
        if image is None:
            return "Image input is required"
        if mask is None:
            return "Mask input is required"
        if not prompt or len(prompt.strip()) == 0:
            return "Inpainting prompt cannot be empty"
        return True

    def inpaint(
        self,
        showo_model,
        image,
        mask,
        prompt: str,
        guidance_scale: float,
        generation_timesteps: int,
        resolution: int,
        seed: int = -1,
        temperature: float = 1.0,
    ):
        """Inpaint image using mask and prompt"""

        # Extract components from bundle
        (
            showo_model_components,
            vq_model,
            tokenizer,
            uni_prompting,
            clip_vision,
            device_obj,
            dtype,
        ) = extract_model_components(showo_model)
        config = get_config_for_comfyui(
            # Override configuration parameters, simulating command line arguments
            batch_size=1,  # inpainting usually processes single image
            guidance_scale=guidance_scale,
            generation_timesteps=generation_timesteps,
            mode="inpainting",
        )
        config.training.batch_size = 1
        config.training.guidance_scale = guidance_scale
        config.training.generation_timesteps = generation_timesteps
        config.dataset.params.resolution = resolution  # Set generation resolution

        # Calculate VQ token count based on resolution
        vq_resolution = resolution // 16
        num_vq_tokens = vq_resolution * vq_resolution
        config.model.showo.num_vq_tokens = num_vq_tokens

        # Set seed if provided
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        device = device_obj
        mask_token_id = showo_model_components.config.mask_token_id

        try:
            # Convert ComfyUI formats
            if len(image.shape) == 4:
                image_tensor = image[0]  # Take first image if batch            else:
                image_tensor = image
            image_pil = Image.fromarray(
                (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            )
            inpainting_image = (
                image_transform(image_pil, resolution=resolution)
                .to(device)
                .unsqueeze(0)
            )

            # Process mask
            if len(mask.shape) == 3:
                mask_tensor = mask.unsqueeze(0)  # Add batch dimension            else:
                mask_tensor = mask

            # Resize mask to VQ resolution
            vq_resolution = resolution // 16
            inpainting_mask = F.interpolate(
                mask_tensor.unsqueeze(1), size=vq_resolution, mode="bicubic"
            ).squeeze(1)
            inpainting_mask[inpainting_mask < 0.5] = 0
            inpainting_mask[inpainting_mask >= 0.5] = 1
            inpainting_mask = inpainting_mask.reshape(1, -1).to(torch.bool).to(device)

            # VQ encode image
            inpainting_image_tokens = vq_model.get_code(inpainting_image) + len(
                uni_prompting.text_tokenizer
            )

            # Apply mask to tokens
            inpainting_image_tokens[inpainting_mask] = (
                mask_token_id  # Build input sequence
            )
            prompts = [prompt.strip()]
            input_ids, _ = uni_prompting((prompts, inpainting_image_tokens), "t2i_gen")

            # Ensure input_ids are on the correct device and dtype
            input_ids = input_ids.to(device_obj, dtype=torch.long)

            # Build attention mask
            if guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(
                    ([""], inpainting_image_tokens), "t2i_gen"
                )
                # Ensure uncond_input_ids are on the correct device
                uncond_input_ids = uncond_input_ids.to(device_obj, dtype=torch.long)

                attention_mask = create_attention_mask_predict_next(
                    torch.cat([input_ids, uncond_input_ids], dim=0),
                    pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                    soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                    eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                    rm_pad_in_image=True,
                )
                # Ensure attention mask is on correct device and dtype
                attention_mask = attention_mask.to(device_obj)
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    attention_mask = attention_mask.to(dtype)
            else:
                attention_mask = create_attention_mask_predict_next(
                    input_ids,
                    pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                    soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                    eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                    rm_pad_in_image=True,
                )  # Ensure attention mask is on correct device and dtype
                attention_mask = attention_mask.to(device_obj)
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    attention_mask = attention_mask.to(dtype)
                uncond_input_ids = None
            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule_fn = get_mask_chedule(schedule, **args)
            else:
                mask_schedule_fn = get_mask_chedule(
                    config.training.get("mask_schedule", "cosine")
                )

            # Generate inpainted content
            with torch.no_grad():
                gen_token_ids = showo_model_components.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule_fn,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )
            gen_token_ids = torch.clamp(
                gen_token_ids, max=config.model.showo.codebook_size - 1, min=0
            )
            vq_resolution = resolution // 16
            images = vq_model.decode_code(
                gen_token_ids, shape=(vq_resolution, vq_resolution)
            )

            # Convert to ComfyUI format
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images = images.permute(0, 2, 3, 1).cpu().float()

            return (images,)

        except Exception as e:
            raise RuntimeError(f"Image inpainting failed: {str(e)}")


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ShowoModelLoader": ShowoModelLoader,
    "ShowoTextToImage": ShowoTextToImage,
    "ShowoImageCaptioning": ShowoImageCaptioning,
    "ShowoImageInpainting": ShowoImageInpainting,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShowoModelLoader": "Show-o Model Loader",
    "ShowoTextToImage": "Show-o Text to Image",
    "ShowoImageCaptioning": "Show-o Image Captioning",
    "ShowoImageInpainting": "Show-o Image Inpainting",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
