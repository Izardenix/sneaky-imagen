import os
import torch
import runpod
import base64
import io
from diffusers import FluxPipeline, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from PIL import Image

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

MODELS_DIR = "/models"
CHECKPOINT_DIR = f"{MODELS_DIR}/checkpoints"

# Global pipeline
pipe = None
pipeline_info = {}

def load_models():
    global pipe, pipeline_info

    print("Loading checkpoint only (no LoRA, no VAE)...")

    checkpoint_path = None
    if os.path.exists(CHECKPOINT_DIR):
        files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.safetensors')]
        if files:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, files[0])
            print(f"Found checkpoint: {checkpoint_path}")

    if not checkpoint_path:
        print("❌ No checkpoint found!")
        return False

    model_type = os.environ.get("MODEL_TYPE", "SDXL")

    print(f"Loading {model_type} pipeline...")

    try:
        if model_type == "Flux":
            pipe = FluxPipeline.from_single_file(
                checkpoint_path,
                torch_dtype=dtype
            )
        else:
            pipe = StableDiffusionXLPipeline.from_single_file(
                checkpoint_path,
                torch_dtype=dtype
            )

        pipe = pipe.to(device)

        if device == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()

        pipeline_info["model_type"] = model_type
        pipeline_info["loaded"] = True

        print("✅ Model loaded successfully (checkpoint only)")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


# Load at startup
load_models()


def handler(job):
    global pipe

    job_input = job["input"]

    if not pipeline_info.get("loaded"):
        return {"error": "Pipeline not loaded"}

    prompt = job_input.get(
        "prompt",
        "a beautiful landscape with mountains and a lake, highly detailed"
    )
    negative_prompt = job_input.get(
        "negative_prompt",
        "blurry, low quality, distorted, ugly"
    )

    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    steps = job_input.get("steps", 30)
    cfg_scale = job_input.get("cfg_scale", 5.5)
    seed = job_input.get("seed", None)
    scheduler_type = job_input.get("scheduler", "Euler a")

    output_format = job_input.get("output_format", "JPEG").upper()
    output_quality = int(job_input.get("output_quality", 90))

    # Scheduler
    if scheduler_type == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++"
        )

    # Seed
    if seed:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        seed = torch.seed()
        generator = torch.Generator(device=device).manual_seed(seed)

    print(f"Generating: {prompt[:60]}... | Seed={seed}")

    try:
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        )

        image = output.images[0]

        # Convert to base64
        buffered = io.BytesIO()

        if output_format == "PNG":
            image.save(buffered, format="PNG")
        else:
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(buffered, format="JPEG", quality=output_quality)

        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 🔥 IMPORTANT: reduce VRAM issues
        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "image": img_str,
            "image_format": output_format.lower(),
            "seed": seed,
            "params": {
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg_scale,
                "model": pipeline_info["model_type"]
            }
        }

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
