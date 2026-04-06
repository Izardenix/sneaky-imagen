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
    num_images = int(job_input.get("num_images", 1))
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
    num_images = int(job_input.get("num_images", 1))

    generators = []
    seeds = []

    if seed is not None:
        # 🔁 deterministic but varied (based on provided seed)
        import random
        random.seed(seed)

        for _ in range(num_images):
            s = random.randint(0, 2**32 - 1)
            seeds.append(s)
            gen = torch.Generator(device=device).manual_seed(s)
            generators.append(gen)
    else:
        # 🎲 fully random seeds
        for _ in range(num_images):
            s = torch.randint(0, 2**32 - 1, (1,)).item()
            seeds.append(s)
            gen = torch.Generator(device=device).manual_seed(s)
            generators.append(gen)

    print(f"Generating {num_images} images with seeds: {seeds}")

    try:
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generators,
            num_images_per_prompt=num_images
        )

        images = output.images

        # Convert all images to base64
        encoded_images = []

        for image in images:
            buffered = io.BytesIO()

            if output_format == "PNG":
                image.save(buffered, format="PNG")
            else:
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                image.save(buffered, format="JPEG", quality=output_quality)

            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            encoded_images.append(img_str)

        # 🔥 IMPORTANT: reduce VRAM issues
        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "images": encoded_images,   # ✅ now array
            "image_format": output_format.lower(),
            "seed": seed,
            "params": {
                "num_images": len(encoded_images),
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
