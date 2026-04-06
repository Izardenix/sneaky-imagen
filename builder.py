import os
import requests
import re

# Configuration
CHECKPOINT_URL = "https://civitai.com/api/download/models/2778052"

# ❌ Disable LoRA & VAE completely
LORA_URLS = []
VAE_URL = None

# Optional: CivitAI token
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "e0f8cbac3f574543aca939a730ed67bc")

MODELS_DIR = "/models"
CHECKPOINT_DIR = f"{MODELS_DIR}/checkpoints"

def download_file(url, output_dir, token=None):
    os.makedirs(output_dir, exist_ok=True)

    if 'civitai.com' in url and token:
        separator = '&' if '?' in url else '?'
        url = f"{url}{separator}token={token}"

    print(f"Downloading from {url}...")

    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()

        content_disp = response.headers.get('content-disposition', '')
        filename_match = re.findall(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disp)

        if filename_match and filename_match[0][0]:
            filename = filename_match[0][0].strip('\'"')
        else:
            filename = "model.safetensors"

        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            print(f"File {filename} already exists, skipping.")
            return output_path

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0 and downloaded % (1024*1024*100) == 0:
                        print(f"Downloaded {downloaded/1024/1024:.2f} MB")

        print(f"Downloaded {filename} to {output_path}")
        return output_path

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


if __name__ == "__main__":
    print("Starting build-time checkpoint download...")

    if CHECKPOINT_URL:
        download_file(CHECKPOINT_URL, CHECKPOINT_DIR, CIVITAI_TOKEN)

    print("Build complete (checkpoint only, no LoRA, no VAE).")
