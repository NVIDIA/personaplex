import os
from pathlib import Path
from huggingface_hub import hf_hub_download, configure_http_backend
import requests

# Bypass SSL Verification
def disabled_ssl_backend(*args, **kwargs):
    kwargs["verify"] = False
    return requests.Session().request(*args, **kwargs)

# Configuration
weights_dir = Path("weights")
weights_dir.mkdir(exist_ok=True)
repo_id = "nvidia/personaplex-7b-v1"

files_to_download = [
    "tokenizer_spm_32k_3.model",
    "model.safetensors",
    "tokenizer-e351c8d8-checkpoint125.safetensors"
]

print(f"🚀 Starting download of PersonaPlex weights to {weights_dir.absolute()} (SSL Disabled)...")

for filename in files_to_download:
    print(f"📥 Downloading {filename}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=weights_dir,
        local_dir_use_symlinks=False,
        force_download=True
    )

print("✨ All weights downloaded successfully!")
