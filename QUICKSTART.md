# PersonaPlex Quick Start Guide

This guide provides the essential steps to get PersonaPlex running quickly.

## Prerequisites

1. Install [Opus audio codec](https://github.com/xiph/opus) development library:
   ```bash
   # Ubuntu/Debian
   sudo apt install libopus-dev

   # macOS
   brew install opus
   ```

2. Accept the [PersonaPlex model license](https://huggingface.co/nvidia/personaplex-7b-v1) on Hugging Face

## Installation

### Step 1: Create Conda Environment

```bash
# Create and activate conda environment
conda create -n personaplex python=3.10 -y
conda activate personaplex
```

### Step 2: Install Moshi Package

**For most GPUs:**
```bash
cd moshi
pip install -e .
cd ..
```

**For Blackwell GPUs (RTX 50 series):**
```bash
# Install PyTorch with CUDA 13.0+ support FIRST (required for Blackwell)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Then install moshi
cd moshi
pip install -e .
cd ..
```

### Step 3: Set Up Hugging Face Token

**Recommended: Use .env file (persists across sessions)**
```bash
# Copy the template and add your token
cp .env.example .env
# Edit .env and replace 'your_token_here' with your actual Hugging Face token
```

**Alternative: Use export (temporary, only for current session)**
```bash
export HF_TOKEN=your_token_here
```

## Running the Web UI

**CRITICAL: Always activate the conda environment first!**

```bash
# 1. Activate the environment
conda activate personaplex

# 2. Launch the server (automatically detects custom UI if it exists)
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"

# 3. Access the Web UI at: https://localhost:8998
```

### Smart Auto-Detection

The server now **automatically detects and uses your custom UI** if you've built it!

- If `client/dist` exists → Your custom UI is served automatically
- If `client/dist` doesn't exist → Default UI is downloaded from HuggingFace

**Verify which UI loaded** by checking the server logs:
- Custom UI: `Found custom UI at .../client/dist, using it instead of default`
- Default UI: `retrieving the static content` (downloads from HuggingFace)

### Building Custom UI (If Modified)

Only needed if you've changed the frontend code:

```bash
cd client
npm run build
cd ..

# Now start the server - it will auto-detect your custom build
conda activate personaplex
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

## Quick Command Reference

| Task | Command |
|------|---------|
| Activate environment | `conda activate personaplex` |
| Start Web UI | `SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"` |
| Start with CPU offload | `SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --cpu-offload` |
| Start with local frontend | `SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --static client/dist` |

## Troubleshooting

**Error: "No module named 'moshi'"**
- Solution: Activate the conda environment: `conda activate personaplex`

**Error: "Access denied" when downloading model**
- Solution: Accept the model license and set up your HF token in `.env` file (see Step 3)

For more issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Next Steps

- See [README.md](README.md) for detailed documentation
- Explore voice customization options
- Try different persona prompts
- Check out offline evaluation mode
