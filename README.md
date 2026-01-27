# PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models

[![Weights](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf)
[![Demo](https://img.shields.io/badge/🎮-Demo-green)](https://research.nvidia.com/labs/adlr/personaplex/)
[![Discord](https://img.shields.io/badge/Discord-Join-purple?logo=discord)](https://discord.gg/5jAXrrbwRb)

**🚀 New to PersonaPlex? See [QUICKSTART.md](QUICKSTART.md) for a fast setup guide!**

**🎨 Developing custom UI? See [FRONTEND_DEVELOPMENT.md](FRONTEND_DEVELOPMENT.md) for frontend development workflow!**

PersonaPlex is a real-time, full-duplex speech-to-speech conversational model that enables persona control through text-based role prompts and audio-based voice conditioning. Trained on a combination of synthetic and real conversations, it produces natural, low-latency spoken interactions with a consistent persona. PersonaPlex is based on the [Moshi](https://arxiv.org/abs/2410.00037) architecture and weights.

<p align="center">
  <img src="assets/architecture_diagram.png" alt="PersonaPlex Model Architecture">
  <br>
  <em>PersonaPlex Architecture</em>
</p>

## Usage

### Prerequisites

Install the [Opus audio codec](https://github.com/xiph/opus) development library:
```bash
# Ubuntu/Debian
sudo apt install libopus-dev

# Fedora/RHEL
sudo dnf install opus-devel

# macOS
brew install opus
```

### Installation

Download this repository and set up the environment:

#### Option 1: Using Conda (Recommended)
```bash
# Create and activate conda environment
conda create -n personaplex python=3.10 -y
conda activate personaplex

# Install the moshi package in editable mode (for development)
cd moshi
pip install -e .
cd ..
```

**Note:** Use `pip install -e .` (editable mode) during development so code changes are immediately reflected without reinstalling.

#### Option 2: For Blackwell GPUs (RTX 50 series)
Blackwell GPUs require PyTorch with CUDA 13.0+ support. Install PyTorch first, then the moshi package:
```bash
# Create and activate conda environment
conda create -n personaplex python=3.10 -y
conda activate personaplex

# Install PyTorch with CUDA 13.0+ support FIRST (required for Blackwell)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Then install the moshi package (will use existing PyTorch)
pip install moshi/.
```

See https://github.com/NVIDIA/personaplex/issues/2 for more details on Blackwell GPU support.


### Accept Model License

Log in to your Huggingface account and accept the PersonaPlex model license [here](https://huggingface.co/nvidia/personaplex-7b-v1).

Then set up your Huggingface authentication using one of these methods:

**Option 1: .env file (Recommended)**
```bash
# Copy the template and add your token
cp .env.example .env
# Edit .env and replace 'your_token_here' with your actual token
```

**Option 2: Environment variable**
```bash
export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

**Option 3: Hugging Face CLI**
```bash
pip install huggingface_hub
huggingface-cli login
```


### Launch Server

**IMPORTANT: First activate the conda environment:**
```bash
conda activate personaplex
```

#### Smart Auto-Detection (Recommended)

The server **automatically detects and serves your custom UI** if `client/dist` exists:
```bash
# If client/dist exists, it will be used automatically!
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

**Auto-detection behavior:**
1. Checks if `client/dist` exists in your project
2. If yes → serves custom UI from `./client/dist`
3. If no → downloads and serves default UI from HuggingFace

**How to verify which UI is loading:**
Check the server logs:
- **Custom UI (auto-detected)**:
  ```
  Found custom UI at .../client/dist, using it instead of default
  static_path = /home/.../personaplex-blackwell/client/dist
  ```
- **Default UI (no custom build)**:
  ```
  retrieving the static content
  static_path = /home/.../.cache/huggingface/.../dist
  ```

#### Manual Override (Optional)

You can still explicitly specify which UI to use with the `--static` flag:
```bash
# Force use of custom UI from specific directory
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --static client/dist

# Disable static serving entirely
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --static none
```


**CPU Offload:** If your GPU has insufficient memory, use the `--cpu-offload` flag to offload model layers to CPU. This requires the `accelerate` package (`pip install accelerate`):
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --cpu-offload
```

Access the Web UI from a browser at `localhost:8998` if running locally, otherwise look for the access link printed by the script:
```
Access the Web UI directly at https://11.54.401.33:8998
```

### Offline Evaluation

For offline evaluation use the offline script that streams in an input wav file and produces an output wav file from the captured output stream. The output file will be the same duration as the input file.

Add `--cpu-offload` to any command below if your GPU has insufficient memory (requires `accelerate` package). Or install cpu-only PyTorch for offline evaluation on pure CPU.

**Assistant example:**
```bash
python -m moshi.offline \
  --voice-prompt "NATF2.pt" \
  --input-wav "assets/test/input_assistant.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

**Service example:**
```bash
python -m moshi.offline \
  --voice-prompt "NATM1.pt" \
  --text-prompt "$(cat assets/test/prompt_service.txt)" \
  --input-wav "assets/test/input_service.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

## Voices

PersonaPlex supports a wide range of voices; we pre-package embeddings for voices that sound more natural and conversational (NAT) and others that are more varied (VAR). The fixed set of voices are labeled:
```
Natural(female): NATF0, NATF1, NATF2, NATF3
Natural(male):   NATM0, NATM1, NATM2, NATM3
Variety(female): VARF0, VARF1, VARF2, VARF3, VARF4
Variety(male):   VARM0, VARM1, VARM2, VARM3, VARM4
```

### Custom Voices

PersonaPlex supports **dynamic custom voice loading** - add new voices and they automatically appear in the Web UI without code changes!

#### Quick Start

**Step 1: Prepare your audio file**

Record a ~10 second WAV file of clear speech. Convert it to mono 24kHz format:
```bash
ffmpeg -i your_recording.wav -ac 1 -ar 24000 my_voice.wav
```

**Step 2: Copy to voices directory**

Copy the converted audio to the voices directory:
```bash
cp my_voice.wav ~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices/
```

**Step 3: Generate voice embeddings**

Run the offline script with `--save-voice-embeddings` to generate the `.pt` file:
```bash
python -m moshi.offline \
  --voice-prompt "my_voice.wav" \
  --save-voice-embeddings \
  --input-wav "assets/test/input_assistant.wav" \
  --output-wav "/tmp/test_output.wav" \
  --output-text "/tmp/test_output.json"
```

This creates `my_voice.pt` in the voices directory.

**Step 4: Use your custom voice**

**With the Web UI:** Restart the server and your custom voice automatically appears in the voice dropdown! Custom voices appear first in the list.
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

**With offline mode:** Use the `.pt` file directly:
```bash
python -m moshi.offline --voice-prompt "my_voice.pt" ...
```

#### Custom Voices Directory

For easier management, you can place custom voices in the `custom_voices/` directory:

```bash
# Create your custom voices directory (if it doesn't exist)
mkdir -p custom_voices

# Place voice files here
cp my_voice.wav custom_voices/
# Generate embeddings...
# The generated my_voice.pt will appear in the Web UI!
```

**Configure custom location (optional):**

Preferred method - add to your `.env` file:
```bash
CUSTOM_VOICE_DIR=/path/to/my/voices
```

Or use environment variable (temporary):
```bash
export CUSTOM_VOICE_DIR=/path/to/my/voices
```

#### Voice File Formats

- **`.pt` files**: Voice embeddings - these are the actual selectable voices in the Web UI
- **`.wav` files**: Source audio (24kHz mono) - used only to GENERATE the `.pt` embeddings

**Important:** Only `.pt` files appear in the voice selector dropdown. The `.wav` files are intermediate source files used during voice generation.

#### API Access

List all available voices programmatically:
```bash
curl http://localhost:8998/api/voices
```

Returns JSON with all voices, their types, and categories.

## Example Usage

### Auto-Detection
```bash
# Build frontend
cd client && npm run build && cd ..

# Server auto-detects - no flag needed!
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
# Logs: "Found custom UI at .../client/dist, using it instead of default"
```

### Custom Voices
```bash
# Add voice file
cp my_voice.wav custom_voices/

# Generate embeddings
python -m moshi.offline --voice-prompt "my_voice.wav" \
  --save-voice-embeddings --input-wav "assets/test/input_assistant.wav" --output-wav "/tmp/out.wav"

# Restart server - voice appears in UI automatically!
```

## Prompting Guide

The model is trained on synthetic conversations for a fixed assistant role and varying customer service roles.

### Assistant Role

The assistant role has the prompt:
```
You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.
```

Use this prompt for the QA assistant focused "User Interruption" evaluation category in [FullDuplexBench](https://arxiv.org/abs/2503.04721).

### Customer Service Roles

The customer service roles support a variety of prompts. Here are some examples for prompting style reference:
```
You work for CitySan Services which is a waste management and your name is Ayelen Lucero. Information: Verify customer name Omar Torres. Current schedule: every other week. Upcoming pickup: April 12th. Compost bin service available for $8/month add-on.
```
```
You work for Jerusalem Shakshuka which is a restaurant and your name is Owen Foster. Information: There are two shakshuka options: Classic (poached eggs, $9.50) and Spicy (scrambled eggs with jalapenos, $10.25). Sides include warm pita ($2.50) and Israeli salad ($3). No combo offers. Available for drive-through until 9 PM.
```
```
You work for AeroRentals Pro which is a drone rental company and your name is Tomaz Novak. Information: AeroRentals Pro has the following availability: PhoenixDrone X ($65/4 hours, $110/8 hours), and the premium SpectraDrone 9 ($95/4 hours, $160/8 hours). Deposit required: $150 for standard models, $300 for premium.
```

### Casual Conversations

The model is also trained on real conversations from the [Fisher English Corpus](https://catalog.ldc.upenn.edu/LDC2004T19) with LLM-labeled prompts for open-ended conversations. Here are some example prompts for casual conversations:
```
You enjoy having a good conversation.
```
```
You enjoy having a good conversation. Have a casual discussion about eating at home versus dining out.
```
```
You enjoy having a good conversation. Have an empathetic discussion about the meaning of family amid uncertainty.
```
```
You enjoy having a good conversation. Have a reflective conversation about career changes and feeling of home. You have lived in California for 21 years and consider San Francisco your home. You work as a teacher and have traveled a lot. You dislike meetings.
```
```
You enjoy having a good conversation. Have a casual conversation about favorite foods and cooking experiences. You are David Green, a former baker now living in Boston. You enjoy cooking diverse international dishes and appreciate many ethnic restaurants.
```

Use the prompt `You enjoy having a good conversation.` for the "Pause Handling", "Backchannel" and "Smooth Turn Taking" evaluation categories of FullDuplexBench.

## Generalization

Personaplex finetunes Moshi and benefits from the generalization capabilities of the underlying [Helium](https://kyutai.org/blog/2025-04-30-helium) LLM. Thanks to the broad training corpus of the backbone, we find that the model will respond plausibly to out-of-distribution prompts and lead to unexpected or fun conversations. We encourage experimentation with different prompts to test the model's emergent ability to handle scenarios outside its training distribution. As an inspiration we feature the following astronaut prompt in the WebUI:
```
You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor.
```

## Troubleshooting

For common issues and solutions, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md), including:
- Code changes not reflected when running server (editable install issue)
- Custom voices not appearing in Web UI
- Frontend build and development issues
- Environment and dependency problems

## License

The present code is provided under the MIT license. The weights for the models are released under the NVIDIA Open Model license.

## Citation

If you use PersonaPlex in your research, please cite our paper:
```bibtex
@article{roy2026personaplex,
  title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models},
  author={Roy, Rajarshi and Raiman, Jonathan and Lee, Sang-gil and Ene, Teodor-Dumitru and Kirby, Robert and Kim, Sungwon and Kim, Jaehyeon and Catanzaro, Bryan},
  year={2026}
}
```
