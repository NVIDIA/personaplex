# PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models

[![Weights](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://arxiv.org/abs/2602.06053)
[![Demo](https://img.shields.io/badge/🎮-Demo-green)](https://research.nvidia.com/labs/adlr/personaplex/)
[![Discord](https://img.shields.io/badge/Discord-Join-purple?logo=discord)](https://discord.gg/5jAXrrbwRb)

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
```

On Apple Silicon Macs, use Nix to create a Python environment with the MLX
runtime and to build the Web UI:
```bash
nix-shell
./scripts/bootstrap-nix-macos.sh
./scripts/build-client.sh
```

### Installation

Download this repository and install with:
```bash
pip install moshi/.
```

Extra step for Blackwell based GPUs as suggested in (See https://github.com/NVIDIA/personaplex/issues/2):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```


### Accept Model License
Log in to your Huggingface account and accept the PersonaPlex model license [here](https://huggingface.co/nvidia/personaplex-7b-v1). <br>
Then set up your Huggingface authentication:
```bash
export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

### Launch Server

Launch server for live interaction (temporary SSL certs for https):
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

**CPU Offload:** If your GPU has insufficient memory, use the `--cpu-offload` flag to offload model layers to CPU. This requires the `accelerate` package (`pip install accelerate`):
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --cpu-offload
```

Access the Web UI from a browser at `localhost:8998` if running locally, otherwise look for the access links printed by the script:
```
Serving Web UI at https://localhost:8998
Serving Web UI on LAN at https://<LAN_IP>:8998
```

On Apple Silicon Macs, `--device auto` resolves to MPS and starts the MLX
runtime when `model-mlx/` is available. You can also make it explicit with
`--device mps`:
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --device auto
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --device mps
```

If `./model` exists, the server uses it before Hugging Face downloads:
`model/model.safetensors`, `model/tokenizer.safetensors`, and `model/voices`.
The text tokenizer is a separate SentencePiece file; put
`tokenizer_spm_32k_3.model` in `./model` or export `HF_TOKEN` after accepting the
model license so it can be downloaded.

### Convert Weights for MLX

Convert a standard PersonaPlex/Moshi checkpoint directory or `model.safetensors`
file into an MLX bundle:
```bash
python -m moshi.convert_mlx \
  --input model \
  --output-dir model-mlx \
  --force \
  --moshi-mlx-native on
```

The converter writes both `model-mlx/model.safetensors` and the native
`model-mlx/model.moshi_mlx.safetensors` used by `moshi_mlx`. It also writes the
MLX runtime `config.json`, copies local side assets such as tokenizer/Mimi files
and `voices/`, then validates the standard safetensors file with
`mlx.core.load`. Use `--self-test` to verify the conversion path without the full
model:
```bash
python -m moshi.convert_mlx --self-test
```

After conversion, this starts the local MLX model on macOS:
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --device mps
```

### Offline Evaluation

For offline evaluation use the offline script that streams in an input wav file and produces an output wav file from the captured output stream. The output file will be the same duration as the input file.

Add `--cpu-offload` to any command below if your GPU has insufficient memory (requires `accelerate` package). Or install cpu-only PyTorch for offline evaluation on pure CPU.

**Assistant example:**
```bash
HF_TOKEN=<TOKEN> \
python -m moshi.offline \
  --voice-prompt "NATF2.pt" \
  --input-wav "assets/test/input_assistant.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

**Service example:**
```bash
HF_TOKEN=<TOKEN> \
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

## License

The present code is provided under the MIT license. The weights for the models are released under the NVIDIA Open Model license.

## Citation

If you use PersonaPlex in your research, please cite our paper:
```bibtex
@misc{roy2026personaplexvoicerolecontrol,
      title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models}, 
      author={Rajarshi Roy and Jonathan Raiman and Sang-gil Lee and Teodor-Dumitru Ene and Robert Kirby and Sungwon Kim and Jaehyeon Kim and Bryan Catanzaro},
      year={2026},
      eprint={2602.06053},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06053}, 
}
```
