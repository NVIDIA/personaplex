import os
import torch
import asyncio
import argparse
from pathlib import Path
import sys

# Add moshi path
sys.path.append(os.path.abspath('moshi'))

from moshi.models import loaders
from sentencepiece import SentencePieceProcessor
from moshi.models.lm import LMGen

async def main():
    parser = argparse.ArgumentParser(description="Generate PersonaPlex Voice Embedding (.pt) from .wav")
    parser.add_argument("--input", type=str, required=True, help="Path to input .wav file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output .pt file")
    parser.add_argument("--weights", type=str, default="weights", help="Directory with model weights")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights_dir = Path(args.weights)
    
    print(f"🎬 Loading models on {device}...")
    mimi = loaders.get_mimi(weights_dir / loaders.MIMI_NAME, device)
    moshi_lm = loaders.get_moshi_lm(weights_dir / loaders.MOSHI_NAME, device=device, cpu_offload=True)
    
    tokenizer_path = weights_dir / loaders.TEXT_TOKENIZER_NAME
    text_tokenizer = SentencePieceProcessor(str(tokenizer_path))
    lm_gen = LMGen(moshi_lm, text_tokenizer)

    print(f"🧬 Extracting voice signature from: {args.input}")
    # 1. Load the wav file into the model's memory
    lm_gen.load_voice_prompt(args.input)

    # 2. Process the audio to generate the internal state (the 'embedding')
    # This step 'conditions' the model
    await lm_gen.step_system_prompts_async(mimi)

    # 3. Save the internal state (cache and embeddings) to a .pt file
    # This file is what you will use in the Notebook or Server
    state_to_save = {
        "embeddings": lm_gen.voice_prompt_embeddings.cpu(),
        "cache": lm_gen.voice_prompt_cache.cpu()
    }
    
    torch.save(state_to_save, args.output)
    print(f"✅ Success! Voice embedding saved as: {args.output}")
    print(f"💡 You can now use this file in the local inference notebook.")

if __name__ == "__main__":
    asyncio.run(main())
