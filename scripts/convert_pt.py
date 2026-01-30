import torch
from moshi.models import loaders
from moshi.models.lm import LMGen
from huggingface_hub import hf_hub_download

device = "cuda"

voice_wav = "assets/test/pepper_mono.wav"

mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)

mimi = loaders.get_mimi(mimi_weight, device)
lm = loaders.get_moshi_lm(moshi_weight, device=device)
lm.eval()

lm_gen = LMGen(
    lm,
    device=device,
    sample_rate=mimi.sample_rate,
    frame_rate=mimi.frame_rate,
    save_voice_prompt_embeddings=True,
)

mimi.streaming_forever(1)
lm_gen.streaming_forever(1)

lm_gen.load_voice_prompt(voice_wav)
lm_gen._step_voice_prompt(mimi)

print("✔ pepper.pt generado correctamente")
