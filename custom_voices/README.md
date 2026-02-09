# Custom Voices Directory

This directory is for storing your custom voice files. Any voice files (.pt or .wav) placed here will automatically appear in the PersonaPlex web interface voice selector.

## Quick Start

1. **Prepare your audio file** (10+ seconds of clear speech):
   ```bash
   ffmpeg -i your_recording.mp3 -ac 1 -ar 24000 your_voice.wav
   ```

2. **Generate voice embeddings**:
   ```bash
   # Copy the WAV file to the voices directory
   VOICES_DIR=$(python -c "from pathlib import Path; from huggingface_hub import snapshot_download; print(Path(snapshot_download(repo_id='nvidia/personaplex-7b-v1', allow_patterns=['voices/*'])) / 'voices')")
   cp your_voice.wav "$VOICES_DIR/"

   # Generate embeddings
   python -m moshi.offline \
     --voice-prompt "your_voice.wav" \
     --save-voice-embeddings \
     --input-wav "assets/test/input_assistant.wav" \
     --output-wav "/tmp/test_output.wav" \
     --output-text "/tmp/test_output.json"
   ```

3. **Restart the server** and your voice will appear in the dropdown!

## File Formats

- **`.pt` files**: Voice embeddings - these are the actual selectable voices that appear in the UI dropdown
- **`.wav` files**: Source audio recordings (24kHz mono) - used only to GENERATE the .pt embeddings, not selectable as voices

**Important**: Only `.pt` files appear in the voice selector dropdown. `.wav` files are intermediate source files used during voice generation.

## Configuration

By default, PersonaPlex looks for voices in:
1. HuggingFace cache: `~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices/`
2. Custom directory: `./custom_voices/` (this directory)

To use a different custom voices directory:

Add to your `.env` file (recommended):
```bash
CUSTOM_VOICE_DIR=/path/to/my/voices
```

Or use environment variable:
```bash
export CUSTOM_VOICE_DIR=/path/to/my/voices
```

## Voice Naming Convention

Pre-packaged voices follow this naming:
- `NATF*` = Natural Female
- `NATM*` = Natural Male
- `VARF*` = Variety Female
- `VARM*` = Variety Male

Custom voices (any other name) will appear first in the dropdown, followed by the categorized pre-packaged voices.

## API Access

You can list all available voices programmatically:
```bash
curl http://localhost:8998/api/voices
```

Returns:
```json
{
  "voices": [
    {"name": "your_voice.pt", "type": "embeddings", "category": "custom", "path": "..."},
    {"name": "NATF0.pt", "type": "embeddings", "category": "natural-female", "path": "..."},
    ...
  ],
  "count": 20
}
```

## Tips

- Use high-quality audio recordings (clear speech, minimal background noise)
- 10-30 seconds of audio is usually sufficient
- The voice will reflect the speaking style and characteristics of the input audio
- Experiment with different recordings to find the best voice for your use case

## Troubleshooting

If your custom voice doesn't appear:
1. Verify the file is in the correct directory (`ls custom_voices/`)
2. Check the file extension is `.pt` or `.wav`
3. Restart the PersonaPlex server
4. Test the API endpoint: `curl http://localhost:8998/api/voices`
5. Check server logs for errors

For more help, see `TROUBLESHOOTING.md` in the repository root.
