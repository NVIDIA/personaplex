# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Voice discovery service for listing available voices."""
from pathlib import Path
from typing import List, Dict
import os


class VoiceDiscovery:
    """Discovers and lists available voice files."""

    @staticmethod
    def get_voice_directories() -> List[Path]:
        """Get all directories where voices can be stored.

        Returns:
            List of Path objects for directories containing voice files
        """
        dirs = []

        # 1. HuggingFace cache voices directory
        hf_cache = os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))
        hf_voices = Path(hf_cache) / 'hub'

        # Find personaplex model snapshot
        for model_dir in hf_voices.glob('models--nvidia--personaplex-7b-v1/snapshots/*'):
            voices_dir = model_dir / 'voices'
            if voices_dir.exists():
                dirs.append(voices_dir)

        # 2. Custom voices directory (from .env or default)
        custom_dir = os.environ.get('CUSTOM_VOICE_DIR', './custom_voices')
        custom_path = Path(custom_dir)
        if custom_path.exists():
            dirs.append(custom_path)

        return dirs

    @staticmethod
    def list_voices() -> List[Dict[str, str]]:
        """List all available voices.

        Only returns .pt embedding files, not .wav source audio files.
        .wav files are used to generate embeddings and should not be listed as voices.

        Returns:
            List of voice info dicts with keys: name, type, category, path
            Sorted with custom voices first, then by category, then alphabetically
        """
        voices = []
        seen_names = set()

        for voice_dir in VoiceDiscovery.get_voice_directories():
            # Find .pt files (voice embeddings only)
            for pt_file in voice_dir.glob('*.pt'):
                name = pt_file.name
                if name not in seen_names:
                    category = VoiceDiscovery._categorize_voice(name)
                    voices.append({
                        'name': name,
                        'type': 'embeddings',
                        'category': category,
                        'path': str(pt_file)
                    })
                    seen_names.add(name)

        # Sort: custom first, then by category, then by name
        def sort_key(v):
            cat_order = {
                'custom': 0,
                'natural-female': 1,
                'natural-male': 2,
                'variety-female': 3,
                'variety-male': 4,
                'other': 5
            }
            return (cat_order.get(v['category'], 99), v['name'])

        return sorted(voices, key=sort_key)

    @staticmethod
    def _categorize_voice(filename: str) -> str:
        """Categorize voice by filename pattern.

        Args:
            filename: Voice filename (.pt extension)

        Returns:
            Category string: custom, natural-female, natural-male,
            variety-female, variety-male, or other
        """
        name = filename.replace('.pt', '')

        if name.startswith('NATF'):
            return 'natural-female'
        elif name.startswith('NATM'):
            return 'natural-male'
        elif name.startswith('VARF'):
            return 'variety-female'
        elif name.startswith('VARM'):
            return 'variety-male'
        else:
            return 'custom'
