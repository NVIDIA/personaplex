# Troubleshooting Guide

## Development Issues

### Code Changes Not Reflected When Running Server

**Symptom:** You made changes to backend code (e.g., added new endpoints, modified routes), but when you restart the server with `python -m moshi.server`, the changes don't appear. New API endpoints return 404, and debug logging doesn't show up.

**Root Cause:** The moshi-personaplex package was installed in regular mode (`pip install .`) instead of editable mode (`pip install -e .`). When installed normally, pip copies the code to site-packages (e.g., `/path/to/envs/personaplex/lib/python3.10/site-packages/moshi`), and Python loads from there instead of your source directory.

**Solution:**

1. Check if the package is installed:
   ```bash
   pip list | grep moshi
   ```

2. Check installation location:
   ```bash
   pip show moshi-personaplex
   ```

   If `Location` shows `site-packages`, it's not in editable mode.

3. Uninstall and reinstall in editable mode:
   ```bash
   pip uninstall -y moshi-personaplex
   cd moshi
   pip install -e .
   ```

4. Verify editable install:
   ```bash
   pip show moshi-personaplex
   ```

   The `Location` should show something like `/path/to/repo/moshi` instead of `site-packages`.

5. Restart the server:
   ```bash
   python -m moshi.server
   ```

**Prevention:** Always use `pip install -e .` (with the `-e` flag) when installing packages for development.

## Server Issues

### Server Returns 404 for API Endpoints

If specific API endpoints return 404:

1. Check if routes are registered correctly by looking for debug logging at server startup
2. Verify the package is installed in editable mode (see above)
3. Check that static routes are registered AFTER API routes in server.py
4. Clear Python cache and restart:
   ```bash
   find moshi -type f -name '*.pyc' -delete
   find moshi -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null
   python -m moshi.server
   ```

### Build Directory Conflicts

If you suspect the `moshi/build/` directory contains old code:

1. Move it out of the way:
   ```bash
   mv moshi/build moshi/build.bak
   ```

2. Clear Python cache:
   ```bash
   find moshi -type f -name '*.pyc' -delete
   find moshi -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null
   ```

3. Restart the server

## Frontend Issues

### Frontend Not Showing New Features

If you modified React components but don't see changes:

1. Rebuild the frontend:
   ```bash
   cd client
   npm run build
   ```

2. Restart the server (it serves the static files from client/dist)

3. Hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)

### Voice Dropdown Shows "Error loading voices"

1. Check if the server is running:
   ```bash
   ps aux | grep moshi.server
   ```

2. Test the API endpoint directly:
   ```bash
   curl http://localhost:8998/api/voices
   ```

3. Check server logs for errors

4. Verify VoiceDiscovery can find voice files:
   ```bash
   python -c "from moshi.voice_discovery import VoiceDiscovery; print(len(VoiceDiscovery.list_voices()))"
   ```

## Environment Issues

### Missing HuggingFace Token

If models fail to download:

1. Create a `.env` file in the repository root
2. Add your HuggingFace token:
   ```
   HF_TOKEN=your_token_here
   ```
3. See `.env.example` for more details

### ffmpeg Not Found

If voice generation fails with "Command 'ffmpeg' not found":

```bash
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

## Getting Help

If you encounter issues not covered here:

1. Check the README.md for setup instructions
2. Review recent commits for breaking changes
3. Open an issue at https://github.com/nvidia/personaplex-7b-v1/issues with:
   - Your environment (OS, Python version, conda/venv)
   - Steps to reproduce the issue
   - Complete error messages and logs
