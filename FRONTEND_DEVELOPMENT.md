# Frontend Development Guide

This guide explains how to develop and test custom UI changes for PersonaPlex.

## Understanding Smart Auto-Detection

**PersonaPlex now automatically detects and serves your custom UI!** You no longer need to use the `--static` flag for development.

### How Auto-Detection Works

When you start the server, it checks:
1. Does `client/dist` exist in the project directory?
2. **YES** → Automatically serves your custom UI
3. **NO** → Downloads and serves the default UI from HuggingFace

### Starting the Server (Auto-Detection)

```bash
# Just start the server normally - no flags needed!
conda activate personaplex
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

**Log output with custom UI detected:**
```
Found custom UI at /home/.../personaplex-blackwell/client/dist, using it instead of default
static_path = /home/.../personaplex-blackwell/client/dist
serving static content from /home/.../personaplex-blackwell/client/dist
```

**Log output without custom UI:**
```
retrieving the static content
static_path = /home/.../.cache/huggingface/.../dist
serving static content from /home/.../.cache/huggingface/.../dist
```

### Manual Override (Optional)

You can still manually specify the UI source if needed:

```bash
# Force specific directory
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --static /path/to/custom/dist

# Disable static serving
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --static none
```

## Frontend Development Workflow

### Prerequisites

1. Install Node.js and npm (if not already installed):
   ```bash
   # Check if already installed
   node --version
   npm --version
   ```

2. Install frontend dependencies:
   ```bash
   cd client
   npm install
   ```

### Development Steps (Simplified!)

#### 1. Make Your Changes
Edit files in the `client/src/` directory:
- `client/src/components/` - React components
- `client/src/styles/` - CSS and styling
- `client/src/App.tsx` - Main application component

#### 2. Build the Frontend
```bash
cd client
npm run build
cd ..
```

This creates/updates the `client/dist` directory with your compiled code.

#### 3. Start Server (Auto-Detection!)
```bash
# From project root - server auto-detects custom UI!
conda activate personaplex
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

#### 4. Verify Custom UI is Loaded
Check the server logs for:
```
Found custom UI at .../client/dist, using it instead of default
static_path = /home/.../personaplex-blackwell/client/dist
```

If you see `retrieving the static content`, the build might not exist. Go back to step 2.

#### 5. Test Your Changes
1. Open the Web UI: https://localhost:8998
2. Hard refresh (Ctrl+Shift+R or Cmd+Shift+R) to clear browser cache
3. Test your modifications

#### 6. Iterate
Repeat steps 1-5 for each change:
```bash
# Make changes to client/src/...
cd client && npm run build && cd ..

# Restart server (Ctrl+C to stop first) - auto-detects custom UI!
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

**That's it! No `--static` flag needed anymore.**

## Troubleshooting

### Changes Not Appearing

**Problem:** You rebuilt the frontend but don't see changes in the browser.

**Solutions:**
1. **Verify server is using custom UI:**
   - Check logs for `static_path = client/dist`
   - If not, restart with `--static client/dist`

2. **Clear browser cache:**
   - Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
   - Or open DevTools (F12) → Network tab → Check "Disable cache"

3. **Verify build completed successfully:**
   ```bash
   cd client
   npm run build
   ls -la dist/  # Should show recent timestamps
   ```

4. **Check for build errors:**
   ```bash
   cd client
   npm run build 2>&1 | grep -i error
   ```

### Server Won't Start with --static Flag

**Problem:** Error when starting server with `--static client/dist`

**Solutions:**
1. **Verify dist directory exists:**
   ```bash
   ls -la client/dist/
   ```
   If missing, build the frontend first: `cd client && npm run build`

2. **Check path is correct:**
   - Use relative path: `--static client/dist`
   - From project root, not from client/ directory

### Frontend Build Fails

**Problem:** `npm run build` fails with errors

**Solutions:**
1. **Check Node.js version:**
   ```bash
   node --version
   # Should be 16.x or higher
   ```

2. **Reinstall dependencies:**
   ```bash
   cd client
   rm -rf node_modules package-lock.json
   npm install
   npm run build
   ```

3. **Check for TypeScript errors:**
   ```bash
   cd client
   npm run type-check
   ```

## Development Tips

### Shell Alias for Quick Development
Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Quick start with custom UI
alias moshi-dev='conda activate personaplex && SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --static client/dist'

# Quick frontend rebuild
alias moshi-build='cd client && npm run build && cd ..'
```

Usage:
```bash
# Make changes to client/src/...
moshi-build     # Rebuild frontend
moshi-dev       # Start server with custom UI
```

### Watch Mode for Live Development

For faster iteration, use the frontend in development mode:

```bash
# Terminal 1: Start backend server (without static flag)
conda activate personaplex
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"

# Terminal 2: Start frontend dev server with hot reload
cd client
npm run dev
```

Then access the UI at the Vite dev server URL (usually http://localhost:5173).

**Note:** This requires configuring CORS in the backend. Check `client/vite.config.ts` for proxy settings.

## Production Deployment

When ready to deploy your custom UI:

1. Build the production bundle:
   ```bash
   cd client
   npm run build
   ```

2. Test the production build:
   ```bash
   SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --static client/dist
   ```

3. Verify everything works correctly

4. Commit your changes:
   ```bash
   git add client/src/ client/dist/
   git commit -m "Add custom UI feature: [description]"
   ```

## File Structure

```
personaplex-blackwell/
├── client/                    # Frontend source code
│   ├── src/                  # Source files (edit these)
│   │   ├── components/       # React components
│   │   ├── styles/          # CSS files
│   │   ├── App.tsx          # Main app
│   │   └── main.tsx         # Entry point
│   ├── dist/                # Built files (generated)
│   │   ├── index.html       # HTML entry
│   │   ├── assets/          # JS/CSS bundles
│   │   └── ...
│   ├── package.json         # Dependencies
│   ├── vite.config.ts       # Build config
│   └── tsconfig.json        # TypeScript config
└── moshi/                   # Backend Python code
```

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `cd client && npm install` |
| Build frontend | `cd client && npm run build` |
| Start with custom UI | `SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --static client/dist` |
| Start with default UI | `SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"` |
| Dev server (hot reload) | `cd client && npm run dev` |
| Type check | `cd client && npm run type-check` |
| Lint code | `cd client && npm run lint` |

## Getting Help

If you encounter issues not covered here:
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common problems
2. Verify your Node.js and npm versions
3. Check the browser console (F12) for JavaScript errors
4. Review server logs for static file serving errors
