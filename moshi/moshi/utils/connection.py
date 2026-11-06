# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import socket
import subprocess
import platform
from shutil import which
import urllib.request
from pathlib import Path
from .logging import setup_logger

logger = setup_logger(__name__)


def query_seed(request):
    """Return the optional integer seed from the request query string.

    Query parameters live on ``request.query``; ``request[key]`` only looks up
    path (match_info) parameters, so reading ``request["seed"]`` raises
    ``KeyError`` whenever a client passes ``?seed=`` in the query string.
    Returns None when no seed is provided.
    """
    if "seed" not in request.query:
        return None
    return int(request.query["seed"])


def get_lan_ip():
    """
    get_lan_ip()
    
    Get the LAN IP address of the current machine.

    Returns
    -------
    str
        LAN IP address as a string.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def ensure_mkcert():
    """
    Ensure mkcert exists; if not, download the correct binary.
    
    Returns
    -------
    str | None
        Path to mkcert binary, or None if installation failed.
    """

    # mkcert already installed?
    mkcert_bin_path = which("mkcert")
    if mkcert_bin_path:
        return mkcert_bin_path

    logger.info("[auto-cert] mkcert not found, attempting to download...")

    # Determine OS + arch
    system = platform.system().lower()   # "linux", "darwin", "windows"
    machine = platform.machine().lower() # "x86_64", "amd64", "arm64", etc.

    # Normalize architecture
    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    elif machine.startswith("arm"):
        arch = "arm"
    else:
        logger.error(f"[auto-cert] Unsupported architecture: {machine}")
        return None

    # Determine filename and URL
    if system == "linux":
        filename = f"mkcert-v1.4.4-linux-{arch}"
    elif system == "darwin":
        filename = f"mkcert-v1.4.4-darwin-{arch}"
    elif system == "windows":
        filename = f"mkcert-v1.4.4-windows-{arch}.exe"
    else:
        logger.error(f"[auto-cert] Unsupported OS: {system}")
        return None

    url = f"https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/{filename}"
    dest_dir = Path.home() / ".local" / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ("mkcert.exe" if system == "windows" else "mkcert")

    logger.info(f"[auto-cert] Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        logger.error(f"[auto-cert] Failed to download mkcert: {e}")
        return None

    # Make executable if Unix
    if system != "windows":
        dest.chmod(0o755)

    logger.info(f"[auto-cert] mkcert installed at {dest}")

    # Verify it works
    try:
        subprocess.check_call([str(dest), "-help"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        logger.error("[auto-cert] mkcert failed to run after install.")
        return None

    return str(dest)


def _run_command(cmd):
    """Run command, return True on success."""
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_mkcert_ca(mkcert_bin: str):
    """
    Install mkcert CA if not installed yet.

    Parameters
    ----------
    mkcert_bin : str
        Path to mkcert binary.
    """
    _run_command([mkcert_bin, "-install"])


def create_cert_if_needed(cert_dir: str):
    """
    Create cert.pem and key.pem using mkcert if they don't already exist.
    Returns (cert_file, key_file) or (None, None) if mkcert unavailable.

    Parameters
    ----------
    cert_dir : str
        Directory to store or find cert.pem and key.pem.
    
    Returns
    -------
    pathlib.Path | None
        Path to cert.pem, or None if not created.
    pathlib.Path | None
        Path to key.pem, or None if not created.
    """
    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)

    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    # Already exists → nothing to do
    if cert_file.exists() and key_file.exists():
        return cert_file, key_file

    mkcert_bin = ensure_mkcert()

    if not mkcert_bin:
        logger.warning("[auto-cert] mkcert not installed; falling back to HTTP.")
        return None, None

    logger.info("[auto-cert] mkcert detected. Ensuring local CA installed...")
    ensure_mkcert_ca(mkcert_bin)

    # Create cert for localhost, loopback, and LAN IP
    lan_ip = get_lan_ip()
    logger.info(f"[auto-cert] Generating certificate for localhost and {lan_ip}...")

    success = _run_command([
        mkcert_bin,
        "-cert-file", str(cert_file),
        "-key-file", str(key_file),
        "localhost",
        "127.0.0.1",
        "::1",
        lan_ip
    ])

    if not success:
        logger.warning("[auto-cert] mkcert failed. Using HTTP.")
        return None, None

    logger.info("[auto-cert] Certificate generated.")
    return cert_file, key_file


def create_ssl_context(cert_dir: str):
    """
    Main entry: create SSL context if certificates can be created.
    Returns (ssl_context, protocol_str) where protocol_str is "http" or "https".

    Parameters
    ----------
    cert_dir : str
        Directory to store or find cert.pem and key.pem.

    Returns
    -------
    ssl.SSLContext | None
        SSL context if HTTPS is available, else None.
    str
        "https" if SSL context created, else "http".
    """
    cert_file, key_file = create_cert_if_needed(cert_dir)

    if cert_file is None:
        # mkcert missing → use HTTP
        return None, "http"

    import ssl
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    return ctx, "https"
