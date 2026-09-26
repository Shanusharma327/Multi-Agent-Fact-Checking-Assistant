"""One-cell Google Colab launcher for the fact-checking dashboard.

Paste into a Colab cell and run:

    !pip install -q -r requirements.txt pyngrok
    import colab_setup; colab_setup.launch("sk-ant-...", ngrok_token="...")
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Optional

PORT = 8501


def launch(api_key: str, ngrok_token: Optional[str] = None, port: int = PORT) -> str:
    """Start Streamlit in the background and return a public URL.

    Args:
        api_key: Anthropic API key, exported for the child process.
        ngrok_token: ngrok auth token. Required for a public tunnel.
        port: Local port to serve on.

    Returns:
        The public tunnel URL, or the local URL if no tunnel was created.
    """
    os.environ["ANTHROPIC_API_KEY"] = api_key

    subprocess.Popen(
        [
            "streamlit",
            "run",
            "app.py",
            f"--server.port={port}",
            "--server.address=0.0.0.0",
            "--server.headless=true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(6)  # Give the server a moment to bind before tunnelling.

    if not ngrok_token:
        return f"http://localhost:{port}"

    from pyngrok import ngrok

    ngrok.set_auth_token(ngrok_token)
    return str(ngrok.connect(port).public_url)
