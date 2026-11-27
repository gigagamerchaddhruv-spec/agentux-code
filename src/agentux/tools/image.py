"""Terminal image graphics rendering tool powered by Chafa.

Allows the AI model and user to view images directly inside the terminal.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

from ..errors import ToolError
from .base import BaseTool

CHAFA_CANDIDATES = [
    "chafa",
    "/opt/local/bin/chafa",
    "/usr/local/bin/chafa",
    "/usr/bin/chafa",
    "/home/linuxbrew/.linuxbrew/bin/chafa",
]


def find_chafa() -> Optional[str]:
    """Locate the chafa executable on the host system."""
    for candidate in CHAFA_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def get_image_info(path: str) -> Optional[Tuple[str, int, int]]:
    """Extract format, width, and height for common image types without external libraries."""
    try:
        with open(path, "rb") as f:
            data = f.read(32)
            if len(data) < 16:
                return None

            # PNG
            if data.startswith(b"\x89PNG\r\n\x1a\n"):
                w, h = struct.unpack(">II", data[16:24])
                return ("PNG", w, h)

            # GIF
            if data.startswith((b"GIF87a", b"GIF89a")):
                w, h = struct.unpack("<HH", data[6:10])
                return ("GIF", w, h)

            # JPEG
            if data.startswith(b"\xff\xd8"):
                f.seek(0)
                b = f.read()
                i = 2
                while i < len(b) - 8:
                    marker, length = struct.unpack(">HH", b[i:i+4])
                    if marker in (0xFFC0, 0xFFC2):
                        h, w = struct.unpack(">HH", b[i+5:i+9])
                        return ("JPEG", w, h)
                    i += length + 2
                return ("JPEG", 0, 0)
    except Exception:
        pass
    return None


class ViewImageTool(BaseTool):
    """Render and inspect local or remote images in terminal graphics format."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = os.path.realpath(root or os.getcwd())

    @property
    def name(self) -> str:
        return "view_image"

    @property
    def description(self) -> str:
        return (
            "View and render images (PNG, JPG, SVG, WebP, GIF) directly in the "
            "terminal using Chafa terminal graphics. Accepts local file paths or URLs."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to local image file (relative to workspace) or remote HTTP/HTTPS URL.",
                },
                "max_width": {
                    "type": "integer",
                    "description": "Maximum width in terminal character columns (default: 60).",
                },
                "max_height": {
                    "type": "integer",
                    "description": "Maximum height in terminal character rows (default: 25).",
                },
            },
            "required": ["path"],
        }

    async def execute(self, path: str, max_width: int = 60, max_height: int = 25) -> str:
        path = path.strip()
        if not path:
            raise ToolError("Image path cannot be empty.")

        local_path = path
        is_temp = False

        if path.startswith(("http://", "https://")):
            # Download remote image to temporary file
            try:
                import tempfile
                parsed = urllib.parse.urlparse(path)
                suffix = os.path.splitext(parsed.path)[1] or ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    local_path = tmp.name
                    is_temp = True
                urllib.request.urlretrieve(path, local_path)
            except Exception as exc:
                return f"Failed to download remote image from {path}: {exc}"
        else:
            # Resolve and verify local path
            if not os.path.isabs(local_path):
                local_path = os.path.join(self.root, local_path)
            local_path = os.path.realpath(local_path)
            if not local_path.startswith(self.root):
                raise ToolError(f"Path escapes workspace directory: {path}")
            if not os.path.isfile(local_path):
                raise ToolError(f"Image file not found: {path}")

        try:
            chafa_bin = find_chafa()
            if chafa_bin:
                size_arg = f"--size={max_width}x{max_height}"
                proc = subprocess.run(
                    [chafa_bin, size_arg, "--format=symbols", local_path],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return f"[Terminal Graphics: {os.path.basename(path)}]\n\n" + proc.stdout

            # Fallback metadata representation if chafa is not installed
            info = get_image_info(local_path)
            if info:
                fmt, w, h = info
                dim_str = f"{w}x{h} px" if w and h else "unknown dimensions"
                size_kb = os.path.getsize(local_path) / 1024
                return f"[📷 Image: {os.path.basename(path)} · {fmt} · {dim_str} · {size_kb:.1f} KB (chafa not installed for ASCII render)]"

            return f"[📷 Image: {os.path.basename(path)} (chafa not installed for inline rendering)]"
        finally:
            if is_temp and os.path.exists(local_path):
                try:
                    os.unlink(local_path)
                except Exception:
                    pass


__all__ = ["ViewImageTool", "find_chafa"]
