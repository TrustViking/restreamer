from __future__ import annotations

import io
import logging
from typing import Tuple

from app.core.models import NormalizedImage

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def guess_image_extension_and_mime(image_bytes: bytes) -> Tuple[str, str]:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return ".gif", "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp", "image/webp"
    return ".jpg", "image/jpeg"


def normalize_thumbnail(image_bytes: bytes, *, logger: logging.Logger) -> NormalizedImage:
    if Image is None:
        ext, mime = guess_image_extension_and_mime(image_bytes)
        logger.warning(
            "Pillow is not installed; keeping original thumbnail format (%s).",
            mime,
        )
        return NormalizedImage(
            bytes_data=image_bytes,
            extension=ext,
            mime_type=mime,
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
            output = io.BytesIO()
            rgb_image.save(output, format="JPEG", quality=90)
            return NormalizedImage(
                bytes_data=output.getvalue(),
                extension=".jpg",
                mime_type="image/jpeg",
            )
    except Exception as conversion_error:
        ext, mime = guess_image_extension_and_mime(image_bytes)
        logger.warning(
            "Thumbnail conversion to JPEG failed; keeping original format (%s). Error: %s",
            mime,
            conversion_error,
        )
        return NormalizedImage(
            bytes_data=image_bytes,
            extension=ext,
            mime_type=mime,
        )
