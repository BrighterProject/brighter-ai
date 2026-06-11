import io

import numpy as np
from PIL import Image

from app.settings import settings


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a [224, 224, 3] uint8 array.

    The returned array contains raw pixel values in [0, 255].
    The model includes an internal rescaling layer, so no further
    normalization is required here.

    Args:
        image_bytes: Raw JPEG or PNG file contents.

    Returns:
        A [224, 224, 3] uint8 ndarray ready for model inference.

    Raises:
        ValueError: If the bytes cannot be decoded as a valid image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    image = image.resize((settings.input_size, settings.input_size))
    return np.array(image, dtype=np.uint8)


def preprocess_batch(image_bytes_list: list[bytes]) -> np.ndarray:
    """Decode multiple raw images and stack into a single [N, 224, 224, 3] array.

    Args:
        image_bytes_list: List of raw JPEG or PNG file contents.

    Returns:
        An [N, 224, 224, 3] uint8 ndarray where N = len(image_bytes_list).

    Raises:
        ValueError: If any image cannot be decoded.
    """
    arrays = [preprocess_image(b) for b in image_bytes_list]
    return np.stack(arrays, axis=0)
