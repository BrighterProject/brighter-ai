import io

import torch
from PIL import Image
from torchvision import transforms

from app.settings import settings

# ImageNet normalization constants — kept server-side so clients don't need to know
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocessing pipeline: decode → resize → tensor → normalize
_transform = transforms.Compose(
    [
        transforms.Resize((settings.input_size, settings.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Decode raw image bytes into a normalized [3, 224, 224] tensor.

    Args:
        image_bytes: Raw JPEG or PNG file contents.

    Returns:
        A [3, 224, 224] FloatTensor ready for model inference.

    Raises:
        ValueError: If the bytes cannot be decoded as a valid image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    return _transform(image)


def preprocess_batch(image_bytes_list: list[bytes]) -> torch.Tensor:
    """Decode multiple raw images and stack into a single [N, 3, 224, 224] tensor.

    Args:
        image_bytes_list: List of raw JPEG or PNG file contents.

    Returns:
        An [N, 3, 224, 224] FloatTensor where N = len(image_bytes_list).

    Raises:
        ValueError: If any image cannot be decoded.
    """
    tensors = [preprocess_image(b) for b in image_bytes_list]
    return torch.stack(tensors, dim=0)
