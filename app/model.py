import os

import torch
import torch.nn.functional as F
from loguru import logger
from torchvision.models import efficientnet_b0

from app.settings import ROOM_CLASSES, settings

# Module-level state — populated at startup
_model: torch.nn.Module | None = None
_device: torch.device = torch.device("cpu")


def load_model(model_path: str | None = None) -> torch.nn.Module:
    """Load the EfficientNet-B0 checkpoint with a custom classification head.

    The model file is expected to contain state_dict keys for a 6-class
    classifier built on top of EfficientNet-B0 features.

    Args:
        model_path: Path to the .pt checkpoint. Falls back to ``settings.model_path``.

    Returns:
        The loaded model in eval mode on CPU.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: If the state_dict cannot be loaded.
    """
    path = model_path or settings.model_path

    if not os.path.isfile(path):
        logger.error(f"Model file not found: {path}")
        raise FileNotFoundError(f"Model file not found: {path}")

    logger.info(f"Loading model from {path} ...")

    model = efficientnet_b0(weights=None)
    # Replace classifier head for 6 room classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, settings.num_classes)

    state_dict = torch.load(path, map_location=_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def init_model() -> None:
    """Initialize the global model instance at application startup.

    Follows a fail-fast strategy: if the model cannot be loaded the service
    crashes immediately and lets the container orchestrator restart it.
    """
    global _model
    _model = load_model()


def predict(input_tensor: torch.Tensor) -> tuple[str, float]:
    """Run inference on a single preprocessed image or a batch.

    For a batch tensor of shape [N, 3, 224, 224], returns the *highest-confidence*
    prediction across the batch (used by the single-predict endpoint).
    For a single image tensor of shape [3, 224, 224], returns that image's prediction.

    Args:
        input_tensor: Either [3, 224, 224] or [N, 3, 224, 224] FloatTensor.

    Returns:
        Tuple of (predicted_room_class, confidence_score).

    Raises:
        RuntimeError: If the model has not been loaded.
    """
    if _model is None:
        raise RuntimeError("Model has not been loaded. Call init_model() at startup.")

    with torch.no_grad():
        # Ensure batched input: [1, 3, 224, 224] for single image
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(_device)
        logits = _model(input_tensor)
        probs = F.softmax(logits, dim=1)

        # For single-image requests, return the top prediction
        top_prob, top_idx = probs.max(dim=1)
        idx = int(top_idx.item())
        confidence = float(top_prob.item())

    return ROOM_CLASSES[idx], confidence


def predict_batch(input_tensor: torch.Tensor) -> list[tuple[str, float]]:
    """Run inference on a batch of preprocessed images.

    Args:
        input_tensor: [N, 3, 224, 224] FloatTensor.

    Returns:
        List of (predicted_room_class, confidence_score) for each image.

    Raises:
        RuntimeError: If the model has not been loaded.
    """
    if _model is None:
        raise RuntimeError("Model has not been loaded. Call init_model() at startup.")

    with torch.no_grad():
        input_tensor = input_tensor.to(_device)
        logits = _model(input_tensor)
        probs = F.softmax(logits, dim=1)

        top_probs, top_indices = probs.max(dim=1)

    return [
        (ROOM_CLASSES[int(idx.item())], float(conf.item()))
        for idx, conf in zip(top_indices, top_probs)
    ]


def is_model_loaded() -> bool:
    """Return whether the model has been successfully loaded."""
    return _model is not None
