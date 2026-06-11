from pathlib import Path

import numpy as np
import tensorflow as tf
from loguru import logger

from app.settings import ROOM_CLASSES, settings

# Module-level state — populated at startup
_model: tf.keras.Model | None = None


def load_model(model_path: str | None = None) -> tf.keras.Model:
    model_path_str = model_path or settings.model_path
    model_path_obj = Path(model_path_str)

    if not model_path_obj.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        model_path_obj = project_root / model_path_obj

    if not model_path_obj.is_file():
        logger.error(f"Model file not found: {model_path_obj}")
        raise FileNotFoundError(f"Model file not found: {model_path_obj}")

    logger.info(f"Loading model from {model_path_obj} ...")
    model = tf.keras.models.load_model(str(model_path_obj))
    logger.info("Model loaded successfully")
    return model


def init_model() -> None:
    global _model
    _model = load_model()


def predict(input_array: np.ndarray) -> tuple[str, float]:
    if _model is None:
        raise RuntimeError("Model has not been loaded. Call init_model() at startup.")

    if input_array.ndim == 3:
        input_array = np.expand_dims(input_array, axis=0)

    logits = _model(input_array, training=False)
    probs = tf.nn.softmax(logits).numpy()

    top_idx = int(probs.argmax(axis=1)[0])
    confidence = float(probs.max(axis=1)[0])

    return ROOM_CLASSES[top_idx], confidence


def predict_batch(input_array: np.ndarray) -> list[tuple[str, float]]:
    if _model is None:
        raise RuntimeError("Model has not been loaded. Call init_model() at startup.")

    logits = _model(input_array, training=False)
    probs = tf.nn.softmax(logits).numpy()

    top_indices = probs.argmax(axis=1)
    top_probs = probs.max(axis=1)

    return [
        (ROOM_CLASSES[int(idx)], float(conf))
        for idx, conf in zip(top_indices, top_probs)
    ]


def is_model_loaded() -> bool:
    """Return whether the model has been successfully loaded."""
    return _model is not None
