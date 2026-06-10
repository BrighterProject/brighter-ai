"""Shared test fixtures — mocks the model layer so no real .pt file is needed."""

import os
from unittest.mock import patch

import pytest
import torch


# Create a minimal fake checkpoint so init_model() can load successfully.
# The fake model is an EfficientNet-B0 with 6 output classes whose weights
# are randomly initialized — we never run real inference on it because
# predict() and predict_batch() are also patched below.
def _create_fake_checkpoint(path: str) -> None:
    """Write a tiny valid .pt checkpoint to *path*."""
    from torchvision.models import efficientnet_b0

    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 6)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


@pytest.fixture
def mock_model():
    """Return the mock prediction result tuple."""
    return ("bedroom", 0.95)


@pytest.fixture
def client(mock_model):
    """Provide a TestClient with model.predict and model.predict_batch patched out.

    No real .pt file is required — we mock the model layer entirely by:
    1. Writing a fake (randomly-initialized) checkpoint that init_model() can load.
    2. Patching predict() and predict_batch() so no actual forward pass runs.
    """
    fake_path = "models/test_room_classifier.pt"
    _create_fake_checkpoint(fake_path)

    with (
        patch("app.settings.settings.model_path", fake_path),
        patch("app.model.predict") as mock_predict,
        patch("app.model.predict_batch") as mock_predict_batch,
    ):
        mock_predict.return_value = mock_model
        mock_predict_batch.return_value = [mock_model, ("kitchen", 0.91)]

        # Import main AFTER patches are in place so lifespan uses mocked config
        from fastapi.testclient import TestClient

        from main import app

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def sample_image_bytes():
    """Return a minimal valid JPEG image as bytes (100x100 gray)."""
    import io

    from PIL import Image

    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes():
    """Return a minimal valid PNG image as bytes (100x100 colored)."""
    import io

    from PIL import Image

    img = Image.new("RGB", (100, 100), color=(64, 128, 192))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
