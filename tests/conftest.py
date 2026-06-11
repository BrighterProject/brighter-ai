"""Shared test fixtures — mocks the model layer so no real .keras file is needed."""

import io
from unittest.mock import patch

import pytest
from PIL import Image


@pytest.fixture
def mock_model():
    """Return the mock prediction result tuple."""
    return ("bedroom", 0.95)


@pytest.fixture
def client(mock_model):
    """Provide a TestClient with model.init_model, predict, and predict_batch patched out.

    No real .keras file is required — we mock the model layer entirely by
    patching init_model() (no-op), predict(), and predict_batch().
    """
    with (
        patch("app.model.init_model"),
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
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes():
    """Return a minimal valid PNG image as bytes (100x100 colored)."""
    img = Image.new("RGB", (100, 100), color=(64, 128, 192))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
