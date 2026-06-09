import io

import pytest
from fastapi import status


class TestSinglePredict:
    """Tests for POST /predict — single image classification."""

    def test_happy_path_jpeg(self, client, sample_image_bytes):
        """Successfully classify a valid JPEG image."""
        response = client.post(
            "/predict",
            files={"image": ("room.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
        )
        assert response.status_code == status.HTTP_200_OK
        body = response.json()
        assert body["room_type"] == "bedroom"
        assert body["confidence"] == pytest.approx(0.95)

    def test_happy_path_png(self, client, sample_png_bytes):
        """Successfully classify a valid PNG image."""
        response = client.post(
            "/predict",
            files={"image": ("room.png", io.BytesIO(sample_png_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK
        body = response.json()
        assert "room_type" in body
        assert "confidence" in body

    def test_wrong_file_type(self, client):
        """Reject non-image file types with 422."""
        response = client.post(
            "/predict",
            files={"image": ("readme.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        body = response.json()
        assert "detail" in body
        assert "JPEG" in body["detail"] or "PNG" in body["detail"]

    def test_oversized_image(self, client, monkeypatch):
        """Reject images exceeding the max size limit with 422."""
        # Lower the limit to 1 byte so any payload triggers the error
        monkeypatch.setattr("app.settings.settings.max_image_size_mb", 0)

        big_image = b"\xff\xd8" + b"\x00" * 1024  # Fake JPEG header + padding
        response = client.post(
            "/predict",
            files={"image": ("huge.jpg", io.BytesIO(big_image), "image/jpeg")},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "exceeds" in response.json()["detail"]

    def test_corrupt_image(self, client):
        """Reject unreadable / corrupt image data with 422."""
        response = client.post(
            "/predict",
            files={"image": ("corrupt.jpg", io.BytesIO(b"not valid jpeg data"), "image/jpeg")},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "detail" in response.json()
