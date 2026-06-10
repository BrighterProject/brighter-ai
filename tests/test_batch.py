import io

from fastapi import status


class TestBatchPredict:
    """Tests for POST /predict/batch — batch image classification."""

    def test_happy_path(self, client, sample_image_bytes):
        """Successfully classify multiple images in one request."""
        files = [
            ("images", ("room1.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")),
            ("images", ("room2.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")),
        ]
        response = client.post("/predict/batch", files=files)
        assert response.status_code == status.HTTP_200_OK
        body = response.json()
        assert "predictions" in body
        assert len(body["predictions"]) == 2
        for pred in body["predictions"]:
            assert "room_type" in pred
            assert "confidence" in pred

    def test_mixed_file_types(self, client, sample_image_bytes, sample_png_bytes):
        """Accept a mix of JPEG and PNG in one batch."""
        files = [
            ("images", ("room1.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")),
            ("images", ("room2.png", io.BytesIO(sample_png_bytes), "image/png")),
        ]
        response = client.post("/predict/batch", files=files)
        assert response.status_code == status.HTTP_200_OK
        body = response.json()
        assert len(body["predictions"]) == 2

    def test_batch_over_20_images(self, client, sample_image_bytes, monkeypatch):
        """Reject batches larger than the maximum allowed size with 422."""
        monkeypatch.setattr("app.settings.settings.max_batch_size", 2)

        files = [
            ("images", (f"room{i}.jpg", io.BytesIO(sample_image_bytes), "image/jpeg"))
            for i in range(3)
        ]
        response = client.post("/predict/batch", files=files)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "exceeds" in response.json()["detail"]

    def test_batch_with_invalid_file_type(self, client, sample_image_bytes):
        """Reject batch containing a non-image file."""
        files = [
            ("images", ("room1.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")),
            ("images", ("readme.txt", io.BytesIO(b"not an image"), "text/plain")),
        ]
        response = client.post("/predict/batch", files=files)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_with_corrupt_image(self, client, sample_image_bytes):
        """Reject batch containing a corrupt (unreadable) image with 422."""
        files = [
            ("images", ("room1.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")),
            ("images", ("corrupt.jpg", io.BytesIO(b"not valid jpeg data"), "image/jpeg")),
        ]
        response = client.post("/predict/batch", files=files)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "detail" in response.json()

    def test_empty_batch(self, client):
        """Batch endpoint with no images returns 422 (missing required field)."""
        response = client.post("/predict/batch", files=[])
        # FastAPI rejects empty file list because `images` is a required File field
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "detail" in response.json()
