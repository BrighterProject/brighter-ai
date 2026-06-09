from fastapi import status


class TestHealthAndMetrics:
    """Tests for GET /health and GET /metrics endpoints."""

    def test_health_ok(self, client):
        """Health endpoint reports service is up and model is loaded."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_metrics_responds(self, client):
        """Prometheus metrics endpoint returns text format."""
        response = client.get("/metrics")
        assert response.status_code == status.HTTP_200_OK
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        # Should contain at least the default Python metrics + our custom ones
        assert "room_classifier" in body or "python_" in body

    def test_root_redirects_to_docs_message(self, client):
        """Root endpoint returns a helpful message pointing to docs."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        body = response.json()
        assert "/docs" in body["message"]
