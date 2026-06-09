from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_path: str = "models/room_classifier.pt"
    max_image_size_mb: int = 10
    max_batch_size: int = 20
    input_size: int = 224
    num_classes: int = 6

    # OTEL / Telemetry
    otel_sdk_disabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    service_name: str = "brighter-ai-blehhing"
    service_version: str = "1.0.0"

    # Logging
    log_level: str = "INFO"

    model_config = ConfigDict(env_prefix="", case_sensitive=False)


settings = Settings()


ROOM_CLASSES = [
    "bedroom",
    "living_room",
    "kitchen",
    "bathroom",
    "balcony",
    "hallway",
]
