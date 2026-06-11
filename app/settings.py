from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_path: str = "model/room_classifier.keras"
    max_image_size_mb: int = 10
    max_batch_size: int = 20
    input_size: int = 224
    num_classes: int = 5

    # OTEL / Telemetry
    otel_sdk_disabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    service_name: str = "brighter-ai-blehhing"
    service_version: str = "1.0.0"

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)


settings = Settings()


ROOM_CLASSES = [
    "bathroom",
    "bedroom",
    "dinning",
    "kitchen",
    "livingroom",
]
