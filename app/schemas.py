from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """Single image classification result."""

    room_type: str = Field(..., description="Predicted room category")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax confidence score for the prediction",
    )


class SinglePredictResponse(BaseModel):
    """Response from POST /predict."""

    room_type: str
    confidence: float


class BatchPredictResponse(BaseModel):
    """Response from POST /predict/batch."""

    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str
    model_loaded: bool
