import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from app.model import predict, predict_batch
from app.preprocessing import preprocess_batch, preprocess_image
from app.schemas import BatchPredictResponse, PredictionResult, SinglePredictResponse
from app.settings import settings
from app.telemetry import PREDICTION_COUNTER, get_tracer

router = APIRouter(tags=["predict"])

# Thread pool for offloading synchronous model inference from the async event loop
_executor = ThreadPoolExecutor(max_workers=4)

# Allowed MIME types — anything else gets rejected as non-image
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}


def _classify_single_sync(image_bytes: bytes) -> SinglePredictResponse:
    """Synchronous wrapper: preprocess + predict for one image."""
    tensor = preprocess_image(image_bytes)
    room_type, confidence = predict(tensor)
    PREDICTION_COUNTER.labels(room_type=room_type).inc()
    return SinglePredictResponse(room_type=room_type, confidence=confidence)


def _classify_batch_sync(image_bytes_list: list[bytes]) -> BatchPredictResponse:
    """Synchronous wrapper: preprocess all + single forward pass for a batch."""
    batch_tensor = preprocess_batch(image_bytes_list)
    results = predict_batch(batch_tensor)
    for room_type, _ in results:
        PREDICTION_COUNTER.labels(room_type=room_type).inc()
    predictions = [
        PredictionResult(room_type=rt, confidence=conf) for rt, conf in results
    ]
    return BatchPredictResponse(predictions=predictions)


def _read_file_chunks(file: UploadFile) -> bytes:
    """Read file in chunks, enforcing the max size limit."""
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    chunks = bytearray()
    while True:
        chunk = file.file.read(8192)
        if not chunk:
            break
        chunks.extend(chunk)
        if len(chunks) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Image exceeds maximum size of {settings.max_image_size_mb} MB",
            )
    return bytes(chunks)


def _validate_image_file(file: UploadFile) -> None:
    """Validate content type is an allowed image format."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid file type '{file.content_type}'. Only JPEG and PNG are supported.",
        )


@router.post("/predict", response_model=SinglePredictResponse)
async def predict_single(image: UploadFile = File(...)) -> SinglePredictResponse:
    """Classify a single real-estate photo into a room type.

    - **image**: JPEG or PNG file (max 10 MB)
    """
    with get_tracer().start_as_current_span("predict_single") as span:
        _validate_image_file(image)

        image_bytes = await asyncio.get_event_loop().run_in_executor(
            _executor, _read_file_chunks, image
        )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                _executor, _classify_single_sync, image_bytes
            )
        except ValueError as exc:
            logger.warning(f"Failed to process image: {exc}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Corrupt or unreadable image: {exc}",
            ) from exc

        span.set_attribute("room_type", result.room_type)
        span.set_attribute("confidence", result.confidence)
        logger.info(f"Single predict: {result.room_type} @ {result.confidence:.4f}")
        return result


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_endpoint(
    images: list[UploadFile] = File(...),
) -> BatchPredictResponse:
    """Classify up to 20 real-estate photos in one request.

    - **images**: List of JPEG or PNG files (each max 10 MB)
    """
    with get_tracer().start_as_current_span("predict_batch") as span:
        if len(images) > settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Batch size exceeds maximum of {settings.max_batch_size} images",
            )

        # Validate all files first
        for img in images:
            _validate_image_file(img)

        # Read all file contents off the event loop
        image_bytes_list = []
        for img in images:
            data = await asyncio.get_event_loop().run_in_executor(
                _executor, _read_file_chunks, img
            )
            image_bytes_list.append(data)

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                _executor, _classify_batch_sync, image_bytes_list
            )
        except ValueError as exc:
            logger.warning(f"Failed to process batch: {exc}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Corrupt or unreadable image in batch: {exc}",
            ) from exc

        span.set_attribute("batch_size", len(result.predictions))
        logger.info(f"Batch predict: {len(result.predictions)} images processed")
        return result
