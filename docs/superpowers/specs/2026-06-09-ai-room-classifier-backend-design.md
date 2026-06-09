# AI Room Classifier — Backend Design

**Date:** 2026-06-09
**Scope:** Backend service only (inference API + Docker packaging). Training pipeline and dataset are managed by teammates separately.

---

## Overview

A FastAPI service that loads a pre-trained PyTorch model and classifies real estate photos into one of six room types: bedroom, living room, kitchen, bathroom, balcony, hallway. Exposes a single-image and a batch prediction endpoint. Designed to integrate into the BrighterProject ecosystem as `brighter-ai-blehhing`, following the same conventions as the other microservices.

---

## Room Classes

- bedroom
- living_room
- kitchen
- bathroom
- balcony
- hallway

---

## Architecture

Single stateless FastAPI service. At startup the model file is loaded from `MODEL_PATH` (env var) into memory. All inference is synchronous within an async endpoint handler (offloaded to a thread pool to avoid blocking the event loop). No database, no auth, no background workers.

```
Request → FastAPI router → preprocessing.py → model.py → JSON response
```

---

## Endpoints

### `POST /predict`
Classify a single image.

- **Body:** `multipart/form-data` with field `image` (JPEG or PNG, max 10 MB)
- **Response 200:**
  ```json
  { "room_type": "bedroom", "confidence": 0.94 }
  ```

### `POST /predict/batch`
Classify up to 20 images in one request.

- **Body:** `multipart/form-data` with field `images` (list of files, same constraints)
- **Response 200:**
  ```json
  {
    "predictions": [
      { "room_type": "kitchen", "confidence": 0.91 },
      { "room_type": "bathroom", "confidence": 0.87 }
    ]
  }
  ```

### `GET /health`
Liveness check.

- **Response 200:**
  ```json
  { "status": "ok", "model_loaded": true }
  ```

### `GET /metrics`
Prometheus metrics in text format. Wired via the same `setup_telemetry()` pattern used across all BrighterProject services. OTEL spans are emitted for each predict request; a Prometheus counter tracks prediction counts per room type.

---

## Project Structure

```
brighter-ai-blehhing/
  app/
    routers/
      predict.py         # /predict and /predict/batch handlers
    model.py             # model loading at startup, predict() function
    preprocessing.py     # resize to 224×224, ImageNet normalization
    schemas.py           # Pydantic request/response models
    settings.py          # MODEL_PATH and other env vars
    telemetry.py         # setup_telemetry() — same pattern as other services
    logging.py           # loguru setup — same pattern as other services
  main.py                # FastAPI app wiring
  models/                # .pt file lives here (gitignored)
  Dockerfile
  pyproject.toml
```

### Responsibility boundaries

| Module | Responsibility |
|---|---|
| `routers/predict.py` | HTTP concerns: parse multipart, validate, return response |
| `preprocessing.py` | Image transforms: decode bytes → tensor (resize, normalize) |
| `model.py` | Model lifecycle: load at startup (crash on failure), `predict(tensor) → (class, confidence)` |
| `schemas.py` | Pydantic models for request/response shapes |
| `telemetry.py` | OTEL + Prometheus wiring, `/metrics` endpoint |

---

## Image Preprocessing

Preprocessing runs on the backend — the client sends raw JPEG/PNG. This keeps ML-specific constants (ImageNet mean/std, input size) server-side and avoids coupling every client to the preprocessing contract.

**Single image pipeline:**
1. Read file in chunks (max 10 MB enforced during read — do not trust `Content-Length`)
2. Decode bytes → PIL Image; raise 422 if corrupt
3. Resize to 224×224
4. Convert to tensor
5. Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

**Batch pipeline:**
Same per-image steps 1–5, then `torch.stack()` all tensors into a single `[N, 3, 224, 224]` tensor and run one forward pass through the model. Do not loop `predict()` N times — that forgoes the entire benefit of batching.

---

## Error Handling

All error responses use the `{"detail": "..."}` envelope consistent with other BrighterProject services.

| Condition | Status |
|---|---|
| Non-image file type | 422 |
| Image > 10 MB | 422 |
| Corrupt/unreadable image | 422 |
| Batch size > 20 | 422 |
| Model file not found at startup | Service raises at startup and crashes (fail-fast) — let the container orchestrator restart it |

---

## Docker & Infrastructure

Base image: `pytorch/pytorch:2.x-cpu` (CPU-only). EfficientNet-B0 is lightweight enough that CPU inference is fast enough for production — using a CUDA base image would add gigabytes of unnecessary NVIDIA drivers.

The `.pt` model file is **not** baked into the image — it is mounted as a volume. This allows teammates to swap the model file without rebuilding the container.

```yaml
# Addition to brighter-compose/docker-compose.yml
ai-ms:
  build: ../brighter-ai-blehhing
  ports:
    - "8005:8005"
  volumes:
    - ../brighter-ai-blehhing/models:/models
  environment:
    MODEL_PATH: /models/room_classifier.pt
    OTEL_SDK_DISABLED: "true"
```

Traefik route: `/ai` prefix, priority 5 (same as other backend services).

---

## Testing

Mock the model layer (`app.model.predict`), not the filesystem. No actual `.pt` file required to run tests.

```
tests/
  conftest.py      # fixtures: test_client, mock_model (AsyncMock returning ("bedroom", 0.95))
  test_predict.py  # happy path, wrong file type, oversized image, corrupt image
  test_batch.py    # happy path, mixed files, over 20 images
  test_health.py   # /health response, /metrics responds
```

Coverage target: 80%+, matching the rest of BrighterProject.

---

## Technology Choices

| Technology | Reason |
|---|---|
| FastAPI | Matches existing BrighterProject stack |
| PyTorch (torchvision) | Industry-standard for CV research; EfficientNet-B0 available via `torchvision.models` |
| EfficientNet-B0 | Strong accuracy/compute trade-off for image classification at 224×224 |
| loguru | Same logging setup as other services |
| OTEL + Prometheus | Same observability setup as other services |
| Docker + CPU PyTorch base image | Slim image (no NVIDIA drivers); EfficientNet-B0 is fast enough on CPU for production inference |

---

## Out of Scope

- Model training pipeline (teammates' responsibility)
- Authentication / authorization
- Image storage
- Model hot-swapping (restart-to-update is sufficient)
- Frontend
