# API Reference

This document provides detailed documentation for the Exoplanet Spectrum Recovery API.

## Base URL

```
http://localhost:8000/api
```

## Authentication

Currently, the API does not require authentication for local development. Production deployments should implement appropriate authentication.

## Endpoints

### Health Check

#### `GET /api/health`

Check API health and system status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-29T12:00:00Z",
  "system": {
    "platform": "Linux",
    "python_version": "3.10.0",
    "pytorch_version": "2.0.0"
  },
  "gpu": {
    "available": true,
    "info": {
      "name": "NVIDIA RTX 4090",
      "memory_allocated": "2.5 GB",
      "memory_cached": "4.0 GB"
    }
  }
}
```

---

### Spectrum Analysis

#### `POST /api/spectrum/analyze`

Upload and analyze a spectrum file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File upload (FITS, CSV, or JSON)

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "wasp-39b_spectrum.fits",
  "spectrum": {
    "wavelength": [0.5, 0.6, ...],
    "flux": [100.2, 102.5, ...],
    "error": [5.1, 4.8, ...]
  },
  "analysis": {
    "snr": 45.2,
    "confidence": 0.92,
    "processing_time": 847,
    "features": [
      {"molecule": "H2O", "wavelength": 1.4, "significance": 4.5}
    ]
  },
  "created_at": "2026-01-29T12:00:00Z"
}
```

#### `GET /api/spectrum/{spectrum_id}`

Retrieve a previously analyzed spectrum.

#### `POST /api/spectrum/recover`

Apply ML-based spectral recovery.

**Request Body:**
```json
{
  "spectrum_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_id": "spectrum_denoiser_v2",
  "wavelength_min": 0.5,
  "wavelength_max": 5.0
}
```

---

### Models

#### `GET /api/models`

List available ML models.

**Response:**
```json
{
  "models": [
    {
      "id": "spectrum_denoiser_v2",
      "name": "Spectrum Denoiser v2",
      "description": "Transformer-based architecture",
      "version": "2.0.0",
      "accuracy": 94.7,
      "status": "active"
    }
  ],
  "total": 4
}
```

#### `GET /api/models/{model_id}`

Get details for a specific model.

#### `POST /api/models/{model_id}/load`

Load a model into memory for inference.

---

### Analyses

#### `GET /api/analyses`

List recent analyses.

**Query Parameters:**
- `limit` (int, default=10): Maximum results
- `offset` (int, default=0): Pagination offset

#### `GET /api/analyses/{analysis_id}`

Get detailed analysis results.

#### `GET /api/analyses/{analysis_id}/export`

Export analysis results.

**Query Parameters:**
- `format` (string): Output format (`csv`, `json`, `fits`)

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message description"
}
```

**Status Codes:**
- `400` - Bad Request (invalid input)
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

---

## Rate Limiting

No rate limiting is currently implemented for local development.

## Websocket API

*Coming soon* - Real-time analysis progress updates.
