# Architecture Overview

This document describes the high-level architecture of the Exoplanet Spectrum Recovery platform.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Browser                              │
│                    (React + Tailwind CSS)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Spectrum   │  │   Models     │  │  Analyses    │          │
│  │   Routes     │  │   Routes     │  │   Routes     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│           │                │                 │                   │
│           ▼                ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Service Layer                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │  Spectrum   │  │     ML      │  │   Data      │      │    │
│  │  │  Service    │  │  Service    │  │  Service    │      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PyTorch ML Models                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Denoiser   │  │  Retrieval   │  │   Feature    │          │
│  │   Network    │  │   Network    │  │  Detector    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Storage                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  SQLite/     │  │    File      │  │   Model      │          │
│  │  PostgreSQL  │  │   Storage    │  │ Checkpoints  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### Frontend (React + Tailwind)

The frontend provides an interactive web interface for:

- **Spectrum Upload**: Drag-and-drop file upload supporting FITS, CSV, JSON
- **Visualization**: Plotly-based interactive charts
- **Analysis Controls**: Model selection, parameter configuration
- **Results Display**: SNR metrics, detected features, uncertainty estimates

Key technologies:
- React 18 with hooks
- Tailwind CSS for styling
- Plotly.js for visualization
- React Query for server state management
- Zustand for client state management

### Backend (FastAPI)

The backend handles:

- **REST API**: RESTful endpoints for all operations
- **File Processing**: Parsing FITS, CSV, JSON spectral data
- **ML Inference**: Loading and running PyTorch models
- **Data Management**: Storing analysis results

Architecture patterns:
- Clean architecture with service layer
- Pydantic models for validation
- Async/await for I/O operations

### ML Models (PyTorch)

Three main model types:

1. **Spectrum Denoiser**
   - CNN-based (v1) or Transformer-based (v2)
   - Trained on synthetic spectra
   - Residual learning for noise removal

2. **Retrieval Network**
   - Predicts atmospheric parameters
   - Uncertainty estimation via MC dropout
   - Multi-head output for different parameters

3. **Feature Detector**
   - Identifies molecular absorption features
   - Multi-label classification
   - Localization of features in wavelength space

## Data Flow

### Spectrum Analysis Pipeline

```
1. User uploads spectrum file
   │
2. Backend validates and parses file
   │
3. Data is normalized and prepared
   │
4. ML model performs inference
   │
5. Results are post-processed
   │
6. Response sent to frontend
   │
7. Visualization updated in browser
```

### Model Training Pipeline

```
1. Generate/load synthetic spectra
   │
2. Apply data augmentation
   │
3. Train model with validation
   │
4. Save best checkpoint
   │
5. Evaluate on test set
   │
6. Deploy to production
```

## Deployment Options

### Local Development

```bash
# Backend
uvicorn app.main:app --reload

# Frontend
npm run dev
```

### Docker

```bash
docker-compose up --build
```

### Cloud Deployment

Recommended platforms:
- **Backend**: AWS Lambda, Google Cloud Run, Azure Container Apps
- **Frontend**: Vercel, Netlify, AWS Amplify
- **GPU Inference**: AWS SageMaker, Google Vertex AI

## Performance Considerations

- Model checkpoints are loaded once at startup
- Batch processing for multiple spectra
- GPU acceleration when available
- Response caching for repeated queries

## Security

- Input validation on all endpoints
- File type verification
- CORS configuration for production
- Rate limiting (recommended for production)
