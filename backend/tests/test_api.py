"""Tests for spectrum API endpoints."""

import pytest
from fastapi.testclient import TestClient
from io import BytesIO
import json

from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self):
        """Test health endpoint returns healthy status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "system" in data
        assert "gpu" in data


class TestSpectrumEndpoints:
    """Tests for spectrum analysis endpoints."""
    
    def test_analyze_csv_spectrum(self):
        """Test analyzing a CSV spectrum file."""
        # Create sample CSV data
        csv_content = "wavelength,flux,error\n"
        csv_content += "\n".join([
            f"{0.5 + i*0.01},{100 + i*0.1},{5.0}"
            for i in range(100)
        ])
        
        files = {
            "file": ("test_spectrum.csv", BytesIO(csv_content.encode()), "text/csv")
        }
        
        response = client.post("/api/spectrum/analyze", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert "spectrum" in data
        assert "analysis" in data
        assert data["filename"] == "test_spectrum.csv"
    
    def test_analyze_json_spectrum(self):
        """Test analyzing a JSON spectrum file."""
        spectrum_data = {
            "wavelength": [0.5 + i*0.01 for i in range(100)],
            "flux": [100 + i*0.1 for i in range(100)],
            "error": [5.0] * 100
        }
        
        files = {
            "file": (
                "test_spectrum.json",
                BytesIO(json.dumps(spectrum_data).encode()),
                "application/json"
            )
        }
        
        response = client.post("/api/spectrum/analyze", files=files)
        assert response.status_code == 200
    
    def test_analyze_unsupported_format(self):
        """Test that unsupported formats are rejected."""
        files = {
            "file": ("test.txt", BytesIO(b"invalid data"), "text/plain")
        }
        
        response = client.post("/api/spectrum/analyze", files=files)
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]


class TestModelsEndpoints:
    """Tests for ML models endpoints."""
    
    def test_list_models(self):
        """Test listing available models."""
        response = client.get("/api/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert data["total"] > 0
    
    def test_get_model_details(self):
        """Test getting model details."""
        response = client.get("/api/models/spectrum_denoiser_v2")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == "spectrum_denoiser_v2"
        assert "accuracy" in data
    
    def test_get_nonexistent_model(self):
        """Test getting a model that doesn't exist."""
        response = client.get("/api/models/nonexistent_model")
        assert response.status_code == 404


class TestAnalysesEndpoints:
    """Tests for analyses history endpoints."""
    
    def test_list_analyses(self):
        """Test listing recent analyses."""
        response = client.get("/api/analyses")
        assert response.status_code == 200
        
        data = response.json()
        assert "analyses" in data
        assert "total" in data
    
    def test_list_analyses_with_pagination(self):
        """Test pagination parameters."""
        response = client.get("/api/analyses?limit=5&offset=0")
        assert response.status_code == 200
        
        data = response.json()
        assert data["limit"] == 5
        assert data["offset"] == 0
