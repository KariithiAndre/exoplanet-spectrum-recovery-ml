import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Health check
export async function healthCheck() {
  const response = await apiClient.get('/health')
  return response.data
}

// Spectrum analysis
export async function analyzeSpectrum(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await apiClient.post('/spectrum/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

// Get available models
export async function getModels() {
  const response = await apiClient.get('/models')
  return response.data
}

// Run recovery with specific model
export async function recoverSpectrum(spectrumId, modelId, options = {}) {
  const response = await apiClient.post('/spectrum/recover', {
    spectrum_id: spectrumId,
    model_id: modelId,
    ...options,
  })
  return response.data
}

// Get spectrum by ID
export async function getSpectrum(spectrumId) {
  const response = await apiClient.get(`/spectrum/${spectrumId}`)
  return response.data
}

// List recent analyses
export async function listAnalyses(limit = 10) {
  const response = await apiClient.get('/analyses', {
    params: { limit },
  })
  return response.data
}

// Export results
export async function exportResults(analysisId, format = 'csv') {
  const response = await apiClient.get(`/analyses/${analysisId}/export`, {
    params: { format },
    responseType: 'blob',
  })
  return response.data
}

export default apiClient
