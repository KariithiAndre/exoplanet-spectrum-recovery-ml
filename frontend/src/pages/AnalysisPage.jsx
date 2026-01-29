import { useState } from 'react'
import SpectrumUploader from '../components/SpectrumUploader'
import SpectrumPlot from '../components/SpectrumPlot'
import { analyzeSpectrum } from '../services/api'
import toast from 'react-hot-toast'

export default function AnalysisPage() {
  const [spectrumData, setSpectrumData] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [analysisResults, setAnalysisResults] = useState(null)

  const handleUpload = async (file) => {
    setIsLoading(true)
    try {
      const result = await analyzeSpectrum(file)
      setSpectrumData(result.spectrum)
      setAnalysisResults(result.analysis)
      toast.success('Spectrum analyzed successfully!')
    } catch (error) {
      toast.error(error.message || 'Failed to analyze spectrum')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-2">Spectrum Analysis</h1>
      <p className="text-space-400 mb-8">
        Upload your exoplanet spectrum data for ML-powered analysis and recovery.
      </p>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Left Panel - Upload & Controls */}
        <div className="space-y-6">
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Upload Spectrum</h2>
            <SpectrumUploader onUpload={handleUpload} isLoading={isLoading} />
          </div>

          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Analysis Options</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-space-300 mb-2">
                  Recovery Model
                </label>
                <select className="input-field">
                  <option value="denoiser_v1">Spectrum Denoiser v1</option>
                  <option value="denoiser_v2">Spectrum Denoiser v2</option>
                  <option value="retrieval_net">Retrieval Network</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-space-300 mb-2">
                  Wavelength Range (μm)
                </label>
                <div className="flex gap-2">
                  <input
                    type="number"
                    placeholder="0.3"
                    className="input-field"
                    step="0.1"
                  />
                  <span className="text-space-400 self-center">to</span>
                  <input
                    type="number"
                    placeholder="15.0"
                    className="input-field"
                    step="0.1"
                  />
                </div>
              </div>
              <button
                className="btn-primary w-full"
                disabled={!spectrumData || isLoading}
              >
                {isLoading ? 'Processing...' : 'Run Analysis'}
              </button>
            </div>
          </div>
        </div>

        {/* Right Panel - Visualization */}
        <div className="lg:col-span-2 space-y-6">
          <div className="card">
            <SpectrumPlot
              data={spectrumData}
              title="Transmission Spectrum"
              height={450}
            />
          </div>

          {analysisResults && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Analysis Results</h2>
              <div className="grid sm:grid-cols-2 gap-4">
                <div className="bg-space-800/50 rounded-lg p-4">
                  <p className="text-space-400 text-sm">Signal-to-Noise Ratio</p>
                  <p className="text-2xl font-semibold">{analysisResults.snr?.toFixed(1) || 'N/A'}</p>
                </div>
                <div className="bg-space-800/50 rounded-lg p-4">
                  <p className="text-space-400 text-sm">Detected Features</p>
                  <p className="text-2xl font-semibold">{analysisResults.features?.length || 0}</p>
                </div>
                <div className="bg-space-800/50 rounded-lg p-4">
                  <p className="text-space-400 text-sm">Confidence</p>
                  <p className="text-2xl font-semibold">{(analysisResults.confidence * 100)?.toFixed(1) || 'N/A'}%</p>
                </div>
                <div className="bg-space-800/50 rounded-lg p-4">
                  <p className="text-space-400 text-sm">Processing Time</p>
                  <p className="text-2xl font-semibold">{analysisResults.processingTime || 'N/A'}ms</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
