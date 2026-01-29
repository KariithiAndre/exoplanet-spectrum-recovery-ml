export default function ModelsPage() {
  const models = [
    {
      id: 'spectrum_denoiser_v1',
      name: 'Spectrum Denoiser v1',
      description: 'Convolutional neural network for spectral noise reduction',
      accuracy: 92.3,
      status: 'active',
      trainedOn: '50,000 synthetic spectra',
    },
    {
      id: 'spectrum_denoiser_v2',
      name: 'Spectrum Denoiser v2',
      description: 'Transformer-based architecture with attention mechanisms',
      accuracy: 94.7,
      status: 'active',
      trainedOn: '100,000 synthetic + 500 real spectra',
    },
    {
      id: 'retrieval_net',
      name: 'Retrieval Network',
      description: 'End-to-end atmospheric parameter estimation',
      accuracy: 89.1,
      status: 'beta',
      trainedOn: '200,000 petitRADTRANS models',
    },
    {
      id: 'feature_detector',
      name: 'Feature Detector',
      description: 'Molecular absorption feature identification',
      accuracy: 96.2,
      status: 'active',
      trainedOn: '75,000 labeled spectra',
    },
  ]

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-2">ML Models</h1>
      <p className="text-space-400 mb-8">
        PyTorch-based models for spectrum analysis and atmospheric retrieval.
      </p>

      <div className="grid md:grid-cols-2 gap-6">
        {models.map((model) => (
          <div key={model.id} className="card hover:border-space-600 transition-colors">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold">{model.name}</h3>
                <span
                  className={`inline-block px-2 py-1 text-xs rounded-full mt-1 ${
                    model.status === 'active'
                      ? 'bg-green-900/50 text-green-400'
                      : 'bg-yellow-900/50 text-yellow-400'
                  }`}
                >
                  {model.status}
                </span>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-space-200">{model.accuracy}%</p>
                <p className="text-xs text-space-400">accuracy</p>
              </div>
            </div>
            <p className="text-space-400 mb-4">{model.description}</p>
            <p className="text-sm text-space-500">
              <span className="text-space-400">Trained on:</span> {model.trainedOn}
            </p>
            <div className="flex gap-2 mt-4">
              <button className="btn-secondary text-sm flex-1">View Details</button>
              <button className="btn-primary text-sm flex-1">Use Model</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
