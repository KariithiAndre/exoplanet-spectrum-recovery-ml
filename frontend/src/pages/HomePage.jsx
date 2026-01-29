import { Link } from 'react-router-dom'

const features = [
  {
    icon: '🔬',
    title: 'Spectral Analysis',
    description: 'Process and analyze exoplanet transmission spectra with advanced denoising algorithms.',
  },
  {
    icon: '🧠',
    title: 'ML-Powered Recovery',
    description: 'Deep learning models trained on synthetic data to recover atmospheric features.',
  },
  {
    icon: '📊',
    title: 'Interactive Visualization',
    description: 'Explore spectra with Plotly-based interactive charts and comparison tools.',
  },
  {
    icon: '🌍',
    title: 'Atmospheric Retrieval',
    description: 'Bayesian inference for atmospheric composition with uncertainty quantification.',
  },
]

const stats = [
  { label: 'Exoplanets Analyzed', value: '500+' },
  { label: 'Model Accuracy', value: '94.2%' },
  { label: 'Processing Speed', value: '<1s' },
  { label: 'Spectral Bands', value: '0.3-15μm' },
]

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 px-4 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-space-900/50 to-transparent" />
        <div className="absolute top-20 left-1/4 w-96 h-96 bg-cosmic-600/20 rounded-full blur-3xl" />
        <div className="absolute bottom-20 right-1/4 w-96 h-96 bg-space-600/20 rounded-full blur-3xl" />
        
        <div className="max-w-4xl mx-auto text-center relative z-10">
          <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-space-200 via-cosmic-300 to-nebula-300 bg-clip-text text-transparent">
            Exoplanet Spectrum Recovery
          </h1>
          <p className="text-xl text-space-300 mb-8 max-w-2xl mx-auto">
            A research-grade platform for analyzing and recovering exoplanet atmospheric 
            spectra using deep learning and Bayesian inference.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/analysis" className="btn-primary text-lg px-8 py-3">
              Start Analysis
            </Link>
            <Link to="/about" className="btn-secondary text-lg px-8 py-3">
              Learn More
            </Link>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-12 border-y border-space-800 bg-space-900/30">
        <div className="max-w-6xl mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat) => (
              <div key={stat.label} className="text-center">
                <p className="text-3xl font-bold text-white mb-1">{stat.value}</p>
                <p className="text-space-400 text-sm">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">
            Powerful Analysis Tools
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature) => (
              <div key={feature.title} className="card hover:border-space-600 transition-colors">
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-space-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-gradient-to-r from-space-900 via-space-800 to-space-900">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Explore?</h2>
          <p className="text-space-300 mb-8">
            Upload your spectrum data and let our ML models help you uncover 
            atmospheric signatures hidden in the noise.
          </p>
          <Link to="/analysis" className="btn-primary text-lg px-8 py-3">
            Get Started
          </Link>
        </div>
      </section>
    </div>
  )
}
