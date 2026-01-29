export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">About This Project</h1>

      <div className="prose prose-invert max-w-none">
        <section className="card mb-8">
          <h2 className="text-xl font-semibold mb-4">Scientific Background</h2>
          <p className="text-space-300 mb-4">
            When an exoplanet transits its host star, a small fraction of starlight passes 
            through the planet's atmosphere before reaching us. This starlight carries 
            absorption signatures from atmospheric molecules like water vapor (H₂O), 
            carbon dioxide (CO₂), methane (CH₄), and others.
          </p>
          <p className="text-space-300">
            However, these signals are incredibly weak—typically only 10-100 parts per million 
            of the stellar flux—and are contaminated by various noise sources including stellar 
            variability, instrumental systematics, and photon noise.
          </p>
        </section>

        <section className="card mb-8">
          <h2 className="text-xl font-semibold mb-4">Our Approach</h2>
          <p className="text-space-300 mb-4">
            This platform combines traditional spectroscopic analysis with modern deep learning 
            techniques to extract atmospheric information from noisy observations:
          </p>
          <ul className="list-disc list-inside text-space-300 space-y-2 ml-4">
            <li>
              <strong className="text-white">Neural Network Denoising:</strong> Trained on 
              synthetic spectra generated with radiative transfer codes
            </li>
            <li>
              <strong className="text-white">Feature Extraction:</strong> Automated identification 
              of molecular absorption features
            </li>
            <li>
              <strong className="text-white">Bayesian Retrieval:</strong> MCMC-based atmospheric 
              parameter estimation with full uncertainty quantification
            </li>
            <li>
              <strong className="text-white">Interactive Analysis:</strong> Real-time visualization 
              and comparison with model atmospheres
            </li>
          </ul>
        </section>

        <section className="card mb-8">
          <h2 className="text-xl font-semibold mb-4">Technology Stack</h2>
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="bg-space-800/50 rounded-lg p-4">
              <h3 className="font-semibold mb-2">Frontend</h3>
              <ul className="text-space-400 text-sm space-y-1">
                <li>React 18</li>
                <li>Tailwind CSS</li>
                <li>Plotly.js</li>
                <li>Vite</li>
              </ul>
            </div>
            <div className="bg-space-800/50 rounded-lg p-4">
              <h3 className="font-semibold mb-2">Backend</h3>
              <ul className="text-space-400 text-sm space-y-1">
                <li>FastAPI</li>
                <li>Python 3.10+</li>
                <li>Astropy</li>
                <li>NumPy/SciPy</li>
              </ul>
            </div>
            <div className="bg-space-800/50 rounded-lg p-4">
              <h3 className="font-semibold mb-2">Machine Learning</h3>
              <ul className="text-space-400 text-sm space-y-1">
                <li>PyTorch 2.0</li>
                <li>scikit-learn</li>
                <li>emcee (MCMC)</li>
              </ul>
            </div>
            <div className="bg-space-800/50 rounded-lg p-4">
              <h3 className="font-semibold mb-2">Data Sources</h3>
              <ul className="text-space-400 text-sm space-y-1">
                <li>NASA Exoplanet Archive</li>
                <li>HST/WFC3 Spectra</li>
                <li>JWST Observations</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="card">
          <h2 className="text-xl font-semibold mb-4">Contact & Contributing</h2>
          <p className="text-space-300 mb-4">
            This is an open-source research project. We welcome contributions from the 
            astronomical and machine learning communities.
          </p>
          <div className="flex flex-wrap gap-4">
            <a 
              href="https://github.com" 
              className="btn-secondary inline-flex items-center gap-2"
            >
              GitHub Repository
            </a>
            <a 
              href="/docs" 
              className="btn-secondary inline-flex items-center gap-2"
            >
              Documentation
            </a>
          </div>
        </section>
      </div>
    </div>
  )
}
