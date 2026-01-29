import { Outlet, Link, useLocation } from 'react-router-dom'
import { clsx } from 'clsx'

const navigation = [
  { name: 'Home', path: '/' },
  { name: 'Analysis', path: '/analysis' },
  { name: 'Models', path: '/models' },
  { name: 'Data', path: '/data' },
  { name: 'About', path: '/about' },
]

export default function Layout() {
  const location = useLocation()

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navigation */}
      <nav className="bg-space-900/80 backdrop-blur-md border-b border-space-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🪐</span>
              <span className="font-semibold text-lg">ExoSpectrum</span>
            </div>
            
            <div className="flex items-center gap-1">
              {navigation.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={clsx(
                    'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                    location.pathname === item.path
                      ? 'bg-space-700 text-white'
                      : 'text-space-300 hover:text-white hover:bg-space-800'
                  )}
                >
                  {item.name}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="bg-space-900/50 border-t border-space-800 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-space-400 text-sm">
              © 2026 Exoplanet Spectrum Recovery Platform
            </p>
            <div className="flex gap-6 text-sm text-space-400">
              <a href="https://github.com" className="hover:text-white transition-colors">
                GitHub
              </a>
              <a href="/docs" className="hover:text-white transition-colors">
                Documentation
              </a>
              <a href="/api/docs" className="hover:text-white transition-colors">
                API
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
