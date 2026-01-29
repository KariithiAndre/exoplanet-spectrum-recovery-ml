export default function DataPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-2">Data Management</h1>
      <p className="text-space-400 mb-8">
        Manage your spectral data, view catalogs, and access training datasets.
      </p>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Recent Uploads */}
        <div className="lg:col-span-2 card">
          <h2 className="text-lg font-semibold mb-4">Recent Uploads</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-space-400 text-sm border-b border-space-800">
                  <th className="pb-3">Filename</th>
                  <th className="pb-3">Type</th>
                  <th className="pb-3">Date</th>
                  <th className="pb-3">Status</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                <tr className="border-b border-space-800/50">
                  <td className="py-3 font-mono text-space-300">wasp-39b_nirspec.fits</td>
                  <td className="py-3 text-space-400">FITS</td>
                  <td className="py-3 text-space-400">2026-01-28</td>
                  <td className="py-3">
                    <span className="px-2 py-1 bg-green-900/50 text-green-400 rounded text-xs">
                      Processed
                    </span>
                  </td>
                </tr>
                <tr className="border-b border-space-800/50">
                  <td className="py-3 font-mono text-space-300">hd209458b_wfc3.csv</td>
                  <td className="py-3 text-space-400">CSV</td>
                  <td className="py-3 text-space-400">2026-01-27</td>
                  <td className="py-3">
                    <span className="px-2 py-1 bg-green-900/50 text-green-400 rounded text-xs">
                      Processed
                    </span>
                  </td>
                </tr>
                <tr className="border-b border-space-800/50">
                  <td className="py-3 font-mono text-space-300">trappist-1e_miri.fits</td>
                  <td className="py-3 text-space-400">FITS</td>
                  <td className="py-3 text-space-400">2026-01-26</td>
                  <td className="py-3">
                    <span className="px-2 py-1 bg-yellow-900/50 text-yellow-400 rounded text-xs">
                      Processing
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Storage Stats */}
        <div className="space-y-6">
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Storage</h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-space-400">Raw Data</span>
                  <span>2.4 GB</span>
                </div>
                <div className="h-2 bg-space-800 rounded-full overflow-hidden">
                  <div className="h-full bg-space-500 rounded-full" style={{ width: '24%' }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-space-400">Processed</span>
                  <span>1.8 GB</span>
                </div>
                <div className="h-2 bg-space-800 rounded-full overflow-hidden">
                  <div className="h-full bg-cosmic-500 rounded-full" style={{ width: '18%' }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-space-400">Models</span>
                  <span>0.5 GB</span>
                </div>
                <div className="h-2 bg-space-800 rounded-full overflow-hidden">
                  <div className="h-full bg-nebula-500 rounded-full" style={{ width: '5%' }} />
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
            <div className="space-y-2">
              <button className="btn-secondary w-full text-left">
                📁 Browse Catalogs
              </button>
              <button className="btn-secondary w-full text-left">
                📥 Import Data
              </button>
              <button className="btn-secondary w-full text-left">
                📤 Export Results
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
