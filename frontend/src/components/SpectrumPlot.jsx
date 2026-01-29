import Plot from 'react-plotly.js'
import { useMemo } from 'react'

const defaultLayout = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(30, 27, 75, 0.5)',
  font: {
    color: '#a5b4fc',
    family: 'Inter, sans-serif',
  },
  xaxis: {
    title: 'Wavelength (μm)',
    gridcolor: 'rgba(99, 102, 241, 0.2)',
    zerolinecolor: 'rgba(99, 102, 241, 0.3)',
  },
  yaxis: {
    title: 'Transit Depth (ppm)',
    gridcolor: 'rgba(99, 102, 241, 0.2)',
    zerolinecolor: 'rgba(99, 102, 241, 0.3)',
  },
  margin: { l: 60, r: 30, t: 40, b: 50 },
  showlegend: true,
  legend: {
    x: 1,
    xanchor: 'right',
    y: 1,
    bgcolor: 'rgba(30, 27, 75, 0.8)',
    bordercolor: 'rgba(99, 102, 241, 0.3)',
    borderwidth: 1,
  },
  hovermode: 'x unified',
}

const defaultConfig = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  displaylogo: false,
}

export default function SpectrumPlot({
  data,
  title = 'Spectrum Analysis',
  height = 400,
  showError = true,
}) {
  const plotData = useMemo(() => {
    if (!data) return []

    const traces = []

    // Main spectrum trace
    if (data.wavelength && data.flux) {
      traces.push({
        x: data.wavelength,
        y: data.flux,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Observed',
        line: { color: '#818cf8', width: 2 },
        marker: { size: 4 },
      })
    }

    // Error bars
    if (showError && data.error) {
      traces[0].error_y = {
        type: 'data',
        array: data.error,
        visible: true,
        color: 'rgba(129, 140, 248, 0.4)',
      }
    }

    // Recovered spectrum (if available)
    if (data.recovered) {
      traces.push({
        x: data.wavelength,
        y: data.recovered,
        type: 'scatter',
        mode: 'lines',
        name: 'Recovered',
        line: { color: '#f43f5e', width: 2, dash: 'solid' },
      })
    }

    // Model fit (if available)
    if (data.model) {
      traces.push({
        x: data.wavelength,
        y: data.model,
        type: 'scatter',
        mode: 'lines',
        name: 'Model',
        line: { color: '#22c55e', width: 2, dash: 'dot' },
      })
    }

    return traces
  }, [data, showError])

  const layout = useMemo(() => ({
    ...defaultLayout,
    title: {
      text: title,
      font: { size: 16, color: '#e0e7ff' },
    },
    height,
  }), [title, height])

  if (!data) {
    return (
      <div 
        className="flex items-center justify-center bg-space-900/50 rounded-xl border border-space-800"
        style={{ height }}
      >
        <p className="text-space-400">No spectrum data to display</p>
      </div>
    )
  }

  return (
    <div className="rounded-xl overflow-hidden border border-space-800">
      <Plot
        data={plotData}
        layout={layout}
        config={defaultConfig}
        className="w-full"
      />
    </div>
  )
}
