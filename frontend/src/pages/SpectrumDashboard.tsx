/**
 * Interactive Spectrum Dashboard
 * 
 * Features:
 * - Raw vs processed spectrum comparison plots
 * - Absorption line overlays with labels
 * - Molecule highlight bands
 * - Zoomable wavelength graphs using Plotly
 * - Real-time analysis status
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { Data, Layout, PlotMouseEvent } from 'plotly.js';
import './SpectrumDashboard.css';

// ===== Type Definitions =====
interface SpectrumData {
  wavelengths: number[];
  flux: number[];
  uncertainties?: number[];
}

interface ProcessedSpectrum extends SpectrumData {
  continuum?: number[];
  normalized?: number[];
  snr?: number[];
}

interface AbsorptionLine {
  wavelength: number;
  molecule: string;
  transition: string;
  strength: 'strong' | 'medium' | 'weak';
  detected: boolean;
  confidence?: number;
}

interface MolecularBand {
  molecule: string;
  startWavelength: number;
  endWavelength: number;
  peakWavelength: number;
  color: string;
  opacity: number;
}

interface AnalysisResult {
  id: string;
  timestamp: string;
  rawSpectrum: SpectrumData;
  processedSpectrum: ProcessedSpectrum;
  detectedLines: AbsorptionLine[];
  molecularBands: MolecularBand[];
  molecules: { name: string; confidence: number; abundance?: number }[];
  planetClass: { type: string; probability: number }[];
  habitability: number;
}

interface DashboardState {
  showRaw: boolean;
  showProcessed: boolean;
  showContinuum: boolean;
  showNormalized: boolean;
  showUncertainties: boolean;
  showAbsorptionLines: boolean;
  showMolecularBands: boolean;
  selectedMolecules: string[];
  wavelengthRange: [number, number];
  plotHeight: number;
}

// ===== Molecular Data =====
const MOLECULAR_COLORS: Record<string, string> = {
  'H2O': '#00bfff',
  'CO2': '#ff6b6b',
  'CH4': '#ffd700',
  'O3': '#9b59b6',
  'O2': '#3498db',
  'NH3': '#e67e22',
  'CO': '#1abc9c',
  'N2O': '#e91e63',
  'SO2': '#795548',
  'H2S': '#607d8b',
  'HCN': '#ff5722',
  'C2H2': '#8bc34a',
  'C2H6': '#00bcd4',
  'PH3': '#9c27b0'
};

const MOLECULAR_BANDS: Record<string, { start: number; end: number; peak: number }[]> = {
  'H2O': [
    { start: 1.3, end: 1.5, peak: 1.4 },
    { start: 1.7, end: 2.1, peak: 1.9 },
    { start: 2.5, end: 2.9, peak: 2.7 },
    { start: 5.5, end: 7.5, peak: 6.3 }
  ],
  'CO2': [
    { start: 2.0, end: 2.1, peak: 2.0 },
    { start: 4.2, end: 4.4, peak: 4.3 },
    { start: 14.0, end: 16.0, peak: 15.0 }
  ],
  'CH4': [
    { start: 2.2, end: 2.4, peak: 2.3 },
    { start: 3.2, end: 3.5, peak: 3.3 },
    { start: 7.5, end: 8.0, peak: 7.7 }
  ],
  'O3': [
    { start: 9.4, end: 9.8, peak: 9.6 }
  ],
  'O2': [
    { start: 0.76, end: 0.77, peak: 0.765 },
    { start: 1.26, end: 1.28, peak: 1.27 }
  ],
  'NH3': [
    { start: 10.0, end: 11.0, peak: 10.5 }
  ],
  'CO': [
    { start: 4.6, end: 4.8, peak: 4.7 }
  ]
};

const ABSORPTION_LINES: AbsorptionLine[] = [
  { wavelength: 1.4, molecule: 'H2O', transition: 'ν1+ν3', strength: 'strong', detected: false },
  { wavelength: 1.9, molecule: 'H2O', transition: '2ν1', strength: 'strong', detected: false },
  { wavelength: 2.7, molecule: 'H2O', transition: 'ν1', strength: 'strong', detected: false },
  { wavelength: 4.3, molecule: 'CO2', transition: 'ν3', strength: 'strong', detected: false },
  { wavelength: 15.0, molecule: 'CO2', transition: 'ν2', strength: 'medium', detected: false },
  { wavelength: 3.3, molecule: 'CH4', transition: 'ν3', strength: 'strong', detected: false },
  { wavelength: 7.7, molecule: 'CH4', transition: 'ν4', strength: 'medium', detected: false },
  { wavelength: 9.6, molecule: 'O3', transition: 'ν3', strength: 'strong', detected: false },
  { wavelength: 0.765, molecule: 'O2', transition: 'A-band', strength: 'strong', detected: false },
  { wavelength: 10.5, molecule: 'NH3', transition: 'ν2', strength: 'medium', detected: false },
  { wavelength: 4.7, molecule: 'CO', transition: 'v=1-0', strength: 'medium', detected: false }
];

// ===== Generate Demo Data =====
const generateDemoData = (): AnalysisResult => {
  const numPoints = 1000;
  const wavelengths: number[] = [];
  const rawFlux: number[] = [];
  const processedFlux: number[] = [];
  const continuum: number[] = [];
  const uncertainties: number[] = [];
  
  const wlMin = 0.6;
  const wlMax = 28.0;
  
  // Generate wavelength grid
  for (let i = 0; i < numPoints; i++) {
    wavelengths.push(wlMin + (wlMax - wlMin) * (i / (numPoints - 1)));
  }
  
  // Generate flux with molecular features
  const detectedMolecules = ['H2O', 'CO2', 'CH4', 'O3'];
  
  for (let i = 0; i < numPoints; i++) {
    const wl = wavelengths[i];
    
    // Base continuum (blackbody-like)
    let cont = Math.exp(-0.5 * Math.pow((wl - 5) / 10, 2));
    continuum.push(cont);
    
    // Add molecular absorption
    let absorption = 1.0;
    for (const mol of detectedMolecules) {
      const bands = MOLECULAR_BANDS[mol] || [];
      for (const band of bands) {
        const distance = Math.abs(wl - band.peak) / ((band.end - band.start) / 2);
        if (distance < 3) {
          absorption *= (1 - 0.4 * Math.exp(-0.5 * distance * distance));
        }
      }
    }
    
    // Raw flux with noise
    const noise = (Math.random() - 0.5) * 0.1 * cont;
    rawFlux.push(cont * absorption + noise);
    
    // Processed flux (denoised)
    processedFlux.push(cont * absorption);
    
    // Uncertainties
    uncertainties.push(0.02 * cont);
  }
  
  // Mark detected lines
  const detectedLines = ABSORPTION_LINES.map(line => ({
    ...line,
    detected: detectedMolecules.includes(line.molecule),
    confidence: detectedMolecules.includes(line.molecule) ? 0.7 + Math.random() * 0.25 : 0
  }));
  
  // Generate molecular bands
  const molecularBands: MolecularBand[] = [];
  for (const mol of detectedMolecules) {
    const color = MOLECULAR_COLORS[mol] || '#888888';
    const bands = MOLECULAR_BANDS[mol] || [];
    for (const band of bands) {
      molecularBands.push({
        molecule: mol,
        startWavelength: band.start,
        endWavelength: band.end,
        peakWavelength: band.peak,
        color,
        opacity: 0.15
      });
    }
  }
  
  return {
    id: 'demo_analysis_001',
    timestamp: new Date().toISOString(),
    rawSpectrum: {
      wavelengths,
      flux: rawFlux,
      uncertainties
    },
    processedSpectrum: {
      wavelengths,
      flux: processedFlux,
      continuum,
      normalized: processedFlux.map((f, i) => f / continuum[i])
    },
    detectedLines,
    molecularBands,
    molecules: [
      { name: 'H2O', confidence: 0.94, abundance: 0.002 },
      { name: 'CO2', confidence: 0.91, abundance: 0.0004 },
      { name: 'CH4', confidence: 0.78, abundance: 0.00002 },
      { name: 'O3', confidence: 0.65, abundance: 0.000001 }
    ],
    planetClass: [
      { type: 'Terrestrial', probability: 0.72 },
      { type: 'Super-Earth', probability: 0.21 },
      { type: 'Mini-Neptune', probability: 0.05 },
      { type: 'Other', probability: 0.02 }
    ],
    habitability: 0.68
  };
};

// ===== Main Dashboard Component =====
const SpectrumDashboard: React.FC = () => {
  // State
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'comparison' | 'normalized' | 'lines'>('comparison');
  const [hoveredLine, setHoveredLine] = useState<AbsorptionLine | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<{ wl: number; flux: number } | null>(null);
  
  const [dashboardState, setDashboardState] = useState<DashboardState>({
    showRaw: true,
    showProcessed: true,
    showContinuum: false,
    showNormalized: false,
    showUncertainties: true,
    showAbsorptionLines: true,
    showMolecularBands: true,
    selectedMolecules: ['H2O', 'CO2', 'CH4', 'O3'],
    wavelengthRange: [0.6, 28],
    plotHeight: 450
  });
  
  // Load demo data
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnalysis(generateDemoData());
      setIsLoading(false);
    }, 1000);
    return () => clearTimeout(timer);
  }, []);
  
  // Toggle functions
  const toggleState = useCallback((key: keyof DashboardState) => {
    setDashboardState(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  }, []);
  
  const toggleMolecule = useCallback((molecule: string) => {
    setDashboardState(prev => ({
      ...prev,
      selectedMolecules: prev.selectedMolecules.includes(molecule)
        ? prev.selectedMolecules.filter(m => m !== molecule)
        : [...prev.selectedMolecules, molecule]
    }));
  }, []);
  
  // Build Plotly traces for main comparison plot
  const comparisonPlotData = useMemo((): Data[] => {
    if (!analysis) return [];
    
    const traces: Data[] = [];
    const { rawSpectrum, processedSpectrum, molecularBands, detectedLines } = analysis;
    const { showRaw, showProcessed, showContinuum, showUncertainties, showMolecularBands, showAbsorptionLines, selectedMolecules } = dashboardState;
    
    // Molecular band shapes (add first so they're behind traces)
    if (showMolecularBands) {
      const shapes: Partial<Plotly.Shape>[] = [];
      for (const band of molecularBands) {
        if (selectedMolecules.includes(band.molecule)) {
          shapes.push({
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: band.startWavelength,
            x1: band.endWavelength,
            y0: 0,
            y1: 1,
            fillcolor: band.color,
            opacity: band.opacity,
            line: { width: 0 }
          });
        }
      }
    }
    
    // Raw spectrum with uncertainty band
    if (showRaw) {
      if (showUncertainties && rawSpectrum.uncertainties) {
        // Upper bound
        traces.push({
          x: rawSpectrum.wavelengths,
          y: rawSpectrum.flux.map((f, i) => f + (rawSpectrum.uncertainties?.[i] || 0)),
          type: 'scatter',
          mode: 'lines',
          fill: 'none',
          line: { color: 'transparent' },
          showlegend: false,
          hoverinfo: 'skip'
        });
        
        // Lower bound with fill
        traces.push({
          x: rawSpectrum.wavelengths,
          y: rawSpectrum.flux.map((f, i) => f - (rawSpectrum.uncertainties?.[i] || 0)),
          type: 'scatter',
          mode: 'lines',
          fill: 'tonexty',
          fillcolor: 'rgba(136, 136, 136, 0.2)',
          line: { color: 'transparent' },
          showlegend: false,
          hoverinfo: 'skip'
        });
      }
      
      traces.push({
        x: rawSpectrum.wavelengths,
        y: rawSpectrum.flux,
        type: 'scatter',
        mode: 'lines',
        name: 'Raw Spectrum',
        line: { color: '#888888', width: 1 },
        hovertemplate: 'λ: %{x:.3f} μm<br>Flux: %{y:.4f}<extra>Raw</extra>'
      });
    }
    
    // Processed spectrum
    if (showProcessed) {
      traces.push({
        x: processedSpectrum.wavelengths,
        y: processedSpectrum.flux,
        type: 'scatter',
        mode: 'lines',
        name: 'Processed Spectrum',
        line: { color: '#ffd700', width: 2 },
        hovertemplate: 'λ: %{x:.3f} μm<br>Flux: %{y:.4f}<extra>Processed</extra>'
      });
    }
    
    // Continuum
    if (showContinuum && processedSpectrum.continuum) {
      traces.push({
        x: processedSpectrum.wavelengths,
        y: processedSpectrum.continuum,
        type: 'scatter',
        mode: 'lines',
        name: 'Continuum',
        line: { color: '#00ff88', width: 1.5, dash: 'dash' },
        hovertemplate: 'λ: %{x:.3f} μm<br>Continuum: %{y:.4f}<extra></extra>'
      });
    }
    
    // Absorption line markers
    if (showAbsorptionLines) {
      const visibleLines = detectedLines.filter(
        line => line.detected && selectedMolecules.includes(line.molecule)
      );
      
      if (visibleLines.length > 0) {
        traces.push({
          x: visibleLines.map(l => l.wavelength),
          y: visibleLines.map(l => {
            // Find flux value at this wavelength
            const idx = processedSpectrum.wavelengths.findIndex(w => w >= l.wavelength);
            return processedSpectrum.flux[idx] || 0;
          }),
          type: 'scatter',
          mode: 'markers+text',
          name: 'Absorption Lines',
          marker: {
            size: 10,
            color: visibleLines.map(l => MOLECULAR_COLORS[l.molecule] || '#ffffff'),
            symbol: 'triangle-down',
            line: { color: '#ffffff', width: 1 }
          },
          text: visibleLines.map(l => l.molecule),
          textposition: 'top center',
          textfont: { size: 10, color: '#ffffff' },
          hovertemplate: '%{text}<br>λ: %{x:.3f} μm<extra></extra>'
        });
      }
    }
    
    return traces;
  }, [analysis, dashboardState]);
  
  // Build Plotly layout
  const comparisonLayout = useMemo((): Partial<Layout> => {
    const shapes: Partial<Plotly.Shape>[] = [];
    
    // Add molecular band shapes
    if (analysis && dashboardState.showMolecularBands) {
      for (const band of analysis.molecularBands) {
        if (dashboardState.selectedMolecules.includes(band.molecule)) {
          shapes.push({
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: band.startWavelength,
            x1: band.endWavelength,
            y0: 0,
            y1: 1,
            fillcolor: band.color,
            opacity: band.opacity,
            line: { width: 0 }
          });
        }
      }
    }
    
    // Add absorption line annotations
    const annotations: Partial<Plotly.Annotations>[] = [];
    if (analysis && dashboardState.showAbsorptionLines) {
      for (const line of analysis.detectedLines) {
        if (line.detected && dashboardState.selectedMolecules.includes(line.molecule)) {
          annotations.push({
            x: line.wavelength,
            y: 1,
            xref: 'x',
            yref: 'paper',
            text: `${line.molecule}`,
            showarrow: true,
            arrowhead: 2,
            arrowsize: 0.8,
            arrowwidth: 1,
            arrowcolor: MOLECULAR_COLORS[line.molecule] || '#fff',
            ax: 0,
            ay: -25,
            font: { size: 9, color: MOLECULAR_COLORS[line.molecule] || '#fff' },
            bgcolor: 'rgba(0,0,0,0.7)',
            borderpad: 3
          });
        }
      }
    }
    
    return {
      title: {
        text: 'Spectrum Analysis',
        font: { color: '#ffffff', size: 18 }
      },
      paper_bgcolor: '#0a0a1a',
      plot_bgcolor: '#0d0d1f',
      font: { color: '#888888' },
      xaxis: {
        title: { text: 'Wavelength (μm)', font: { color: '#aaa' } },
        color: '#666',
        gridcolor: 'rgba(255,255,255,0.05)',
        zerolinecolor: 'rgba(255,255,255,0.1)',
        range: dashboardState.wavelengthRange,
        rangeslider: { visible: true, bgcolor: '#1a1a2e', bordercolor: '#333' }
      },
      yaxis: {
        title: { text: 'Flux (normalized)', font: { color: '#aaa' } },
        color: '#666',
        gridcolor: 'rgba(255,255,255,0.05)',
        zerolinecolor: 'rgba(255,255,255,0.1)'
      },
      legend: {
        x: 1,
        y: 1,
        xanchor: 'right',
        bgcolor: 'rgba(10,10,26,0.8)',
        bordercolor: 'rgba(255,255,255,0.1)',
        font: { color: '#ccc' }
      },
      margin: { t: 60, r: 40, b: 80, l: 70 },
      hovermode: 'x unified',
      shapes,
      annotations
    };
  }, [analysis, dashboardState]);
  
  // Normalized spectrum plot data
  const normalizedPlotData = useMemo((): Data[] => {
    if (!analysis?.processedSpectrum.normalized) return [];
    
    const traces: Data[] = [];
    const { processedSpectrum, molecularBands, detectedLines } = analysis;
    const { selectedMolecules, showAbsorptionLines } = dashboardState;
    
    // Normalized spectrum
    traces.push({
      x: processedSpectrum.wavelengths,
      y: processedSpectrum.normalized,
      type: 'scatter',
      mode: 'lines',
      name: 'Normalized',
      line: { color: '#00d4ff', width: 2 },
      hovertemplate: 'λ: %{x:.3f} μm<br>Normalized: %{y:.4f}<extra></extra>'
    });
    
    // Reference line at 1.0
    traces.push({
      x: [processedSpectrum.wavelengths[0], processedSpectrum.wavelengths[processedSpectrum.wavelengths.length - 1]],
      y: [1, 1],
      type: 'scatter',
      mode: 'lines',
      name: 'Continuum Level',
      line: { color: '#00ff88', width: 1, dash: 'dash' },
      hoverinfo: 'skip'
    });
    
    return traces;
  }, [analysis, dashboardState]);
  
  // Absorption lines table plot
  const linesPlotData = useMemo((): Data[] => {
    if (!analysis) return [];
    
    const visibleLines = analysis.detectedLines.filter(
      line => dashboardState.selectedMolecules.includes(line.molecule)
    );
    
    return [{
      type: 'bar',
      x: visibleLines.map(l => `${l.molecule} ${l.wavelength}μm`),
      y: visibleLines.map(l => (l.confidence || 0) * 100),
      marker: {
        color: visibleLines.map(l => l.detected ? MOLECULAR_COLORS[l.molecule] || '#ffd700' : '#444'),
        line: { color: '#ffffff', width: 1 }
      },
      text: visibleLines.map(l => l.detected ? `${((l.confidence || 0) * 100).toFixed(0)}%` : 'Not detected'),
      textposition: 'outside',
      textfont: { color: '#ccc', size: 10 },
      hovertemplate: '%{x}<br>Confidence: %{y:.1f}%<extra></extra>'
    }];
  }, [analysis, dashboardState]);
  
  // Handle plot click
  const handlePlotClick = useCallback((event: PlotMouseEvent) => {
    if (event.points && event.points[0]) {
      const point = event.points[0];
      setSelectedPoint({
        wl: point.x as number,
        flux: point.y as number
      });
    }
  }, []);
  
  // Render loading state
  if (isLoading) {
    return (
      <div className="dashboard-page">
        <div className="loading-container">
          <div className="loading-spinner">
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
          </div>
          <p>Loading spectrum analysis...</p>
        </div>
      </div>
    );
  }
  
  if (!analysis) {
    return (
      <div className="dashboard-page">
        <div className="error-container">
          <h2>No Analysis Data</h2>
          <p>Upload a spectrum to begin analysis.</p>
          <a href="/upload" className="primary-btn">Upload Spectrum</a>
        </div>
      </div>
    );
  }
  
  return (
    <div className="dashboard-page">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-left">
          <a href="/" className="back-link">← Mission Control</a>
          <h1>Spectrum Dashboard</h1>
          <span className="analysis-id">ID: {analysis.id}</span>
        </div>
        <div className="header-right">
          <span className="timestamp">
            Analyzed: {new Date(analysis.timestamp).toLocaleString()}
          </span>
        </div>
      </header>
      
      <main className="dashboard-main">
        {/* Left Sidebar - Controls */}
        <aside className="control-sidebar">
          <section className="control-section">
            <h3>📊 Display Options</h3>
            
            <label className="toggle-control">
              <input
                type="checkbox"
                checked={dashboardState.showRaw}
                onChange={() => toggleState('showRaw')}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Raw Spectrum</span>
            </label>
            
            <label className="toggle-control">
              <input
                type="checkbox"
                checked={dashboardState.showProcessed}
                onChange={() => toggleState('showProcessed')}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Processed Spectrum</span>
            </label>
            
            <label className="toggle-control">
              <input
                type="checkbox"
                checked={dashboardState.showContinuum}
                onChange={() => toggleState('showContinuum')}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Continuum Fit</span>
            </label>
            
            <label className="toggle-control">
              <input
                type="checkbox"
                checked={dashboardState.showUncertainties}
                onChange={() => toggleState('showUncertainties')}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Uncertainties</span>
            </label>
          </section>
          
          <section className="control-section">
            <h3>🔬 Overlays</h3>
            
            <label className="toggle-control">
              <input
                type="checkbox"
                checked={dashboardState.showAbsorptionLines}
                onChange={() => toggleState('showAbsorptionLines')}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Absorption Lines</span>
            </label>
            
            <label className="toggle-control">
              <input
                type="checkbox"
                checked={dashboardState.showMolecularBands}
                onChange={() => toggleState('showMolecularBands')}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Molecular Bands</span>
            </label>
          </section>
          
          <section className="control-section">
            <h3>🧪 Molecules</h3>
            <div className="molecule-toggles">
              {Object.keys(MOLECULAR_COLORS).slice(0, 8).map(mol => (
                <button
                  key={mol}
                  className={`molecule-toggle ${dashboardState.selectedMolecules.includes(mol) ? 'active' : ''}`}
                  style={{
                    borderColor: dashboardState.selectedMolecules.includes(mol) 
                      ? MOLECULAR_COLORS[mol] 
                      : 'transparent',
                    backgroundColor: dashboardState.selectedMolecules.includes(mol)
                      ? `${MOLECULAR_COLORS[mol]}20`
                      : 'transparent'
                  }}
                  onClick={() => toggleMolecule(mol)}
                >
                  <span 
                    className="molecule-dot"
                    style={{ backgroundColor: MOLECULAR_COLORS[mol] }}
                  />
                  {mol}
                </button>
              ))}
            </div>
          </section>
          
          <section className="control-section">
            <h3>📏 Wavelength Range</h3>
            <div className="range-inputs">
              <input
                type="number"
                value={dashboardState.wavelengthRange[0]}
                onChange={(e) => setDashboardState(prev => ({
                  ...prev,
                  wavelengthRange: [parseFloat(e.target.value), prev.wavelengthRange[1]]
                }))}
                min={0.1}
                max={dashboardState.wavelengthRange[1] - 0.1}
                step={0.1}
              />
              <span>to</span>
              <input
                type="number"
                value={dashboardState.wavelengthRange[1]}
                onChange={(e) => setDashboardState(prev => ({
                  ...prev,
                  wavelengthRange: [prev.wavelengthRange[0], parseFloat(e.target.value)]
                }))}
                min={dashboardState.wavelengthRange[0] + 0.1}
                max={30}
                step={0.1}
              />
              <span>μm</span>
            </div>
            <div className="range-presets">
              <button onClick={() => setDashboardState(prev => ({ ...prev, wavelengthRange: [0.6, 5] }))}>
                NIR
              </button>
              <button onClick={() => setDashboardState(prev => ({ ...prev, wavelengthRange: [5, 15] }))}>
                MIR
              </button>
              <button onClick={() => setDashboardState(prev => ({ ...prev, wavelengthRange: [0.6, 28] }))}>
                Full
              </button>
            </div>
          </section>
        </aside>
        
        {/* Main Content */}
        <div className="dashboard-content">
          {/* Tab Navigation */}
          <div className="tab-nav">
            <button
              className={`tab-btn ${activeTab === 'comparison' ? 'active' : ''}`}
              onClick={() => setActiveTab('comparison')}
            >
              📈 Raw vs Processed
            </button>
            <button
              className={`tab-btn ${activeTab === 'normalized' ? 'active' : ''}`}
              onClick={() => setActiveTab('normalized')}
            >
              📊 Normalized
            </button>
            <button
              className={`tab-btn ${activeTab === 'lines' ? 'active' : ''}`}
              onClick={() => setActiveTab('lines')}
            >
              📍 Line Detection
            </button>
          </div>
          
          {/* Plot Container */}
          <div className="plot-wrapper">
            {activeTab === 'comparison' && (
              <Plot
                data={comparisonPlotData}
                layout={comparisonLayout}
                config={{
                  displayModeBar: true,
                  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                  displaylogo: false,
                  responsive: true,
                  scrollZoom: true
                }}
                style={{ width: '100%', height: dashboardState.plotHeight }}
                onClick={handlePlotClick}
              />
            )}
            
            {activeTab === 'normalized' && (
              <Plot
                data={normalizedPlotData}
                layout={{
                  ...comparisonLayout,
                  title: { text: 'Normalized Spectrum', font: { color: '#ffffff', size: 18 } },
                  yaxis: {
                    ...comparisonLayout.yaxis,
                    title: { text: 'Normalized Flux', font: { color: '#aaa' } }
                  }
                }}
                config={{
                  displayModeBar: true,
                  displaylogo: false,
                  responsive: true,
                  scrollZoom: true
                }}
                style={{ width: '100%', height: dashboardState.plotHeight }}
              />
            )}
            
            {activeTab === 'lines' && (
              <Plot
                data={linesPlotData}
                layout={{
                  ...comparisonLayout,
                  title: { text: 'Absorption Line Detection Confidence', font: { color: '#ffffff', size: 18 } },
                  xaxis: {
                    ...comparisonLayout.xaxis,
                    title: { text: 'Absorption Line', font: { color: '#aaa' } },
                    tickangle: -45,
                    rangeslider: { visible: false }
                  },
                  yaxis: {
                    ...comparisonLayout.yaxis,
                    title: { text: 'Detection Confidence (%)', font: { color: '#aaa' } },
                    range: [0, 110]
                  }
                }}
                config={{
                  displayModeBar: true,
                  displaylogo: false,
                  responsive: true
                }}
                style={{ width: '100%', height: dashboardState.plotHeight }}
              />
            )}
          </div>
          
          {/* Info Bar */}
          {selectedPoint && (
            <div className="info-bar">
              <div className="info-item">
                <span className="info-label">Wavelength</span>
                <span className="info-value">{selectedPoint.wl.toFixed(4)} μm</span>
              </div>
              <div className="info-item">
                <span className="info-label">Flux</span>
                <span className="info-value">{selectedPoint.flux.toExponential(4)}</span>
              </div>
              <button className="close-btn" onClick={() => setSelectedPoint(null)}>✕</button>
            </div>
          )}
        </div>
        
        {/* Right Sidebar - Results */}
        <aside className="results-sidebar">
          <section className="results-section">
            <h3>🎯 Detection Results</h3>
            <div className="molecule-list">
              {analysis.molecules.map(mol => (
                <div key={mol.name} className="molecule-result">
                  <div className="molecule-header">
                    <span 
                      className="molecule-badge"
                      style={{ backgroundColor: MOLECULAR_COLORS[mol.name] || '#888' }}
                    >
                      {mol.name}
                    </span>
                    <span className="confidence">{(mol.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill"
                      style={{ 
                        width: `${mol.confidence * 100}%`,
                        backgroundColor: MOLECULAR_COLORS[mol.name] || '#ffd700'
                      }}
                    />
                  </div>
                  {mol.abundance && (
                    <span className="abundance">
                      Abundance: {mol.abundance.toExponential(1)}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </section>
          
          <section className="results-section">
            <h3>🌍 Planet Classification</h3>
            <div className="classification-list">
              {analysis.planetClass.map((cls, idx) => (
                <div key={cls.type} className={`class-item ${idx === 0 ? 'primary' : ''}`}>
                  <span className="class-type">{cls.type}</span>
                  <span className="class-prob">{(cls.probability * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </section>
          
          <section className="results-section">
            <h3>💚 Habitability Index</h3>
            <div className="habitability-gauge">
              <div className="gauge-track">
                <div 
                  className="gauge-fill"
                  style={{ 
                    width: `${analysis.habitability * 100}%`,
                    background: `linear-gradient(90deg, 
                      #ff4444 0%, 
                      #ffaa00 50%, 
                      #00ff88 100%)`
                  }}
                />
              </div>
              <div className="gauge-labels">
                <span>Hostile</span>
                <span>Marginal</span>
                <span>Habitable</span>
              </div>
              <div className="gauge-value">
                {(analysis.habitability * 100).toFixed(0)}%
              </div>
            </div>
          </section>
          
          <section className="results-section">
            <h3>📋 Spectrum Stats</h3>
            <div className="stats-grid">
              <div className="stat">
                <span className="stat-label">Data Points</span>
                <span className="stat-value">{analysis.rawSpectrum.wavelengths.length}</span>
              </div>
              <div className="stat">
                <span className="stat-label">λ Range</span>
                <span className="stat-value">
                  {analysis.rawSpectrum.wavelengths[0].toFixed(1)} - {analysis.rawSpectrum.wavelengths[analysis.rawSpectrum.wavelengths.length - 1].toFixed(1)} μm
                </span>
              </div>
              <div className="stat">
                <span className="stat-label">Lines Detected</span>
                <span className="stat-value">
                  {analysis.detectedLines.filter(l => l.detected).length} / {analysis.detectedLines.length}
                </span>
              </div>
              <div className="stat">
                <span className="stat-label">Molecules</span>
                <span className="stat-value">{analysis.molecules.length}</span>
              </div>
            </div>
          </section>
          
          <div className="action-buttons">
            <button className="action-btn primary">
              📥 Export Report
            </button>
            <button className="action-btn">
              🔬 Run Explainability
            </button>
          </div>
        </aside>
      </main>
    </div>
  );
};

export default SpectrumDashboard;
