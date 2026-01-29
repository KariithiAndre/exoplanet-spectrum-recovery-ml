/**
 * Spectrum Upload Page
 * 
 * Features:
 * - CSV file upload with drag-and-drop
 * - Live preview plots using Canvas
 * - Simulation mode for synthetic spectra
 * - Integration with backend API
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import './SpectrumUploadPage.css';

// ===== Type Definitions =====
interface SpectrumData {
  id: string;
  name: string;
  wavelengths: number[];
  fluxValues: number[];
  uncertainties?: number[];
  metadata: SpectrumMetadata;
}

interface SpectrumMetadata {
  source: 'upload' | 'simulation';
  fileName?: string;
  uploadDate: string;
  wavelengthUnit: string;
  fluxUnit: string;
  numPoints: number;
  wavelengthRange: [number, number];
  simulationParams?: SimulationParams;
}

interface SimulationParams {
  planetType: 'terrestrial' | 'super-earth' | 'mini-neptune' | 'gas-giant' | 'hot-jupiter';
  starType: 'M' | 'K' | 'G' | 'F' | 'A';
  temperature: number;
  molecules: string[];
  noiseLevel: number;
  resolution: number;
}

interface PlotConfig {
  showGrid: boolean;
  showUncertainties: boolean;
  logScale: boolean;
  smoothing: number;
  colorScheme: 'default' | 'thermal' | 'cool' | 'monochrome';
}

// ===== Utility Functions =====
const generateId = (): string => {
  return `spectrum_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

const parseCSV = (content: string): { wavelengths: number[]; fluxValues: number[]; uncertainties?: number[] } => {
  const lines = content.trim().split('\n');
  const wavelengths: number[] = [];
  const fluxValues: number[] = [];
  const uncertainties: number[] = [];
  
  let hasHeader = false;
  let hasUncertainties = false;
  
  // Check first line for header
  const firstLine = lines[0].split(/[,\t;]/);
  if (isNaN(parseFloat(firstLine[0]))) {
    hasHeader = true;
    hasUncertainties = firstLine.length >= 3;
  } else {
    hasUncertainties = firstLine.length >= 3;
  }
  
  const startIdx = hasHeader ? 1 : 0;
  
  for (let i = startIdx; i < lines.length; i++) {
    const parts = lines[i].split(/[,\t;]/).map(p => p.trim());
    if (parts.length >= 2) {
      const wl = parseFloat(parts[0]);
      const flux = parseFloat(parts[1]);
      
      if (!isNaN(wl) && !isNaN(flux)) {
        wavelengths.push(wl);
        fluxValues.push(flux);
        
        if (hasUncertainties && parts.length >= 3) {
          const unc = parseFloat(parts[2]);
          uncertainties.push(isNaN(unc) ? 0 : unc);
        }
      }
    }
  }
  
  return {
    wavelengths,
    fluxValues,
    uncertainties: uncertainties.length > 0 ? uncertainties : undefined
  };
};

const generateSyntheticSpectrum = (params: SimulationParams): { wavelengths: number[]; fluxValues: number[]; uncertainties: number[] } => {
  const { temperature, molecules, noiseLevel, resolution } = params;
  
  // Generate wavelength grid (0.6 to 28 microns for JWST range)
  const numPoints = resolution;
  const wavelengths: number[] = [];
  const fluxValues: number[] = [];
  const uncertainties: number[] = [];
  
  const wlMin = 0.6;
  const wlMax = 28.0;
  
  for (let i = 0; i < numPoints; i++) {
    const wl = wlMin + (wlMax - wlMin) * (i / (numPoints - 1));
    wavelengths.push(wl);
  }
  
  // Blackbody continuum
  const h = 6.626e-34;
  const c = 3e8;
  const k = 1.381e-23;
  
  // Molecular absorption features
  const molecularBands: Record<string, { center: number; width: number; depth: number }[]> = {
    'H2O': [
      { center: 1.4, width: 0.2, depth: 0.3 },
      { center: 1.9, width: 0.3, depth: 0.4 },
      { center: 2.7, width: 0.4, depth: 0.5 },
      { center: 6.3, width: 0.8, depth: 0.6 }
    ],
    'CO2': [
      { center: 4.3, width: 0.3, depth: 0.7 },
      { center: 15.0, width: 2.0, depth: 0.5 }
    ],
    'CH4': [
      { center: 2.3, width: 0.2, depth: 0.4 },
      { center: 3.3, width: 0.3, depth: 0.5 },
      { center: 7.7, width: 0.5, depth: 0.4 }
    ],
    'O3': [
      { center: 9.6, width: 0.8, depth: 0.3 }
    ],
    'NH3': [
      { center: 10.5, width: 1.0, depth: 0.35 }
    ],
    'CO': [
      { center: 4.7, width: 0.2, depth: 0.25 }
    ]
  };
  
  for (let i = 0; i < numPoints; i++) {
    const wl = wavelengths[i];
    const wlMeters = wl * 1e-6;
    
    // Planck function (simplified, normalized)
    const exponent = (h * c) / (wlMeters * k * temperature);
    let flux = Math.pow(wl, -5) / (Math.exp(Math.min(exponent, 700)) - 1);
    
    // Normalize
    flux = flux / Math.pow(wlMin, -5);
    
    // Apply molecular absorption
    for (const mol of molecules) {
      const bands = molecularBands[mol] || [];
      for (const band of bands) {
        const distance = Math.abs(wl - band.center) / band.width;
        if (distance < 3) {
          const absorption = band.depth * Math.exp(-0.5 * distance * distance);
          flux *= (1 - absorption);
        }
      }
    }
    
    // Add noise
    const noise = (Math.random() - 0.5) * 2 * noiseLevel * flux;
    flux += noise;
    flux = Math.max(flux, 0.01);
    
    fluxValues.push(flux);
    uncertainties.push(noiseLevel * flux);
  }
  
  // Normalize flux values
  const maxFlux = Math.max(...fluxValues);
  for (let i = 0; i < numPoints; i++) {
    fluxValues[i] /= maxFlux;
    uncertainties[i] /= maxFlux;
  }
  
  return { wavelengths, fluxValues, uncertainties };
};

// ===== Spectrum Plot Component =====
interface SpectrumPlotProps {
  spectrum: SpectrumData | null;
  config: PlotConfig;
  width: number;
  height: number;
}

const SpectrumPlot: React.FC<SpectrumPlotProps> = ({ spectrum, config, width, height }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const colorSchemes: Record<string, { line: string; fill: string; grid: string; axis: string }> = {
    default: { line: '#ffd700', fill: 'rgba(255, 215, 0, 0.1)', grid: 'rgba(255, 255, 255, 0.1)', axis: '#888' },
    thermal: { line: '#ff6b35', fill: 'rgba(255, 107, 53, 0.1)', grid: 'rgba(255, 107, 53, 0.1)', axis: '#ff8c5a' },
    cool: { line: '#00d4ff', fill: 'rgba(0, 212, 255, 0.1)', grid: 'rgba(0, 212, 255, 0.1)', axis: '#5ce1ff' },
    monochrome: { line: '#ffffff', fill: 'rgba(255, 255, 255, 0.05)', grid: 'rgba(255, 255, 255, 0.08)', axis: '#666' }
  };
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas resolution
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    
    // Clear canvas
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, width, height);
    
    const colors = colorSchemes[config.colorScheme];
    const padding = { top: 40, right: 40, bottom: 60, left: 80 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    
    // Draw empty state
    if (!spectrum) {
      ctx.fillStyle = '#444';
      ctx.font = '16px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Upload a spectrum or run simulation to preview', width / 2, height / 2);
      return;
    }
    
    const { wavelengths, fluxValues, uncertainties } = spectrum;
    
    // Apply smoothing if needed
    let smoothedFlux = [...fluxValues];
    if (config.smoothing > 1) {
      const windowSize = Math.min(config.smoothing, Math.floor(fluxValues.length / 10));
      smoothedFlux = fluxValues.map((_, i) => {
        let sum = 0;
        let count = 0;
        for (let j = Math.max(0, i - windowSize); j <= Math.min(fluxValues.length - 1, i + windowSize); j++) {
          sum += fluxValues[j];
          count++;
        }
        return sum / count;
      });
    }
    
    // Calculate data range
    const wlMin = Math.min(...wavelengths);
    const wlMax = Math.max(...wavelengths);
    let fluxMin = Math.min(...smoothedFlux);
    let fluxMax = Math.max(...smoothedFlux);
    
    if (config.logScale) {
      fluxMin = Math.max(fluxMin, 1e-6);
      fluxMax = Math.max(fluxMax, fluxMin * 10);
    }
    
    // Add padding to flux range
    const fluxPadding = (fluxMax - fluxMin) * 0.1;
    fluxMin -= fluxPadding;
    fluxMax += fluxPadding;
    if (fluxMin < 0 && !config.logScale) fluxMin = 0;
    
    // Coordinate transformation functions
    const toCanvasX = (wl: number) => padding.left + ((wl - wlMin) / (wlMax - wlMin)) * plotWidth;
    const toCanvasY = (flux: number) => {
      if (config.logScale) {
        const logMin = Math.log10(Math.max(fluxMin, 1e-10));
        const logMax = Math.log10(fluxMax);
        const logFlux = Math.log10(Math.max(flux, 1e-10));
        return padding.top + (1 - (logFlux - logMin) / (logMax - logMin)) * plotHeight;
      }
      return padding.top + (1 - (flux - fluxMin) / (fluxMax - fluxMin)) * plotHeight;
    };
    
    // Draw grid
    if (config.showGrid) {
      ctx.strokeStyle = colors.grid;
      ctx.lineWidth = 1;
      
      // Vertical grid lines
      const numXGridLines = 10;
      for (let i = 0; i <= numXGridLines; i++) {
        const x = padding.left + (i / numXGridLines) * plotWidth;
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + plotHeight);
        ctx.stroke();
      }
      
      // Horizontal grid lines
      const numYGridLines = 8;
      for (let i = 0; i <= numYGridLines; i++) {
        const y = padding.top + (i / numYGridLines) * plotHeight;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + plotWidth, y);
        ctx.stroke();
      }
    }
    
    // Draw uncertainties if available
    if (config.showUncertainties && uncertainties) {
      ctx.fillStyle = colors.fill;
      ctx.beginPath();
      
      // Upper bound
      for (let i = 0; i < wavelengths.length; i++) {
        const x = toCanvasX(wavelengths[i]);
        const y = toCanvasY(smoothedFlux[i] + uncertainties[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      
      // Lower bound (reverse)
      for (let i = wavelengths.length - 1; i >= 0; i--) {
        const x = toCanvasX(wavelengths[i]);
        const y = toCanvasY(smoothedFlux[i] - uncertainties[i]);
        ctx.lineTo(x, y);
      }
      
      ctx.closePath();
      ctx.fill();
    }
    
    // Draw spectrum line
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i < wavelengths.length; i++) {
      const x = toCanvasX(wavelengths[i]);
      const y = toCanvasY(smoothedFlux[i]);
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // Draw axes
    ctx.strokeStyle = colors.axis;
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + plotHeight);
    ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotHeight);
    ctx.stroke();
    
    // Axis labels
    ctx.fillStyle = '#888';
    ctx.font = '12px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    
    // X-axis labels
    const numXLabels = 6;
    for (let i = 0; i <= numXLabels; i++) {
      const wl = wlMin + (i / numXLabels) * (wlMax - wlMin);
      const x = toCanvasX(wl);
      ctx.fillText(wl.toFixed(1), x, padding.top + plotHeight + 25);
    }
    
    // X-axis title
    ctx.font = '14px Inter, sans-serif';
    ctx.fillText(`Wavelength (${spectrum.metadata.wavelengthUnit})`, width / 2, height - 10);
    
    // Y-axis labels
    ctx.textAlign = 'right';
    ctx.font = '12px JetBrains Mono, monospace';
    const numYLabels = 5;
    for (let i = 0; i <= numYLabels; i++) {
      let flux: number;
      if (config.logScale) {
        const logMin = Math.log10(Math.max(fluxMin, 1e-10));
        const logMax = Math.log10(fluxMax);
        flux = Math.pow(10, logMin + (1 - i / numYLabels) * (logMax - logMin));
      } else {
        flux = fluxMax - (i / numYLabels) * (fluxMax - fluxMin);
      }
      const y = padding.top + (i / numYLabels) * plotHeight;
      ctx.fillText(flux.toExponential(1), padding.left - 10, y + 4);
    }
    
    // Y-axis title
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.font = '14px Inter, sans-serif';
    ctx.fillText(`Flux (${spectrum.metadata.fluxUnit})`, 0, 0);
    ctx.restore();
    
    // Title
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(spectrum.name, width / 2, 25);
    
  }, [spectrum, config, width, height]);
  
  return (
    <canvas
      ref={canvasRef}
      className="spectrum-canvas"
      style={{ width, height }}
    />
  );
};

// ===== Main Component =====
const SpectrumUploadPage: React.FC = () => {
  // State
  const [mode, setMode] = useState<'upload' | 'simulation'>('upload');
  const [spectrum, setSpectrum] = useState<SpectrumData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  
  // Plot configuration
  const [plotConfig, setPlotConfig] = useState<PlotConfig>({
    showGrid: true,
    showUncertainties: true,
    logScale: false,
    smoothing: 1,
    colorScheme: 'default'
  });
  
  // Simulation parameters
  const [simParams, setSimParams] = useState<SimulationParams>({
    planetType: 'terrestrial',
    starType: 'G',
    temperature: 288,
    molecules: ['H2O', 'CO2'],
    noiseLevel: 0.02,
    resolution: 500
  });
  
  // Upload history
  const [uploadHistory, setUploadHistory] = useState<SpectrumData[]>([]);
  
  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const plotContainerRef = useRef<HTMLDivElement>(null);
  const [plotDimensions, setPlotDimensions] = useState({ width: 800, height: 400 });
  
  // Update plot dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (plotContainerRef.current) {
        const rect = plotContainerRef.current.getBoundingClientRect();
        setPlotDimensions({
          width: Math.max(rect.width - 40, 400),
          height: Math.max(rect.height - 20, 300)
        });
      }
    };
    
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);
  
  // Handle file upload
  const handleFile = useCallback(async (file: File) => {
    if (!file.name.endsWith('.csv')) {
      setError('Please upload a CSV file');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const content = await file.text();
      const parsed = parseCSV(content);
      
      if (parsed.wavelengths.length === 0) {
        throw new Error('No valid data found in CSV');
      }
      
      if (parsed.wavelengths.length < 10) {
        throw new Error('Spectrum must have at least 10 data points');
      }
      
      const newSpectrum: SpectrumData = {
        id: generateId(),
        name: file.name.replace('.csv', ''),
        wavelengths: parsed.wavelengths,
        fluxValues: parsed.fluxValues,
        uncertainties: parsed.uncertainties,
        metadata: {
          source: 'upload',
          fileName: file.name,
          uploadDate: new Date().toISOString(),
          wavelengthUnit: 'μm',
          fluxUnit: 'normalized',
          numPoints: parsed.wavelengths.length,
          wavelengthRange: [Math.min(...parsed.wavelengths), Math.max(...parsed.wavelengths)]
        }
      };
      
      setSpectrum(newSpectrum);
      setUploadHistory(prev => [newSpectrum, ...prev.slice(0, 9)]);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse CSV file');
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  // Drag and drop handlers
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [handleFile]);
  
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  }, [handleFile]);
  
  // Run simulation
  const runSimulation = useCallback(() => {
    setIsLoading(true);
    setError(null);
    
    setTimeout(() => {
      try {
        const result = generateSyntheticSpectrum(simParams);
        
        const planetNames: Record<string, string> = {
          'terrestrial': 'Earth-like',
          'super-earth': 'Super-Earth',
          'mini-neptune': 'Mini-Neptune',
          'gas-giant': 'Gas Giant',
          'hot-jupiter': 'Hot Jupiter'
        };
        
        const newSpectrum: SpectrumData = {
          id: generateId(),
          name: `Simulated ${planetNames[simParams.planetType]} (${simParams.molecules.join(', ')})`,
          wavelengths: result.wavelengths,
          fluxValues: result.fluxValues,
          uncertainties: result.uncertainties,
          metadata: {
            source: 'simulation',
            uploadDate: new Date().toISOString(),
            wavelengthUnit: 'μm',
            fluxUnit: 'normalized',
            numPoints: result.wavelengths.length,
            wavelengthRange: [Math.min(...result.wavelengths), Math.max(...result.wavelengths)],
            simulationParams: { ...simParams }
          }
        };
        
        setSpectrum(newSpectrum);
        setUploadHistory(prev => [newSpectrum, ...prev.slice(0, 9)]);
        
      } catch (err) {
        setError('Simulation failed');
      } finally {
        setIsLoading(false);
      }
    }, 500);
  }, [simParams]);
  
  // Toggle molecule selection
  const toggleMolecule = (mol: string) => {
    setSimParams(prev => ({
      ...prev,
      molecules: prev.molecules.includes(mol)
        ? prev.molecules.filter(m => m !== mol)
        : [...prev.molecules, mol]
    }));
  };
  
  // Load from history
  const loadFromHistory = (spec: SpectrumData) => {
    setSpectrum(spec);
  };
  
  // Clear current spectrum
  const clearSpectrum = () => {
    setSpectrum(null);
    setError(null);
  };
  
  // Export current spectrum
  const exportSpectrum = () => {
    if (!spectrum) return;
    
    let csv = 'wavelength,flux';
    if (spectrum.uncertainties) csv += ',uncertainty';
    csv += '\n';
    
    for (let i = 0; i < spectrum.wavelengths.length; i++) {
      csv += `${spectrum.wavelengths[i]},${spectrum.fluxValues[i]}`;
      if (spectrum.uncertainties) csv += `,${spectrum.uncertainties[i]}`;
      csv += '\n';
    }
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${spectrum.name}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  // Send to analysis
  const sendToAnalysis = async () => {
    if (!spectrum) return;
    
    // In production, this would call the backend API
    alert(`Spectrum "${spectrum.name}" queued for analysis!`);
  };
  
  return (
    <div className="upload-page">
      {/* Header */}
      <header className="upload-header">
        <div className="header-left">
          <a href="/" className="back-link">← Mission Control</a>
          <h1>Spectrum Analyzer</h1>
        </div>
        <div className="header-right">
          <div className="mode-toggle">
            <button
              className={`mode-btn ${mode === 'upload' ? 'active' : ''}`}
              onClick={() => setMode('upload')}
            >
              📤 Upload
            </button>
            <button
              className={`mode-btn ${mode === 'simulation' ? 'active' : ''}`}
              onClick={() => setMode('simulation')}
            >
              🔬 Simulate
            </button>
          </div>
        </div>
      </header>
      
      <main className="upload-main">
        {/* Left Panel - Input */}
        <section className="input-panel">
          {mode === 'upload' ? (
            <>
              <h2>📁 Upload Spectrum</h2>
              
              <div
                className={`drop-zone ${dragActive ? 'active' : ''} ${isLoading ? 'loading' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileSelect}
                  hidden
                />
                
                {isLoading ? (
                  <div className="loading-spinner">
                    <div className="spinner"></div>
                    <p>Processing...</p>
                  </div>
                ) : (
                  <>
                    <div className="drop-icon">📊</div>
                    <p className="drop-title">Drop CSV file here</p>
                    <p className="drop-subtitle">or click to browse</p>
                    <div className="format-hint">
                      <strong>Format:</strong> wavelength, flux [, uncertainty]
                    </div>
                  </>
                )}
              </div>
              
              {error && (
                <div className="error-message">
                  ⚠️ {error}
                </div>
              )}
              
              <div className="format-info">
                <h3>📋 Supported Formats</h3>
                <ul>
                  <li>CSV with comma, tab, or semicolon delimiters</li>
                  <li>Optional header row</li>
                  <li>Optional uncertainty column</li>
                  <li>Wavelength range: 0.1 - 30 μm</li>
                </ul>
              </div>
            </>
          ) : (
            <>
              <h2>🔬 Simulation Mode</h2>
              
              <div className="sim-controls">
                <div className="control-group">
                  <label>Planet Type</label>
                  <select
                    value={simParams.planetType}
                    onChange={(e) => setSimParams(prev => ({
                      ...prev,
                      planetType: e.target.value as SimulationParams['planetType'],
                      temperature: e.target.value === 'hot-jupiter' ? 1500 :
                                   e.target.value === 'gas-giant' ? 150 :
                                   e.target.value === 'mini-neptune' ? 200 : 288
                    }))}
                  >
                    <option value="terrestrial">Terrestrial (Earth-like)</option>
                    <option value="super-earth">Super-Earth</option>
                    <option value="mini-neptune">Mini-Neptune</option>
                    <option value="gas-giant">Gas Giant</option>
                    <option value="hot-jupiter">Hot Jupiter</option>
                  </select>
                </div>
                
                <div className="control-group">
                  <label>Host Star Type</label>
                  <select
                    value={simParams.starType}
                    onChange={(e) => setSimParams(prev => ({
                      ...prev,
                      starType: e.target.value as SimulationParams['starType']
                    }))}
                  >
                    <option value="M">M-dwarf (Red dwarf)</option>
                    <option value="K">K-type (Orange dwarf)</option>
                    <option value="G">G-type (Sun-like)</option>
                    <option value="F">F-type (Yellow-white)</option>
                    <option value="A">A-type (White)</option>
                  </select>
                </div>
                
                <div className="control-group">
                  <label>Temperature: {simParams.temperature} K</label>
                  <input
                    type="range"
                    min="50"
                    max="3000"
                    value={simParams.temperature}
                    onChange={(e) => setSimParams(prev => ({
                      ...prev,
                      temperature: parseInt(e.target.value)
                    }))}
                  />
                </div>
                
                <div className="control-group">
                  <label>Atmospheric Molecules</label>
                  <div className="molecule-grid">
                    {['H2O', 'CO2', 'CH4', 'O3', 'NH3', 'CO'].map(mol => (
                      <button
                        key={mol}
                        className={`molecule-btn ${simParams.molecules.includes(mol) ? 'active' : ''}`}
                        onClick={() => toggleMolecule(mol)}
                      >
                        {mol}
                      </button>
                    ))}
                  </div>
                </div>
                
                <div className="control-group">
                  <label>Noise Level: {(simParams.noiseLevel * 100).toFixed(0)}%</label>
                  <input
                    type="range"
                    min="0"
                    max="20"
                    value={simParams.noiseLevel * 100}
                    onChange={(e) => setSimParams(prev => ({
                      ...prev,
                      noiseLevel: parseInt(e.target.value) / 100
                    }))}
                  />
                </div>
                
                <div className="control-group">
                  <label>Resolution: {simParams.resolution} points</label>
                  <input
                    type="range"
                    min="100"
                    max="2000"
                    step="100"
                    value={simParams.resolution}
                    onChange={(e) => setSimParams(prev => ({
                      ...prev,
                      resolution: parseInt(e.target.value)
                    }))}
                  />
                </div>
                
                <button
                  className="sim-run-btn"
                  onClick={runSimulation}
                  disabled={isLoading || simParams.molecules.length === 0}
                >
                  {isLoading ? '⏳ Generating...' : '🚀 Generate Spectrum'}
                </button>
              </div>
            </>
          )}
          
          {/* History */}
          {uploadHistory.length > 0 && (
            <div className="history-panel">
              <h3>📜 Recent Spectra</h3>
              <div className="history-list">
                {uploadHistory.map(spec => (
                  <button
                    key={spec.id}
                    className={`history-item ${spectrum?.id === spec.id ? 'active' : ''}`}
                    onClick={() => loadFromHistory(spec)}
                  >
                    <span className="history-icon">
                      {spec.metadata.source === 'upload' ? '📤' : '🔬'}
                    </span>
                    <span className="history-name">{spec.name}</span>
                    <span className="history-points">{spec.metadata.numPoints} pts</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>
        
        {/* Center - Plot */}
        <section className="plot-panel" ref={plotContainerRef}>
          <div className="plot-header">
            <h2>📈 Live Preview</h2>
            <div className="plot-actions">
              {spectrum && (
                <>
                  <button className="action-btn" onClick={exportSpectrum} title="Export CSV">
                    💾 Export
                  </button>
                  <button className="action-btn primary" onClick={sendToAnalysis} title="Send to Analysis">
                    🔍 Analyze
                  </button>
                  <button className="action-btn danger" onClick={clearSpectrum} title="Clear">
                    ✕
                  </button>
                </>
              )}
            </div>
          </div>
          
          <div className="plot-container">
            <SpectrumPlot
              spectrum={spectrum}
              config={plotConfig}
              width={plotDimensions.width}
              height={plotDimensions.height}
            />
          </div>
          
          {spectrum && (
            <div className="spectrum-info">
              <div className="info-item">
                <span className="info-label">Source</span>
                <span className="info-value">{spectrum.metadata.source}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Points</span>
                <span className="info-value">{spectrum.metadata.numPoints}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Range</span>
                <span className="info-value">
                  {spectrum.metadata.wavelengthRange[0].toFixed(2)} - {spectrum.metadata.wavelengthRange[1].toFixed(2)} {spectrum.metadata.wavelengthUnit}
                </span>
              </div>
              {spectrum.uncertainties && (
                <div className="info-item">
                  <span className="info-label">Uncertainties</span>
                  <span className="info-value">✓ Included</span>
                </div>
              )}
            </div>
          )}
        </section>
        
        {/* Right Panel - Plot Config */}
        <section className="config-panel">
          <h2>⚙️ Display Options</h2>
          
          <div className="config-group">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={plotConfig.showGrid}
                onChange={(e) => setPlotConfig(prev => ({
                  ...prev,
                  showGrid: e.target.checked
                }))}
              />
              <span>Show Grid</span>
            </label>
          </div>
          
          <div className="config-group">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={plotConfig.showUncertainties}
                onChange={(e) => setPlotConfig(prev => ({
                  ...prev,
                  showUncertainties: e.target.checked
                }))}
              />
              <span>Show Uncertainties</span>
            </label>
          </div>
          
          <div className="config-group">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={plotConfig.logScale}
                onChange={(e) => setPlotConfig(prev => ({
                  ...prev,
                  logScale: e.target.checked
                }))}
              />
              <span>Log Scale (Y-axis)</span>
            </label>
          </div>
          
          <div className="config-group">
            <label>Smoothing: {plotConfig.smoothing}x</label>
            <input
              type="range"
              min="1"
              max="20"
              value={plotConfig.smoothing}
              onChange={(e) => setPlotConfig(prev => ({
                ...prev,
                smoothing: parseInt(e.target.value)
              }))}
            />
          </div>
          
          <div className="config-group">
            <label>Color Scheme</label>
            <div className="color-schemes">
              {(['default', 'thermal', 'cool', 'monochrome'] as const).map(scheme => (
                <button
                  key={scheme}
                  className={`color-btn ${scheme} ${plotConfig.colorScheme === scheme ? 'active' : ''}`}
                  onClick={() => setPlotConfig(prev => ({ ...prev, colorScheme: scheme }))}
                  title={scheme}
                />
              ))}
            </div>
          </div>
          
          <div className="config-group keyboard-hints">
            <h3>⌨️ Shortcuts</h3>
            <div className="shortcut">
              <kbd>G</kbd> Toggle Grid
            </div>
            <div className="shortcut">
              <kbd>U</kbd> Toggle Uncertainties
            </div>
            <div className="shortcut">
              <kbd>L</kbd> Toggle Log Scale
            </div>
            <div className="shortcut">
              <kbd>S</kbd> Export CSV
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default SpectrumUploadPage;
