/**
 * Research Comparison Page
 * 
 * Scientific comparison tool for multiple exoplanet spectra:
 * - Side-by-side spectrum visualization
 * - Synchronized zoom/pan across spectra
 * - Molecule detection comparison matrix
 * - Exportable charts (PNG, SVG, PDF)
 * - Summary statistics tables
 * - Publication-ready formatting
 */

import React, { useState, useCallback, useMemo, useRef } from 'react';
import Plot from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';
import './ResearchComparison.css';

// ===== Type Definitions =====
interface SpectrumEntry {
  id: string;
  name: string;
  target: string;
  wavelengths: number[];
  flux: number[];
  uncertainties?: number[];
  metadata: {
    telescope: string;
    instrument: string;
    observationDate: string;
    exposureTime: number;
    snr: number;
    resolution: number;
  };
  analysis: {
    molecules: MoleculeResult[];
    planetClass: string;
    habitability: number;
    temperature: number;
  };
}

interface MoleculeResult {
  formula: string;
  detected: boolean;
  confidence: number;
  abundance?: number;
}

interface ComparisonState {
  selectedSpectra: string[];
  syncZoom: boolean;
  normalizeFlux: boolean;
  showUncertainties: boolean;
  showMoleculeOverlay: boolean;
  wavelengthRange: [number, number];
  colorScheme: 'distinct' | 'sequential' | 'diverging';
}

interface TableColumn {
  key: string;
  label: string;
  format?: (value: any) => string;
  sortable?: boolean;
}

// ===== Color Palettes =====
const COLOR_PALETTES = {
  distinct: ['#ffd700', '#00d4ff', '#ff6b6b', '#00ff88', '#9b59b6', '#ff8c00', '#1abc9c', '#e91e63'],
  sequential: ['#ffd700', '#ffcc00', '#ffaa00', '#ff8800', '#ff6600', '#ff4400', '#ff2200', '#ff0000'],
  diverging: ['#00d4ff', '#33ddff', '#66e6ff', '#99eeff', '#ffee99', '#ffe666', '#ffdd33', '#ffd700']
};

// ===== Demo Data Generator =====
const generateDemoSpectra = (): SpectrumEntry[] => {
  const targets = [
    { name: 'TRAPPIST-1e', class: 'Terrestrial', temp: 251, hab: 0.82 },
    { name: 'K2-18b', class: 'Sub-Neptune', temp: 284, hab: 0.65 },
    { name: 'LHS 1140b', class: 'Super-Earth', temp: 230, hab: 0.71 },
    { name: 'Proxima Cen b', class: 'Terrestrial', temp: 234, hab: 0.58 },
    { name: 'TOI-700d', class: 'Terrestrial', temp: 268, hab: 0.74 }
  ];
  
  const moleculePool = ['H₂O', 'CO₂', 'CH₄', 'O₃', 'O₂', 'NH₃', 'CO', 'N₂O'];
  
  return targets.map((target, idx) => {
    const numPoints = 800;
    const wavelengths: number[] = [];
    const flux: number[] = [];
    const uncertainties: number[] = [];
    
    for (let i = 0; i < numPoints; i++) {
      const wl = 0.6 + (27.4 * i / (numPoints - 1));
      wavelengths.push(wl);
      
      // Base continuum with unique shape per target
      let f = Math.exp(-0.3 * Math.pow((wl - 5 - idx) / 12, 2));
      
      // Add molecular features
      const features = [
        { center: 1.4, width: 0.15, depth: 0.2 + Math.random() * 0.2 },
        { center: 2.7, width: 0.3, depth: 0.3 + Math.random() * 0.2 },
        { center: 4.3, width: 0.25, depth: 0.4 + Math.random() * 0.3 },
        { center: 6.3, width: 0.5, depth: 0.25 + Math.random() * 0.2 },
        { center: 9.6, width: 0.6, depth: 0.15 + Math.random() * 0.15 },
        { center: 15.0, width: 1.5, depth: 0.2 + Math.random() * 0.2 }
      ];
      
      features.forEach(feat => {
        const dist = Math.abs(wl - feat.center) / feat.width;
        if (dist < 3) {
          f *= (1 - feat.depth * Math.exp(-0.5 * dist * dist));
        }
      });
      
      // Add noise
      const noise = (Math.random() - 0.5) * 0.03 * f;
      flux.push(f + noise);
      uncertainties.push(0.015 * f);
    }
    
    // Normalize
    const maxFlux = Math.max(...flux);
    flux.forEach((_, i) => {
      flux[i] /= maxFlux;
      uncertainties[i] /= maxFlux;
    });
    
    // Generate molecule detections
    const molecules: MoleculeResult[] = moleculePool.map(mol => ({
      formula: mol,
      detected: Math.random() > 0.4,
      confidence: 0.3 + Math.random() * 0.65,
      abundance: Math.random() > 0.5 ? Math.pow(10, -3 - Math.random() * 4) : undefined
    }));
    
    return {
      id: `spectrum_${idx + 1}`,
      name: `${target.name} Spectrum`,
      target: target.name,
      wavelengths,
      flux,
      uncertainties,
      metadata: {
        telescope: ['JWST', 'ELT', 'GMT', 'TMT'][idx % 4],
        instrument: ['NIRSpec', 'MIRI', 'NIRCam', 'METIS'][idx % 4],
        observationDate: `202${5 + Math.floor(idx / 2)}-${String(idx + 3).padStart(2, '0')}-${String(10 + idx * 5).padStart(2, '0')}`,
        exposureTime: 10000 + Math.floor(Math.random() * 50000),
        snr: 50 + Math.floor(Math.random() * 150),
        resolution: 1000 + Math.floor(Math.random() * 2000)
      },
      analysis: {
        molecules,
        planetClass: target.class,
        habitability: target.hab,
        temperature: target.temp
      }
    };
  });
};

// ===== Export Utilities =====
const exportToCSV = (spectra: SpectrumEntry[], filename: string) => {
  let csv = 'Target,Planet Class,Temperature (K),Habitability,';
  csv += 'H₂O,CO₂,CH₄,O₃,O₂,NH₃,CO,N₂O,';
  csv += 'Telescope,Instrument,Observation Date,SNR,Resolution\n';
  
  spectra.forEach(spec => {
    csv += `${spec.target},${spec.analysis.planetClass},${spec.analysis.temperature},${spec.analysis.habitability.toFixed(2)},`;
    
    ['H₂O', 'CO₂', 'CH₄', 'O₃', 'O₂', 'NH₃', 'CO', 'N₂O'].forEach(mol => {
      const result = spec.analysis.molecules.find(m => m.formula === mol);
      csv += result?.detected ? `${(result.confidence * 100).toFixed(0)}%,` : 'ND,';
    });
    
    csv += `${spec.metadata.telescope},${spec.metadata.instrument},${spec.metadata.observationDate},`;
    csv += `${spec.metadata.snr},${spec.metadata.resolution}\n`;
  });
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

const exportToJSON = (spectra: SpectrumEntry[], filename: string) => {
  const data = spectra.map(spec => ({
    target: spec.target,
    classification: spec.analysis.planetClass,
    temperature_K: spec.analysis.temperature,
    habitability_index: spec.analysis.habitability,
    molecules: Object.fromEntries(
      spec.analysis.molecules.map(m => [m.formula, {
        detected: m.detected,
        confidence: m.confidence,
        abundance: m.abundance
      }])
    ),
    observation: {
      telescope: spec.metadata.telescope,
      instrument: spec.metadata.instrument,
      date: spec.metadata.observationDate,
      snr: spec.metadata.snr,
      resolution: spec.metadata.resolution,
      exposure_seconds: spec.metadata.exposureTime
    }
  }));
  
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

// ===== Sub-Components =====

// Spectrum Selector
interface SpectrumSelectorProps {
  spectra: SpectrumEntry[];
  selected: string[];
  onSelectionChange: (ids: string[]) => void;
  maxSelection?: number;
}

const SpectrumSelector: React.FC<SpectrumSelectorProps> = ({
  spectra, selected, onSelectionChange, maxSelection = 6
}) => {
  const toggleSpectrum = (id: string) => {
    if (selected.includes(id)) {
      onSelectionChange(selected.filter(s => s !== id));
    } else if (selected.length < maxSelection) {
      onSelectionChange([...selected, id]);
    }
  };
  
  return (
    <div className="spectrum-selector">
      <div className="selector-header">
        <h3>Select Spectra to Compare</h3>
        <span className="selection-count">{selected.length}/{maxSelection}</span>
      </div>
      <div className="selector-list">
        {spectra.map((spec, idx) => (
          <div
            key={spec.id}
            className={`selector-item ${selected.includes(spec.id) ? 'selected' : ''}`}
            onClick={() => toggleSpectrum(spec.id)}
          >
            <div
              className="selector-color"
              style={{ backgroundColor: selected.includes(spec.id) 
                ? COLOR_PALETTES.distinct[selected.indexOf(spec.id)] 
                : '#333' 
              }}
            />
            <div className="selector-info">
              <span className="selector-target">{spec.target}</span>
              <span className="selector-meta">{spec.metadata.telescope} • {spec.analysis.planetClass}</span>
            </div>
            <div className="selector-hab" style={{
              color: spec.analysis.habitability > 0.7 ? '#00ff88' :
                     spec.analysis.habitability > 0.5 ? '#ffd700' : '#888'
            }}>
              {(spec.analysis.habitability * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Comparison Plot
interface ComparisonPlotProps {
  spectra: SpectrumEntry[];
  state: ComparisonState;
  colors: string[];
  onRangeChange: (range: [number, number]) => void;
}

const ComparisonPlot: React.FC<ComparisonPlotProps> = ({
  spectra, state, colors, onRangeChange
}) => {
  const plotRef = useRef<any>(null);
  
  const plotData = useMemo((): Data[] => {
    const traces: Data[] = [];
    
    spectra.forEach((spec, idx) => {
      const color = colors[idx];
      
      // Normalize if needed
      let fluxData = [...spec.flux];
      if (state.normalizeFlux) {
        const max = Math.max(...fluxData);
        fluxData = fluxData.map(f => f / max);
      }
      
      // Uncertainty band
      if (state.showUncertainties && spec.uncertainties) {
        traces.push({
          x: [...spec.wavelengths, ...spec.wavelengths.slice().reverse()],
          y: [
            ...fluxData.map((f, i) => f + (spec.uncertainties?.[i] || 0)),
            ...fluxData.map((f, i) => f - (spec.uncertainties?.[i] || 0)).reverse()
          ],
          type: 'scatter',
          fill: 'toself',
          fillcolor: `${color}15`,
          line: { color: 'transparent' },
          showlegend: false,
          hoverinfo: 'skip'
        });
      }
      
      // Main trace
      traces.push({
        x: spec.wavelengths,
        y: fluxData,
        type: 'scatter',
        mode: 'lines',
        name: spec.target,
        line: { color, width: 2 },
        hovertemplate: `${spec.target}<br>λ: %{x:.3f} μm<br>Flux: %{y:.4f}<extra></extra>`
      });
    });
    
    return traces;
  }, [spectra, state, colors]);
  
  const layout = useMemo((): Partial<Layout> => ({
    title: {
      text: 'Exoplanet Spectra Comparison',
      font: { color: '#ffffff', size: 18, family: 'Inter' }
    },
    paper_bgcolor: '#0a0a1a',
    plot_bgcolor: '#0d0d1f',
    font: { color: '#888888', family: 'Inter' },
    xaxis: {
      title: { text: 'Wavelength (μm)', font: { color: '#aaa' } },
      color: '#666',
      gridcolor: 'rgba(255,255,255,0.05)',
      range: state.wavelengthRange,
      rangeslider: { visible: true, bgcolor: '#1a1a2e' }
    },
    yaxis: {
      title: { text: state.normalizeFlux ? 'Normalized Flux' : 'Flux', font: { color: '#aaa' } },
      color: '#666',
      gridcolor: 'rgba(255,255,255,0.05)'
    },
    legend: {
      x: 1.02,
      y: 1,
      bgcolor: 'rgba(10,10,26,0.9)',
      bordercolor: 'rgba(255,255,255,0.1)',
      font: { color: '#ccc' }
    },
    margin: { t: 60, r: 150, b: 80, l: 70 },
    hovermode: 'x unified'
  }), [state]);
  
  return (
    <div className="comparison-plot">
      <Plot
        ref={plotRef}
        data={plotData}
        layout={layout}
        config={{
          displayModeBar: true,
          modeBarButtonsToAdd: ['downloadSVG'],
          displaylogo: false,
          responsive: true,
          scrollZoom: true,
          toImageButtonOptions: {
            format: 'svg',
            filename: 'exoplanet_spectra_comparison',
            width: 1200,
            height: 600
          }
        }}
        style={{ width: '100%', height: '500px' }}
        onRelayout={(e: any) => {
          if (e['xaxis.range[0]'] && e['xaxis.range[1]']) {
            onRangeChange([e['xaxis.range[0]'], e['xaxis.range[1]']]);
          }
        }}
      />
    </div>
  );
};

// Molecule Detection Matrix
interface MoleculeMatrixProps {
  spectra: SpectrumEntry[];
  colors: string[];
}

const MoleculeMatrix: React.FC<MoleculeMatrixProps> = ({ spectra, colors }) => {
  const molecules = ['H₂O', 'CO₂', 'CH₄', 'O₃', 'O₂', 'NH₃', 'CO', 'N₂O'];
  
  return (
    <div className="molecule-matrix">
      <h3>🧪 Molecule Detection Matrix</h3>
      <div className="matrix-container">
        <table>
          <thead>
            <tr>
              <th>Target</th>
              {molecules.map(mol => (
                <th key={mol}>{mol}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {spectra.map((spec, idx) => (
              <tr key={spec.id}>
                <td className="target-cell">
                  <span className="color-dot" style={{ backgroundColor: colors[idx] }} />
                  {spec.target}
                </td>
                {molecules.map(mol => {
                  const result = spec.analysis.molecules.find(m => m.formula === mol);
                  return (
                    <td key={mol} className={`detection-cell ${result?.detected ? 'detected' : 'not-detected'}`}>
                      {result?.detected ? (
                        <span className="confidence" style={{
                          backgroundColor: `rgba(0, 255, 136, ${result.confidence})`,
                          color: result.confidence > 0.6 ? '#0a0a1a' : '#00ff88'
                        }}>
                          {(result.confidence * 100).toFixed(0)}%
                        </span>
                      ) : (
                        <span className="not-detected-mark">—</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="matrix-legend">
        <span className="legend-item">
          <span className="legend-box detected" /> Detected (confidence shown)
        </span>
        <span className="legend-item">
          <span className="legend-box not-detected" /> Not Detected
        </span>
      </div>
    </div>
  );
};

// Summary Statistics Table
interface SummaryTableProps {
  spectra: SpectrumEntry[];
  colors: string[];
}

const SummaryTable: React.FC<SummaryTableProps> = ({ spectra, colors }) => {
  const [sortKey, setSortKey] = useState<string>('target');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  
  const columns: TableColumn[] = [
    { key: 'target', label: 'Target', sortable: true },
    { key: 'planetClass', label: 'Class', sortable: true },
    { key: 'temperature', label: 'Temp (K)', sortable: true, format: v => v.toFixed(0) },
    { key: 'habitability', label: 'Habitability', sortable: true, format: v => `${(v * 100).toFixed(0)}%` },
    { key: 'moleculeCount', label: 'Molecules', sortable: true },
    { key: 'snr', label: 'SNR', sortable: true },
    { key: 'telescope', label: 'Telescope', sortable: true },
    { key: 'resolution', label: 'Resolution', sortable: true, format: v => v.toLocaleString() }
  ];
  
  const tableData = useMemo(() => {
    return spectra.map((spec, idx) => ({
      id: spec.id,
      color: colors[idx],
      target: spec.target,
      planetClass: spec.analysis.planetClass,
      temperature: spec.analysis.temperature,
      habitability: spec.analysis.habitability,
      moleculeCount: spec.analysis.molecules.filter(m => m.detected).length,
      snr: spec.metadata.snr,
      telescope: spec.metadata.telescope,
      resolution: spec.metadata.resolution
    }));
  }, [spectra, colors]);
  
  const sortedData = useMemo(() => {
    return [...tableData].sort((a, b) => {
      const aVal = a[sortKey as keyof typeof a];
      const bVal = b[sortKey as keyof typeof b];
      const cmp = typeof aVal === 'string' 
        ? aVal.localeCompare(bVal as string)
        : (aVal as number) - (bVal as number);
      return sortDirection === 'asc' ? cmp : -cmp;
    });
  }, [tableData, sortKey, sortDirection]);
  
  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDirection('asc');
    }
  };
  
  return (
    <div className="summary-table">
      <h3>📊 Summary Statistics</h3>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              {columns.map(col => (
                <th
                  key={col.key}
                  className={col.sortable ? 'sortable' : ''}
                  onClick={() => col.sortable && handleSort(col.key)}
                >
                  {col.label}
                  {sortKey === col.key && (
                    <span className="sort-indicator">
                      {sortDirection === 'asc' ? ' ↑' : ' ↓'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedData.map(row => (
              <tr key={row.id}>
                <td>
                  <span className="color-dot" style={{ backgroundColor: row.color }} />
                  {row.target}
                </td>
                <td>{row.planetClass}</td>
                <td>{row.temperature}</td>
                <td style={{
                  color: row.habitability > 0.7 ? '#00ff88' :
                         row.habitability > 0.5 ? '#ffd700' : '#888'
                }}>
                  {(row.habitability * 100).toFixed(0)}%
                </td>
                <td>{row.moleculeCount}</td>
                <td>{row.snr}</td>
                <td>{row.telescope}</td>
                <td>{row.resolution.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// Comparison Statistics
interface ComparisonStatsProps {
  spectra: SpectrumEntry[];
}

const ComparisonStats: React.FC<ComparisonStatsProps> = ({ spectra }) => {
  const stats = useMemo(() => {
    const habs = spectra.map(s => s.analysis.habitability);
    const temps = spectra.map(s => s.analysis.temperature);
    const molCounts = spectra.map(s => s.analysis.molecules.filter(m => m.detected).length);
    
    const commonMolecules = ['H₂O', 'CO₂', 'CH₄', 'O₃', 'O₂', 'NH₃', 'CO', 'N₂O'].filter(mol =>
      spectra.every(s => s.analysis.molecules.find(m => m.formula === mol)?.detected)
    );
    
    const uniqueMolecules = new Map<string, string[]>();
    spectra.forEach(spec => {
      spec.analysis.molecules.filter(m => m.detected).forEach(mol => {
        if (!uniqueMolecules.has(mol.formula)) {
          uniqueMolecules.set(mol.formula, []);
        }
        uniqueMolecules.get(mol.formula)!.push(spec.target);
      });
    });
    
    return {
      avgHabitability: habs.reduce((a, b) => a + b, 0) / habs.length,
      maxHabitability: Math.max(...habs),
      minHabitability: Math.min(...habs),
      avgTemp: temps.reduce((a, b) => a + b, 0) / temps.length,
      avgMolecules: molCounts.reduce((a, b) => a + b, 0) / molCounts.length,
      commonMolecules,
      totalUniqueMolecules: uniqueMolecules.size
    };
  }, [spectra]);
  
  return (
    <div className="comparison-stats">
      <h3>📈 Comparison Statistics</h3>
      <div className="stats-grid">
        <div className="stat-card">
          <span className="stat-value">{spectra.length}</span>
          <span className="stat-label">Spectra Compared</span>
        </div>
        <div className="stat-card">
          <span className="stat-value" style={{ color: '#00ff88' }}>
            {(stats.avgHabitability * 100).toFixed(0)}%
          </span>
          <span className="stat-label">Avg Habitability</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{stats.avgTemp.toFixed(0)} K</span>
          <span className="stat-label">Avg Temperature</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{stats.avgMolecules.toFixed(1)}</span>
          <span className="stat-label">Avg Molecules/Target</span>
        </div>
        <div className="stat-card wide">
          <span className="stat-value">{stats.commonMolecules.length > 0 ? stats.commonMolecules.join(', ') : 'None'}</span>
          <span className="stat-label">Common Detections</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{stats.totalUniqueMolecules}</span>
          <span className="stat-label">Total Species</span>
        </div>
      </div>
    </div>
  );
};

// ===== Main Component =====
const ResearchComparison: React.FC = () => {
  // Data
  const [allSpectra] = useState<SpectrumEntry[]>(generateDemoSpectra);
  
  // State
  const [comparisonState, setComparisonState] = useState<ComparisonState>({
    selectedSpectra: [allSpectra[0].id, allSpectra[1].id],
    syncZoom: true,
    normalizeFlux: true,
    showUncertainties: true,
    showMoleculeOverlay: false,
    wavelengthRange: [0.6, 28],
    colorScheme: 'distinct'
  });
  
  // Selected spectra data
  const selectedSpectraData = useMemo(() => {
    return comparisonState.selectedSpectra
      .map(id => allSpectra.find(s => s.id === id))
      .filter((s): s is SpectrumEntry => s !== undefined);
  }, [allSpectra, comparisonState.selectedSpectra]);
  
  // Colors for selected spectra
  const spectraColors = useMemo(() => {
    return comparisonState.selectedSpectra.map((_, idx) => 
      COLOR_PALETTES[comparisonState.colorScheme][idx % 8]
    );
  }, [comparisonState.selectedSpectra, comparisonState.colorScheme]);
  
  // Handlers
  const handleSelectionChange = useCallback((ids: string[]) => {
    setComparisonState(prev => ({ ...prev, selectedSpectra: ids }));
  }, []);
  
  const handleRangeChange = useCallback((range: [number, number]) => {
    setComparisonState(prev => ({ ...prev, wavelengthRange: range }));
  }, []);
  
  const toggleOption = useCallback((key: keyof ComparisonState) => {
    setComparisonState(prev => ({ ...prev, [key]: !prev[key] }));
  }, []);
  
  return (
    <div className="research-page">
      {/* Header */}
      <header className="research-header">
        <div className="header-left">
          <a href="/" className="back-link">← Mission Control</a>
          <h1>Research Comparison</h1>
        </div>
        <div className="header-right">
          <div className="export-buttons">
            <button 
              className="export-btn"
              onClick={() => exportToCSV(selectedSpectraData, 'exoplanet_comparison.csv')}
              disabled={selectedSpectraData.length === 0}
            >
              📄 Export CSV
            </button>
            <button 
              className="export-btn"
              onClick={() => exportToJSON(selectedSpectraData, 'exoplanet_comparison.json')}
              disabled={selectedSpectraData.length === 0}
            >
              📋 Export JSON
            </button>
          </div>
        </div>
      </header>
      
      <main className="research-main">
        {/* Left Sidebar - Selection & Options */}
        <aside className="selection-sidebar">
          <SpectrumSelector
            spectra={allSpectra}
            selected={comparisonState.selectedSpectra}
            onSelectionChange={handleSelectionChange}
          />
          
          <div className="plot-options">
            <h3>⚙️ Display Options</h3>
            
            <label className="option-toggle">
              <input
                type="checkbox"
                checked={comparisonState.normalizeFlux}
                onChange={() => toggleOption('normalizeFlux')}
              />
              <span>Normalize Flux</span>
            </label>
            
            <label className="option-toggle">
              <input
                type="checkbox"
                checked={comparisonState.showUncertainties}
                onChange={() => toggleOption('showUncertainties')}
              />
              <span>Show Uncertainties</span>
            </label>
            
            <label className="option-toggle">
              <input
                type="checkbox"
                checked={comparisonState.syncZoom}
                onChange={() => toggleOption('syncZoom')}
              />
              <span>Sync Zoom</span>
            </label>
            
            <div className="color-scheme-selector">
              <label>Color Scheme</label>
              <div className="scheme-buttons">
                {(['distinct', 'sequential', 'diverging'] as const).map(scheme => (
                  <button
                    key={scheme}
                    className={comparisonState.colorScheme === scheme ? 'active' : ''}
                    onClick={() => setComparisonState(prev => ({ ...prev, colorScheme: scheme }))}
                  >
                    <div className="scheme-preview">
                      {COLOR_PALETTES[scheme].slice(0, 4).map((c, i) => (
                        <span key={i} style={{ backgroundColor: c }} />
                      ))}
                    </div>
                    {scheme}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </aside>
        
        {/* Main Content */}
        <div className="research-content">
          {selectedSpectraData.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">🔭</div>
              <h2>Select Spectra to Compare</h2>
              <p>Choose 2-6 exoplanet spectra from the left panel to begin comparison.</p>
            </div>
          ) : (
            <>
              {/* Comparison Plot */}
              <section className="content-section">
                <ComparisonPlot
                  spectra={selectedSpectraData}
                  state={comparisonState}
                  colors={spectraColors}
                  onRangeChange={handleRangeChange}
                />
              </section>
              
              {/* Statistics */}
              <section className="content-section">
                <ComparisonStats spectra={selectedSpectraData} />
              </section>
              
              {/* Molecule Matrix */}
              <section className="content-section">
                <MoleculeMatrix
                  spectra={selectedSpectraData}
                  colors={spectraColors}
                />
              </section>
              
              {/* Summary Table */}
              <section className="content-section">
                <SummaryTable
                  spectra={selectedSpectraData}
                  colors={spectraColors}
                />
              </section>
            </>
          )}
        </div>
      </main>
    </div>
  );
};

export default ResearchComparison;
