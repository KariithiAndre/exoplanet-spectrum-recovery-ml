/**
 * Analysis Results Panel
 * 
 * Comprehensive display component for spectrum analysis results:
 * - Detected molecules with confidence and abundance
 * - Planet classification probabilities
 * - Habitability probability gauge
 * - Confidence bars with uncertainty ranges
 * - Uncertainty visualization (error bars, distributions)
 */

import React, { useState, useMemo } from 'react';
import './ResultsPanel.css';

// ===== Type Definitions =====
interface MoleculeDetection {
  name: string;
  formula: string;
  detected: boolean;
  confidence: number;
  confidenceLower: number;
  confidenceUpper: number;
  abundance?: number;
  abundanceUncertainty?: number;
  significance: number; // sigma
  category: 'biosignature' | 'atmospheric' | 'volcanic' | 'industrial';
}

interface PlanetClassification {
  type: string;
  probability: number;
  probabilityLower: number;
  probabilityUpper: number;
  description: string;
  icon: string;
}

interface HabitabilityMetrics {
  overall: number;
  overallUncertainty: number;
  temperature: { value: number; uncertainty: number; label: string };
  atmosphere: { value: number; uncertainty: number; label: string };
  water: { value: number; uncertainty: number; label: string };
  stability: { value: number; uncertainty: number; label: string };
}

interface UncertaintySource {
  name: string;
  contribution: number;
  type: 'systematic' | 'statistical' | 'model';
}

interface AnalysisResults {
  id: string;
  timestamp: string;
  molecules: MoleculeDetection[];
  planetClasses: PlanetClassification[];
  habitability: HabitabilityMetrics;
  uncertaintySources: UncertaintySource[];
  modelConfidence: number;
  dataQuality: number;
}

// ===== Demo Data Generator =====
const generateDemoResults = (): AnalysisResults => ({
  id: 'analysis_2026_001',
  timestamp: new Date().toISOString(),
  molecules: [
    {
      name: 'Water',
      formula: 'H₂O',
      detected: true,
      confidence: 0.94,
      confidenceLower: 0.89,
      confidenceUpper: 0.97,
      abundance: 0.002,
      abundanceUncertainty: 0.0005,
      significance: 8.2,
      category: 'biosignature'
    },
    {
      name: 'Carbon Dioxide',
      formula: 'CO₂',
      detected: true,
      confidence: 0.91,
      confidenceLower: 0.85,
      confidenceUpper: 0.95,
      abundance: 0.0004,
      abundanceUncertainty: 0.0001,
      significance: 7.1,
      category: 'atmospheric'
    },
    {
      name: 'Methane',
      formula: 'CH₄',
      detected: true,
      confidence: 0.78,
      confidenceLower: 0.68,
      confidenceUpper: 0.86,
      abundance: 0.00002,
      abundanceUncertainty: 0.000008,
      significance: 4.5,
      category: 'biosignature'
    },
    {
      name: 'Ozone',
      formula: 'O₃',
      detected: true,
      confidence: 0.65,
      confidenceLower: 0.52,
      confidenceUpper: 0.76,
      abundance: 0.000001,
      abundanceUncertainty: 0.0000005,
      significance: 3.2,
      category: 'biosignature'
    },
    {
      name: 'Oxygen',
      formula: 'O₂',
      detected: false,
      confidence: 0.35,
      confidenceLower: 0.22,
      confidenceUpper: 0.48,
      significance: 1.8,
      category: 'biosignature'
    },
    {
      name: 'Ammonia',
      formula: 'NH₃',
      detected: false,
      confidence: 0.28,
      confidenceLower: 0.15,
      confidenceUpper: 0.42,
      significance: 1.2,
      category: 'atmospheric'
    },
    {
      name: 'Sulfur Dioxide',
      formula: 'SO₂',
      detected: false,
      confidence: 0.15,
      confidenceLower: 0.05,
      confidenceUpper: 0.28,
      significance: 0.8,
      category: 'volcanic'
    }
  ],
  planetClasses: [
    {
      type: 'Terrestrial',
      probability: 0.72,
      probabilityLower: 0.65,
      probabilityUpper: 0.79,
      description: 'Rocky planet with solid surface',
      icon: '🌍'
    },
    {
      type: 'Super-Earth',
      probability: 0.18,
      probabilityLower: 0.12,
      probabilityUpper: 0.25,
      description: 'Larger rocky planet, 1-10 Earth masses',
      icon: '🪨'
    },
    {
      type: 'Mini-Neptune',
      probability: 0.07,
      probabilityLower: 0.03,
      probabilityUpper: 0.12,
      description: 'Small gas/ice giant',
      icon: '🔵'
    },
    {
      type: 'Ocean World',
      probability: 0.03,
      probabilityLower: 0.01,
      probabilityUpper: 0.06,
      description: 'Water-dominated surface',
      icon: '🌊'
    }
  ],
  habitability: {
    overall: 0.68,
    overallUncertainty: 0.12,
    temperature: { value: 0.75, uncertainty: 0.15, label: 'Surface Temperature' },
    atmosphere: { value: 0.82, uncertainty: 0.10, label: 'Atmospheric Composition' },
    water: { value: 0.65, uncertainty: 0.18, label: 'Water Presence' },
    stability: { value: 0.52, uncertainty: 0.20, label: 'Orbital Stability' }
  },
  uncertaintySources: [
    { name: 'Photon Noise', contribution: 0.35, type: 'statistical' },
    { name: 'Stellar Contamination', contribution: 0.25, type: 'systematic' },
    { name: 'Model Assumptions', contribution: 0.20, type: 'model' },
    { name: 'Calibration Error', contribution: 0.12, type: 'systematic' },
    { name: 'Background Subtraction', contribution: 0.08, type: 'statistical' }
  ],
  modelConfidence: 0.85,
  dataQuality: 0.92
});

// ===== Molecule Colors =====
const MOLECULE_COLORS: Record<string, string> = {
  'H₂O': '#00bfff',
  'CO₂': '#ff6b6b',
  'CH₄': '#ffd700',
  'O₃': '#9b59b6',
  'O₂': '#3498db',
  'NH₃': '#e67e22',
  'SO₂': '#795548',
  'CO': '#1abc9c',
  'N₂O': '#e91e63',
  'H₂S': '#607d8b'
};

const CATEGORY_COLORS: Record<string, string> = {
  biosignature: '#00ff88',
  atmospheric: '#00bfff',
  volcanic: '#ff6b35',
  industrial: '#9b59b6'
};

// ===== Sub-Components =====

// Confidence Bar with Uncertainty Range
interface ConfidenceBarProps {
  value: number;
  lower: number;
  upper: number;
  color: string;
  showLabels?: boolean;
  height?: number;
}

const ConfidenceBar: React.FC<ConfidenceBarProps> = ({
  value,
  lower,
  upper,
  color,
  showLabels = true,
  height = 8
}) => {
  const uncertaintyWidth = (upper - lower) * 100;
  const uncertaintyLeft = lower * 100;
  
  return (
    <div className="confidence-bar-container">
      <div className="confidence-bar-track" style={{ height }}>
        {/* Uncertainty range */}
        <div
          className="confidence-uncertainty"
          style={{
            left: `${uncertaintyLeft}%`,
            width: `${uncertaintyWidth}%`,
            backgroundColor: `${color}40`
          }}
        />
        {/* Main value */}
        <div
          className="confidence-fill"
          style={{
            width: `${value * 100}%`,
            backgroundColor: color
          }}
        />
        {/* Uncertainty bounds markers */}
        <div
          className="uncertainty-marker lower"
          style={{ left: `${lower * 100}%`, borderColor: color }}
        />
        <div
          className="uncertainty-marker upper"
          style={{ left: `${upper * 100}%`, borderColor: color }}
        />
      </div>
      {showLabels && (
        <div className="confidence-labels">
          <span className="lower-bound">{(lower * 100).toFixed(0)}%</span>
          <span className="main-value" style={{ color }}>{(value * 100).toFixed(0)}%</span>
          <span className="upper-bound">{(upper * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
};

// Circular Gauge for Habitability
interface CircularGaugeProps {
  value: number;
  uncertainty: number;
  size?: number;
  label: string;
}

const CircularGauge: React.FC<CircularGaugeProps> = ({
  value,
  uncertainty,
  size = 200,
  label
}) => {
  const radius = (size - 20) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeWidth = 12;
  
  const mainOffset = circumference * (1 - value);
  const lowerOffset = circumference * (1 - Math.max(0, value - uncertainty));
  const upperOffset = circumference * (1 - Math.min(1, value + uncertainty));
  
  // Color gradient based on value
  const getColor = (v: number) => {
    if (v < 0.3) return '#ff4444';
    if (v < 0.5) return '#ff8c00';
    if (v < 0.7) return '#ffd700';
    return '#00ff88';
  };
  
  return (
    <div className="circular-gauge" style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${size} ${size}`}>
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={strokeWidth}
        />
        
        {/* Uncertainty range (lower) */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={`${getColor(value)}30`}
          strokeWidth={strokeWidth + 8}
          strokeDasharray={circumference}
          strokeDashoffset={lowerOffset}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
        
        {/* Main value arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={getColor(value)}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={mainOffset}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset 1s ease' }}
        />
        
        {/* Glow effect */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={getColor(value)}
          strokeWidth={2}
          strokeDasharray={circumference}
          strokeDashoffset={mainOffset}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          filter="url(#glow)"
          opacity={0.6}
        />
        
        {/* Glow filter */}
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      </svg>
      
      <div className="gauge-center">
        <span className="gauge-value" style={{ color: getColor(value) }}>
          {(value * 100).toFixed(0)}%
        </span>
        <span className="gauge-uncertainty">
          ±{(uncertainty * 100).toFixed(0)}%
        </span>
        <span className="gauge-label">{label}</span>
      </div>
    </div>
  );
};

// Mini metric bar
interface MetricBarProps {
  label: string;
  value: number;
  uncertainty: number;
}

const MetricBar: React.FC<MetricBarProps> = ({ label, value, uncertainty }) => {
  const getColor = (v: number) => {
    if (v < 0.3) return '#ff4444';
    if (v < 0.5) return '#ff8c00';
    if (v < 0.7) return '#ffd700';
    return '#00ff88';
  };
  
  return (
    <div className="metric-bar">
      <div className="metric-header">
        <span className="metric-label">{label}</span>
        <span className="metric-value" style={{ color: getColor(value) }}>
          {(value * 100).toFixed(0)}% <span className="metric-uncertainty">±{(uncertainty * 100).toFixed(0)}</span>
        </span>
      </div>
      <div className="metric-track">
        <div
          className="metric-uncertainty-range"
          style={{
            left: `${Math.max(0, value - uncertainty) * 100}%`,
            width: `${Math.min(uncertainty * 2, 1 - value + uncertainty) * 100}%`,
            backgroundColor: `${getColor(value)}30`
          }}
        />
        <div
          className="metric-fill"
          style={{
            width: `${value * 100}%`,
            backgroundColor: getColor(value)
          }}
        />
      </div>
    </div>
  );
};

// Uncertainty Breakdown Chart
interface UncertaintyChartProps {
  sources: UncertaintySource[];
}

const UncertaintyChart: React.FC<UncertaintyChartProps> = ({ sources }) => {
  const typeColors: Record<string, string> = {
    statistical: '#00bfff',
    systematic: '#ff6b6b',
    model: '#ffd700'
  };
  
  const sortedSources = [...sources].sort((a, b) => b.contribution - a.contribution);
  
  return (
    <div className="uncertainty-chart">
      {sortedSources.map((source, idx) => (
        <div key={source.name} className="uncertainty-source">
          <div className="source-info">
            <span className="source-name">{source.name}</span>
            <span className="source-type" style={{ color: typeColors[source.type] }}>
              {source.type}
            </span>
          </div>
          <div className="source-bar">
            <div
              className="source-fill"
              style={{
                width: `${source.contribution * 100}%`,
                backgroundColor: typeColors[source.type],
                animationDelay: `${idx * 0.1}s`
              }}
            />
          </div>
          <span className="source-value">{(source.contribution * 100).toFixed(0)}%</span>
        </div>
      ))}
      
      <div className="uncertainty-legend">
        <span className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: typeColors.statistical }} />
          Statistical
        </span>
        <span className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: typeColors.systematic }} />
          Systematic
        </span>
        <span className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: typeColors.model }} />
          Model
        </span>
      </div>
    </div>
  );
};

// Significance Indicator
interface SignificanceIndicatorProps {
  sigma: number;
}

const SignificanceIndicator: React.FC<SignificanceIndicatorProps> = ({ sigma }) => {
  const getLevel = () => {
    if (sigma >= 5) return { label: 'Discovery', color: '#00ff88', stars: 5 };
    if (sigma >= 3) return { label: 'Evidence', color: '#ffd700', stars: 3 };
    if (sigma >= 2) return { label: 'Hint', color: '#ff8c00', stars: 2 };
    return { label: 'Tentative', color: '#888', stars: 1 };
  };
  
  const level = getLevel();
  
  return (
    <div className="significance-indicator">
      <span className="sigma-value" style={{ color: level.color }}>
        {sigma.toFixed(1)}σ
      </span>
      <div className="sigma-stars">
        {[1, 2, 3, 4, 5].map(i => (
          <span
            key={i}
            className={`star ${i <= level.stars ? 'active' : ''}`}
            style={{ color: i <= level.stars ? level.color : '#333' }}
          >
            ★
          </span>
        ))}
      </div>
      <span className="sigma-label" style={{ color: level.color }}>
        {level.label}
      </span>
    </div>
  );
};

// ===== Main Results Panel Component =====
interface ResultsPanelProps {
  results?: AnalysisResults;
  compact?: boolean;
}

const ResultsPanel: React.FC<ResultsPanelProps> = ({ results, compact = false }) => {
  const [activeSection, setActiveSection] = useState<'molecules' | 'classification' | 'habitability' | 'uncertainty'>('molecules');
  const [showOnlyDetected, setShowOnlyDetected] = useState(false);
  
  // Use demo data if no results provided
  const data = useMemo(() => results || generateDemoResults(), [results]);
  
  // Filter molecules
  const displayedMolecules = useMemo(() => {
    return showOnlyDetected 
      ? data.molecules.filter(m => m.detected)
      : data.molecules;
  }, [data.molecules, showOnlyDetected]);
  
  // Sort by confidence
  const sortedMolecules = useMemo(() => {
    return [...displayedMolecules].sort((a, b) => b.confidence - a.confidence);
  }, [displayedMolecules]);
  
  // Count biosignatures
  const biosignatureCount = useMemo(() => {
    return data.molecules.filter(m => m.detected && m.category === 'biosignature').length;
  }, [data.molecules]);
  
  if (compact) {
    return (
      <div className="results-panel compact">
        <div className="compact-header">
          <h3>Analysis Results</h3>
          <span className="quality-badge" style={{ 
            backgroundColor: data.dataQuality > 0.8 ? '#00ff88' : data.dataQuality > 0.5 ? '#ffd700' : '#ff4444' 
          }}>
            {(data.dataQuality * 100).toFixed(0)}% Quality
          </span>
        </div>
        
        <div className="compact-grid">
          <div className="compact-stat">
            <span className="stat-value">{data.molecules.filter(m => m.detected).length}</span>
            <span className="stat-label">Molecules</span>
          </div>
          <div className="compact-stat">
            <span className="stat-value" style={{ color: '#00ff88' }}>{biosignatureCount}</span>
            <span className="stat-label">Biosignatures</span>
          </div>
          <div className="compact-stat">
            <span className="stat-value">{(data.habitability.overall * 100).toFixed(0)}%</span>
            <span className="stat-label">Habitability</span>
          </div>
          <div className="compact-stat">
            <span className="stat-value">{data.planetClasses[0].type}</span>
            <span className="stat-label">Classification</span>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="results-panel">
      {/* Header */}
      <div className="panel-header">
        <h2>🔬 Analysis Results</h2>
        <div className="header-badges">
          <span className="quality-badge" style={{ 
            backgroundColor: data.dataQuality > 0.8 ? 'rgba(0,255,136,0.2)' : 'rgba(255,215,0,0.2)',
            borderColor: data.dataQuality > 0.8 ? '#00ff88' : '#ffd700',
            color: data.dataQuality > 0.8 ? '#00ff88' : '#ffd700'
          }}>
            📊 {(data.dataQuality * 100).toFixed(0)}% Data Quality
          </span>
          <span className="confidence-badge" style={{ 
            backgroundColor: 'rgba(0,191,255,0.2)',
            borderColor: '#00bfff',
            color: '#00bfff'
          }}>
            🎯 {(data.modelConfidence * 100).toFixed(0)}% Model Confidence
          </span>
        </div>
      </div>
      
      {/* Section Navigation */}
      <div className="section-nav">
        <button
          className={activeSection === 'molecules' ? 'active' : ''}
          onClick={() => setActiveSection('molecules')}
        >
          🧪 Molecules
        </button>
        <button
          className={activeSection === 'classification' ? 'active' : ''}
          onClick={() => setActiveSection('classification')}
        >
          🌍 Classification
        </button>
        <button
          className={activeSection === 'habitability' ? 'active' : ''}
          onClick={() => setActiveSection('habitability')}
        >
          💚 Habitability
        </button>
        <button
          className={activeSection === 'uncertainty' ? 'active' : ''}
          onClick={() => setActiveSection('uncertainty')}
        >
          📈 Uncertainty
        </button>
      </div>
      
      {/* Section Content */}
      <div className="section-content">
        {/* Molecules Section */}
        {activeSection === 'molecules' && (
          <div className="molecules-section">
            <div className="section-toolbar">
              <label className="filter-toggle">
                <input
                  type="checkbox"
                  checked={showOnlyDetected}
                  onChange={(e) => setShowOnlyDetected(e.target.checked)}
                />
                <span>Show only detected</span>
              </label>
              <span className="detection-summary">
                {data.molecules.filter(m => m.detected).length} of {data.molecules.length} detected
              </span>
            </div>
            
            <div className="molecules-list">
              {sortedMolecules.map(molecule => (
                <div
                  key={molecule.formula}
                  className={`molecule-card ${molecule.detected ? 'detected' : 'not-detected'}`}
                >
                  <div className="molecule-header">
                    <div className="molecule-identity">
                      <span
                        className="molecule-formula"
                        style={{ color: MOLECULE_COLORS[molecule.formula] || '#888' }}
                      >
                        {molecule.formula}
                      </span>
                      <span className="molecule-name">{molecule.name}</span>
                    </div>
                    <div className="molecule-status">
                      <span
                        className="category-badge"
                        style={{ 
                          backgroundColor: `${CATEGORY_COLORS[molecule.category]}20`,
                          color: CATEGORY_COLORS[molecule.category]
                        }}
                      >
                        {molecule.category}
                      </span>
                      <span className={`detection-badge ${molecule.detected ? 'yes' : 'no'}`}>
                        {molecule.detected ? '✓ Detected' : '✗ Not Detected'}
                      </span>
                    </div>
                  </div>
                  
                  <div className="molecule-metrics">
                    <div className="confidence-section">
                      <span className="metric-title">Detection Confidence</span>
                      <ConfidenceBar
                        value={molecule.confidence}
                        lower={molecule.confidenceLower}
                        upper={molecule.confidenceUpper}
                        color={MOLECULE_COLORS[molecule.formula] || '#ffd700'}
                      />
                    </div>
                    
                    <div className="molecule-details">
                      <SignificanceIndicator sigma={molecule.significance} />
                      
                      {molecule.abundance && (
                        <div className="abundance-info">
                          <span className="abundance-label">Abundance</span>
                          <span className="abundance-value">
                            {molecule.abundance.toExponential(1)}
                            {molecule.abundanceUncertainty && (
                              <span className="abundance-uncertainty">
                                ±{molecule.abundanceUncertainty.toExponential(1)}
                              </span>
                            )}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Classification Section */}
        {activeSection === 'classification' && (
          <div className="classification-section">
            <div className="primary-classification">
              <div className="primary-icon">{data.planetClasses[0].icon}</div>
              <div className="primary-info">
                <span className="primary-type">{data.planetClasses[0].type}</span>
                <span className="primary-description">{data.planetClasses[0].description}</span>
                <ConfidenceBar
                  value={data.planetClasses[0].probability}
                  lower={data.planetClasses[0].probabilityLower}
                  upper={data.planetClasses[0].probabilityUpper}
                  color="#ffd700"
                />
              </div>
            </div>
            
            <div className="classification-list">
              <h4>Alternative Classifications</h4>
              {data.planetClasses.slice(1).map(cls => (
                <div key={cls.type} className="classification-item">
                  <span className="class-icon">{cls.icon}</span>
                  <div className="class-info">
                    <span className="class-type">{cls.type}</span>
                    <span className="class-description">{cls.description}</span>
                  </div>
                  <div className="class-probability">
                    <span className="prob-value">{(cls.probability * 100).toFixed(0)}%</span>
                    <span className="prob-range">
                      ({(cls.probabilityLower * 100).toFixed(0)}-{(cls.probabilityUpper * 100).toFixed(0)}%)
                    </span>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="classification-chart">
              <h4>Probability Distribution</h4>
              <div className="probability-bars">
                {data.planetClasses.map(cls => (
                  <div key={cls.type} className="prob-bar-item">
                    <span className="prob-icon">{cls.icon}</span>
                    <div className="prob-bar-track">
                      <div
                        className="prob-bar-uncertainty"
                        style={{
                          left: `${cls.probabilityLower * 100}%`,
                          width: `${(cls.probabilityUpper - cls.probabilityLower) * 100}%`
                        }}
                      />
                      <div
                        className="prob-bar-fill"
                        style={{ width: `${cls.probability * 100}%` }}
                      />
                    </div>
                    <span className="prob-label">{cls.type}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {/* Habitability Section */}
        {activeSection === 'habitability' && (
          <div className="habitability-section">
            <div className="habitability-main">
              <CircularGauge
                value={data.habitability.overall}
                uncertainty={data.habitability.overallUncertainty}
                size={220}
                label="Habitability Index"
              />
              
              <div className="habitability-interpretation">
                {data.habitability.overall >= 0.7 ? (
                  <p className="interpretation high">
                    <span className="icon">🌱</span>
                    <strong>Potentially Habitable</strong> - This exoplanet shows promising conditions for life as we know it.
                  </p>
                ) : data.habitability.overall >= 0.4 ? (
                  <p className="interpretation medium">
                    <span className="icon">⚠️</span>
                    <strong>Marginal Habitability</strong> - Some conditions may support extremophile life forms.
                  </p>
                ) : (
                  <p className="interpretation low">
                    <span className="icon">❌</span>
                    <strong>Likely Uninhabitable</strong> - Current conditions appear hostile to known life.
                  </p>
                )}
              </div>
            </div>
            
            <div className="habitability-breakdown">
              <h4>Contributing Factors</h4>
              <MetricBar
                label={data.habitability.temperature.label}
                value={data.habitability.temperature.value}
                uncertainty={data.habitability.temperature.uncertainty}
              />
              <MetricBar
                label={data.habitability.atmosphere.label}
                value={data.habitability.atmosphere.value}
                uncertainty={data.habitability.atmosphere.uncertainty}
              />
              <MetricBar
                label={data.habitability.water.label}
                value={data.habitability.water.value}
                uncertainty={data.habitability.water.uncertainty}
              />
              <MetricBar
                label={data.habitability.stability.label}
                value={data.habitability.stability.value}
                uncertainty={data.habitability.stability.uncertainty}
              />
            </div>
          </div>
        )}
        
        {/* Uncertainty Section */}
        {activeSection === 'uncertainty' && (
          <div className="uncertainty-section">
            <div className="uncertainty-overview">
              <h4>Uncertainty Budget</h4>
              <p className="uncertainty-description">
                Breakdown of factors contributing to measurement and model uncertainty.
                Understanding these sources helps assess the reliability of the analysis.
              </p>
              <UncertaintyChart sources={data.uncertaintySources} />
            </div>
            
            <div className="error-propagation">
              <h4>Error Propagation Summary</h4>
              <div className="propagation-grid">
                <div className="propagation-item">
                  <span className="prop-label">Total Statistical</span>
                  <span className="prop-value">
                    {(data.uncertaintySources
                      .filter(s => s.type === 'statistical')
                      .reduce((sum, s) => sum + s.contribution, 0) * 100
                    ).toFixed(0)}%
                  </span>
                </div>
                <div className="propagation-item">
                  <span className="prop-label">Total Systematic</span>
                  <span className="prop-value">
                    {(data.uncertaintySources
                      .filter(s => s.type === 'systematic')
                      .reduce((sum, s) => sum + s.contribution, 0) * 100
                    ).toFixed(0)}%
                  </span>
                </div>
                <div className="propagation-item">
                  <span className="prop-label">Model Uncertainty</span>
                  <span className="prop-value">
                    {(data.uncertaintySources
                      .filter(s => s.type === 'model')
                      .reduce((sum, s) => sum + s.contribution, 0) * 100
                    ).toFixed(0)}%
                  </span>
                </div>
                <div className="propagation-item total">
                  <span className="prop-label">Combined (RSS)</span>
                  <span className="prop-value">
                    {(Math.sqrt(data.uncertaintySources
                      .reduce((sum, s) => sum + s.contribution ** 2, 0)
                    ) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
            
            <div className="confidence-intervals">
              <h4>Confidence Interval Guide</h4>
              <div className="ci-explanation">
                <div className="ci-item">
                  <span className="ci-visual narrow" />
                  <div className="ci-text">
                    <strong>Narrow interval</strong>
                    <span>High precision, reliable measurement</span>
                  </div>
                </div>
                <div className="ci-item">
                  <span className="ci-visual medium" />
                  <div className="ci-text">
                    <strong>Medium interval</strong>
                    <span>Moderate uncertainty, use with caution</span>
                  </div>
                </div>
                <div className="ci-item">
                  <span className="ci-visual wide" />
                  <div className="ci-text">
                    <strong>Wide interval</strong>
                    <span>High uncertainty, needs more data</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Footer */}
      <div className="panel-footer">
        <span className="analysis-id">Analysis ID: {data.id}</span>
        <span className="timestamp">
          {new Date(data.timestamp).toLocaleString()}
        </span>
      </div>
    </div>
  );
};

export default ResultsPanel;
