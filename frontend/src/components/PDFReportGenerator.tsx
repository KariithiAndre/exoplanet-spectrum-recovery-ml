/**
 * PDF Report Generator
 * 
 * Generates publication-quality scientific reports including:
 * - Spectrum plots (embedded as images)
 * - Detected molecules table
 * - Model predictions and confidence
 * - Scientific explanation text
 * - Metadata and observation details
 */

import React, { useState, useCallback, useRef } from 'react';
import './PDFReportGenerator.css';

// ===== Type Definitions =====
interface SpectrumData {
  wavelengths: number[];
  flux: number[];
  uncertainties?: number[];
  processedFlux?: number[];
}

interface MoleculeDetection {
  formula: string;
  name: string;
  detected: boolean;
  confidence: number;
  significance: number;
  abundance?: number;
  wavelengthRange: [number, number];
  biomarker: boolean;
}

interface ModelPrediction {
  planetClass: string;
  classProbability: number;
  habitabilityIndex: number;
  habitabilityFactors: {
    temperature: { value: number; score: number };
    atmosphere: { value: string; score: number };
    water: { value: boolean; score: number };
    radiation: { value: string; score: number };
  };
  surfaceTemperature: number;
  atmosphericPressure: number;
  modelConfidence: number;
  uncertaintyBudget: {
    statistical: number;
    systematic: number;
    model: number;
  };
}

interface ObservationMetadata {
  targetName: string;
  targetRA: string;
  targetDec: string;
  telescope: string;
  instrument: string;
  observationDate: string;
  exposureTime: number;
  snr: number;
  resolution: number;
  observer?: string;
  programId?: string;
}

interface ReportConfig {
  title: string;
  author: string;
  institution: string;
  includeRawSpectrum: boolean;
  includeProcessedSpectrum: boolean;
  includeMoleculeTable: boolean;
  includePredictions: boolean;
  includeExplanation: boolean;
  includeMetadata: boolean;
  includeUncertainty: boolean;
  colorScheme: 'color' | 'grayscale';
  pageSize: 'letter' | 'a4';
}

interface ReportData {
  spectrum: SpectrumData;
  molecules: MoleculeDetection[];
  predictions: ModelPrediction;
  metadata: ObservationMetadata;
  analysisNotes?: string;
}

// ===== Demo Data Generator =====
const generateDemoData = (): ReportData => {
  const numPoints = 500;
  const wavelengths: number[] = [];
  const flux: number[] = [];
  const uncertainties: number[] = [];
  const processedFlux: number[] = [];

  for (let i = 0; i < numPoints; i++) {
    const wl = 0.6 + (27.4 * i / (numPoints - 1));
    wavelengths.push(wl);
    
    let f = Math.exp(-0.2 * Math.pow((wl - 8) / 15, 2));
    
    // Add molecular features
    const features = [
      { center: 1.4, width: 0.12, depth: 0.25 },
      { center: 2.7, width: 0.25, depth: 0.35 },
      { center: 4.3, width: 0.2, depth: 0.45 },
      { center: 6.3, width: 0.4, depth: 0.3 },
      { center: 9.6, width: 0.5, depth: 0.2 },
      { center: 15.0, width: 1.2, depth: 0.25 }
    ];
    
    features.forEach(feat => {
      const dist = Math.abs(wl - feat.center) / feat.width;
      if (dist < 3) {
        f *= (1 - feat.depth * Math.exp(-0.5 * dist * dist));
      }
    });
    
    const noise = (Math.random() - 0.5) * 0.04 * f;
    flux.push(f + noise);
    uncertainties.push(0.02 * f);
    processedFlux.push(f);
  }

  return {
    spectrum: { wavelengths, flux, uncertainties, processedFlux },
    molecules: [
      { formula: 'H₂O', name: 'Water', detected: true, confidence: 0.94, significance: 8.2, abundance: 1.2e-3, wavelengthRange: [1.3, 1.5], biomarker: true },
      { formula: 'CO₂', name: 'Carbon Dioxide', detected: true, confidence: 0.89, significance: 6.5, abundance: 4.5e-4, wavelengthRange: [4.2, 4.4], biomarker: false },
      { formula: 'CH₄', name: 'Methane', detected: true, confidence: 0.76, significance: 4.1, abundance: 2.1e-5, wavelengthRange: [2.3, 2.4], biomarker: true },
      { formula: 'O₃', name: 'Ozone', detected: true, confidence: 0.82, significance: 5.3, abundance: 8.3e-6, wavelengthRange: [9.4, 9.8], biomarker: true },
      { formula: 'O₂', name: 'Oxygen', detected: false, confidence: 0.35, significance: 1.8, wavelengthRange: [0.76, 0.77], biomarker: true },
      { formula: 'NH₃', name: 'Ammonia', detected: false, confidence: 0.22, significance: 1.1, wavelengthRange: [10.3, 10.8], biomarker: false },
      { formula: 'CO', name: 'Carbon Monoxide', detected: true, confidence: 0.68, significance: 3.2, abundance: 1.5e-5, wavelengthRange: [4.6, 4.7], biomarker: false },
      { formula: 'N₂O', name: 'Nitrous Oxide', detected: false, confidence: 0.28, significance: 1.4, wavelengthRange: [4.4, 4.6], biomarker: true }
    ],
    predictions: {
      planetClass: 'Super-Earth',
      classProbability: 0.87,
      habitabilityIndex: 0.73,
      habitabilityFactors: {
        temperature: { value: 285, score: 0.82 },
        atmosphere: { value: 'N₂/CO₂ dominant', score: 0.68 },
        water: { value: true, score: 0.91 },
        radiation: { value: 'Moderate', score: 0.65 }
      },
      surfaceTemperature: 285,
      atmosphericPressure: 1.2,
      modelConfidence: 0.81,
      uncertaintyBudget: {
        statistical: 0.12,
        systematic: 0.18,
        model: 0.25
      }
    },
    metadata: {
      targetName: 'TRAPPIST-1e',
      targetRA: '23h 06m 29.37s',
      targetDec: '-05° 02′ 29.0″',
      telescope: 'James Webb Space Telescope',
      instrument: 'NIRSpec G395H',
      observationDate: '2025-09-15',
      exposureTime: 28800,
      snr: 127,
      resolution: 2700,
      observer: 'Exoplanet Survey Team',
      programId: 'JWST-GO-3842'
    },
    analysisNotes: 'High-quality spectrum with clear molecular signatures. Water vapor detection is robust with high confidence. Methane detection requires further validation with additional observations.'
  };
};

// ===== Scientific Explanation Generator =====
const generateScientificExplanation = (data: ReportData): string => {
  const detectedMolecules = data.molecules.filter(m => m.detected);
  const biomarkers = detectedMolecules.filter(m => m.biomarker);
  const hab = data.predictions.habitabilityIndex;
  
  let explanation = `## Scientific Analysis Summary\n\n`;
  
  explanation += `### Spectroscopic Analysis\n\n`;
  explanation += `The transmission spectrum of ${data.metadata.targetName} was obtained using the ${data.metadata.instrument} `;
  explanation += `instrument aboard the ${data.metadata.telescope} with a total exposure time of ${(data.metadata.exposureTime / 3600).toFixed(1)} hours. `;
  explanation += `The observations achieved a signal-to-noise ratio of ${data.metadata.snr} at a spectral resolution of R=${data.metadata.resolution}.\n\n`;
  
  explanation += `### Molecular Detections\n\n`;
  explanation += `A total of ${detectedMolecules.length} molecular species were detected in the planetary atmosphere:\n\n`;
  
  detectedMolecules.forEach(mol => {
    explanation += `- **${mol.name} (${mol.formula})**: Detected with ${(mol.confidence * 100).toFixed(0)}% confidence `;
    explanation += `(${mol.significance.toFixed(1)}σ significance)`;
    if (mol.abundance) {
      explanation += `, estimated mixing ratio: ${mol.abundance.toExponential(1)}`;
    }
    explanation += `\n`;
  });
  
  if (biomarkers.length > 0) {
    explanation += `\n### Biosignature Assessment\n\n`;
    explanation += `Of the detected species, ${biomarkers.length} are considered potential biosignature gases: `;
    explanation += biomarkers.map(m => m.formula).join(', ') + '. ';
    
    if (biomarkers.some(m => m.formula === 'H₂O') && biomarkers.some(m => m.formula === 'CH₄')) {
      explanation += `The simultaneous detection of water and methane is particularly noteworthy, as this combination `;
      explanation += `may indicate active geological or biological processes. `;
    }
    
    if (biomarkers.some(m => m.formula === 'O₃')) {
      explanation += `The presence of ozone suggests significant oxygen content in the atmosphere, `;
      explanation += `which could be indicative of oxygenic photosynthesis. `;
    }
  }
  
  explanation += `\n\n### Planetary Classification\n\n`;
  explanation += `Based on the spectroscopic analysis and atmospheric modeling, ${data.metadata.targetName} is classified as a `;
  explanation += `**${data.predictions.planetClass}** with ${(data.predictions.classProbability * 100).toFixed(0)}% probability. `;
  explanation += `The estimated surface temperature is ${data.predictions.surfaceTemperature} K `;
  explanation += `with an atmospheric pressure of ${data.predictions.atmosphericPressure.toFixed(1)} bar.\n\n`;
  
  explanation += `### Habitability Assessment\n\n`;
  explanation += `The composite habitability index for ${data.metadata.targetName} is **${(hab * 100).toFixed(0)}%**, `;
  
  if (hab > 0.7) {
    explanation += `indicating a potentially habitable world with conditions conducive to liquid water. `;
  } else if (hab > 0.5) {
    explanation += `suggesting moderate habitability potential that warrants further investigation. `;
  } else {
    explanation += `indicating challenging conditions for Earth-like life. `;
  }
  
  explanation += `This assessment is based on four key factors:\n\n`;
  explanation += `1. **Temperature**: ${data.predictions.habitabilityFactors.temperature.value} K `;
  explanation += `(Score: ${(data.predictions.habitabilityFactors.temperature.score * 100).toFixed(0)}%)\n`;
  explanation += `2. **Atmosphere**: ${data.predictions.habitabilityFactors.atmosphere.value} `;
  explanation += `(Score: ${(data.predictions.habitabilityFactors.atmosphere.score * 100).toFixed(0)}%)\n`;
  explanation += `3. **Water Presence**: ${data.predictions.habitabilityFactors.water.value ? 'Detected' : 'Not detected'} `;
  explanation += `(Score: ${(data.predictions.habitabilityFactors.water.score * 100).toFixed(0)}%)\n`;
  explanation += `4. **Radiation Environment**: ${data.predictions.habitabilityFactors.radiation.value} `;
  explanation += `(Score: ${(data.predictions.habitabilityFactors.radiation.score * 100).toFixed(0)}%)\n`;
  
  explanation += `\n### Uncertainty Analysis\n\n`;
  explanation += `The overall model confidence is ${(data.predictions.modelConfidence * 100).toFixed(0)}%. `;
  explanation += `The uncertainty budget is distributed as follows:\n\n`;
  explanation += `- Statistical uncertainty: ${(data.predictions.uncertaintyBudget.statistical * 100).toFixed(0)}%\n`;
  explanation += `- Systematic uncertainty: ${(data.predictions.uncertaintyBudget.systematic * 100).toFixed(0)}%\n`;
  explanation += `- Model uncertainty: ${(data.predictions.uncertaintyBudget.model * 100).toFixed(0)}%\n`;
  
  if (data.analysisNotes) {
    explanation += `\n### Additional Notes\n\n${data.analysisNotes}\n`;
  }
  
  return explanation;
};

// ===== PDF Generation (using canvas for plots) =====
const generatePDFContent = async (
  data: ReportData, 
  config: ReportConfig,
  plotCanvas: HTMLCanvasElement | null
): Promise<void> => {
  // Dynamic import of jsPDF
  const { jsPDF } = await import('jspdf');
  
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: config.pageSize
  });
  
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 20;
  const contentWidth = pageWidth - 2 * margin;
  let y = margin;
  
  const colors = {
    primary: config.colorScheme === 'color' ? [255, 215, 0] : [60, 60, 60],
    text: [40, 40, 40],
    secondary: [100, 100, 100],
    accent: config.colorScheme === 'color' ? [0, 180, 220] : [80, 80, 80],
    success: config.colorScheme === 'color' ? [0, 200, 100] : [60, 60, 60],
    warning: config.colorScheme === 'color' ? [255, 150, 0] : [100, 100, 100]
  };
  
  // Helper: Add new page if needed
  const checkNewPage = (neededHeight: number) => {
    if (y + neededHeight > pageHeight - margin) {
      doc.addPage();
      y = margin;
      return true;
    }
    return false;
  };
  
  // Helper: Draw section header
  const drawSectionHeader = (title: string) => {
    checkNewPage(15);
    doc.setFillColor(...colors.primary as [number, number, number]);
    doc.rect(margin, y, contentWidth, 8, 'F');
    doc.setTextColor(40, 40, 40);
    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.text(title, margin + 3, y + 5.5);
    y += 12;
  };
  
  // ===== TITLE PAGE =====
  doc.setFillColor(...colors.primary as [number, number, number]);
  doc.rect(0, 0, pageWidth, 60, 'F');
  
  doc.setTextColor(40, 40, 40);
  doc.setFontSize(24);
  doc.setFont('helvetica', 'bold');
  doc.text(config.title || 'Exoplanet Spectrum Analysis Report', pageWidth / 2, 30, { align: 'center' });
  
  doc.setFontSize(14);
  doc.setFont('helvetica', 'normal');
  doc.text(data.metadata.targetName, pageWidth / 2, 45, { align: 'center' });
  
  y = 75;
  
  // Author info
  doc.setTextColor(...colors.text as [number, number, number]);
  doc.setFontSize(12);
  doc.text(`Author: ${config.author}`, margin, y);
  y += 7;
  doc.text(`Institution: ${config.institution}`, margin, y);
  y += 7;
  doc.text(`Date: ${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}`, margin, y);
  y += 15;
  
  // ===== OBSERVATION METADATA =====
  if (config.includeMetadata) {
    drawSectionHeader('Observation Details');
    
    const metaItems = [
      ['Target', data.metadata.targetName],
      ['Coordinates', `RA: ${data.metadata.targetRA}, Dec: ${data.metadata.targetDec}`],
      ['Telescope', data.metadata.telescope],
      ['Instrument', data.metadata.instrument],
      ['Observation Date', data.metadata.observationDate],
      ['Exposure Time', `${(data.metadata.exposureTime / 3600).toFixed(2)} hours`],
      ['Signal-to-Noise', data.metadata.snr.toString()],
      ['Spectral Resolution', `R = ${data.metadata.resolution}`]
    ];
    
    if (data.metadata.programId) {
      metaItems.push(['Program ID', data.metadata.programId]);
    }
    
    doc.setFontSize(10);
    metaItems.forEach(([label, value]) => {
      checkNewPage(6);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(...colors.secondary as [number, number, number]);
      doc.text(`${label}:`, margin, y);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...colors.text as [number, number, number]);
      doc.text(value, margin + 45, y);
      y += 6;
    });
    y += 5;
  }
  
  // ===== SPECTRUM PLOT =====
  if ((config.includeRawSpectrum || config.includeProcessedSpectrum) && plotCanvas) {
    checkNewPage(80);
    drawSectionHeader('Transmission Spectrum');
    
    try {
      const imgData = plotCanvas.toDataURL('image/png');
      const imgWidth = contentWidth;
      const imgHeight = 60;
      doc.addImage(imgData, 'PNG', margin, y, imgWidth, imgHeight);
      y += imgHeight + 5;
      
      doc.setFontSize(9);
      doc.setTextColor(...colors.secondary as [number, number, number]);
      doc.setFont('helvetica', 'italic');
      doc.text('Figure 1: Transmission spectrum showing normalized flux as a function of wavelength.', margin, y);
      y += 10;
    } catch (e) {
      doc.setFontSize(10);
      doc.text('[Spectrum plot could not be embedded]', margin, y);
      y += 10;
    }
  }
  
  // ===== MOLECULE DETECTION TABLE =====
  if (config.includeMoleculeTable) {
    checkNewPage(60);
    drawSectionHeader('Detected Molecules');
    
    const detectedMols = data.molecules.filter(m => m.detected);
    const nonDetectedMols = data.molecules.filter(m => !m.detected);
    
    // Table header
    const colWidths = [25, 35, 25, 25, 35];
    const headers = ['Formula', 'Name', 'Confidence', 'Significance', 'Abundance'];
    
    doc.setFillColor(240, 240, 240);
    doc.rect(margin, y, contentWidth, 7, 'F');
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...colors.text as [number, number, number]);
    
    let xPos = margin + 2;
    headers.forEach((header, i) => {
      doc.text(header, xPos, y + 5);
      xPos += colWidths[i];
    });
    y += 8;
    
    // Detected molecules
    doc.setFont('helvetica', 'normal');
    detectedMols.forEach(mol => {
      checkNewPage(7);
      xPos = margin + 2;
      
      doc.setTextColor(...colors.success as [number, number, number]);
      doc.text(mol.formula, xPos, y + 4);
      xPos += colWidths[0];
      
      doc.setTextColor(...colors.text as [number, number, number]);
      doc.text(mol.name, xPos, y + 4);
      xPos += colWidths[1];
      
      doc.text(`${(mol.confidence * 100).toFixed(0)}%`, xPos, y + 4);
      xPos += colWidths[2];
      
      doc.text(`${mol.significance.toFixed(1)}σ`, xPos, y + 4);
      xPos += colWidths[3];
      
      doc.text(mol.abundance ? mol.abundance.toExponential(1) : '-', xPos, y + 4);
      
      // Row separator
      doc.setDrawColor(220, 220, 220);
      doc.line(margin, y + 6, margin + contentWidth, y + 6);
      y += 7;
    });
    
    // Non-detected summary
    if (nonDetectedMols.length > 0) {
      y += 3;
      doc.setFontSize(9);
      doc.setTextColor(...colors.secondary as [number, number, number]);
      doc.text(`Non-detections: ${nonDetectedMols.map(m => m.formula).join(', ')}`, margin, y);
      y += 8;
    }
    
    // Biomarker note
    const biomarkers = detectedMols.filter(m => m.biomarker);
    if (biomarkers.length > 0) {
      doc.setFontSize(9);
      doc.setFont('helvetica', 'italic');
      doc.setTextColor(...colors.accent as [number, number, number]);
      doc.text(`Potential biosignatures detected: ${biomarkers.map(m => m.formula).join(', ')}`, margin, y);
      y += 10;
    }
  }
  
  // ===== MODEL PREDICTIONS =====
  if (config.includePredictions) {
    checkNewPage(50);
    drawSectionHeader('Model Predictions');
    
    doc.setFontSize(10);
    
    // Planet classification
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...colors.text as [number, number, number]);
    doc.text('Planetary Classification:', margin, y);
    doc.setFont('helvetica', 'normal');
    doc.text(`${data.predictions.planetClass} (${(data.predictions.classProbability * 100).toFixed(0)}% probability)`, margin + 50, y);
    y += 8;
    
    // Habitability
    doc.setFont('helvetica', 'bold');
    doc.text('Habitability Index:', margin, y);
    doc.setFont('helvetica', 'normal');
    const habColor = data.predictions.habitabilityIndex > 0.7 ? colors.success : 
                     data.predictions.habitabilityIndex > 0.5 ? colors.warning : colors.secondary;
    doc.setTextColor(...habColor as [number, number, number]);
    doc.text(`${(data.predictions.habitabilityIndex * 100).toFixed(0)}%`, margin + 50, y);
    y += 8;
    
    // Physical parameters
    doc.setTextColor(...colors.text as [number, number, number]);
    doc.setFont('helvetica', 'bold');
    doc.text('Surface Temperature:', margin, y);
    doc.setFont('helvetica', 'normal');
    doc.text(`${data.predictions.surfaceTemperature} K`, margin + 50, y);
    y += 6;
    
    doc.setFont('helvetica', 'bold');
    doc.text('Atmospheric Pressure:', margin, y);
    doc.setFont('helvetica', 'normal');
    doc.text(`${data.predictions.atmosphericPressure.toFixed(2)} bar`, margin + 50, y);
    y += 6;
    
    doc.setFont('helvetica', 'bold');
    doc.text('Model Confidence:', margin, y);
    doc.setFont('helvetica', 'normal');
    doc.text(`${(data.predictions.modelConfidence * 100).toFixed(0)}%`, margin + 50, y);
    y += 10;
    
    // Habitability factors table
    if (config.includeUncertainty) {
      doc.setFontSize(9);
      doc.setFont('helvetica', 'bold');
      doc.text('Habitability Factor Breakdown:', margin, y);
      y += 6;
      
      const factors = [
        ['Temperature', `${data.predictions.habitabilityFactors.temperature.value} K`, data.predictions.habitabilityFactors.temperature.score],
        ['Atmosphere', data.predictions.habitabilityFactors.atmosphere.value, data.predictions.habitabilityFactors.atmosphere.score],
        ['Water', data.predictions.habitabilityFactors.water.value ? 'Detected' : 'Not detected', data.predictions.habitabilityFactors.water.score],
        ['Radiation', data.predictions.habitabilityFactors.radiation.value, data.predictions.habitabilityFactors.radiation.score]
      ];
      
      doc.setFont('helvetica', 'normal');
      factors.forEach(([name, value, score]) => {
        checkNewPage(6);
        doc.setTextColor(...colors.secondary as [number, number, number]);
        doc.text(`  • ${name}:`, margin, y);
        doc.setTextColor(...colors.text as [number, number, number]);
        doc.text(`${value} (Score: ${((score as number) * 100).toFixed(0)}%)`, margin + 35, y);
        y += 5;
      });
      y += 5;
      
      // Uncertainty budget
      doc.setFont('helvetica', 'bold');
      doc.text('Uncertainty Budget:', margin, y);
      y += 6;
      doc.setFont('helvetica', 'normal');
      doc.text(`  • Statistical: ${(data.predictions.uncertaintyBudget.statistical * 100).toFixed(0)}%`, margin, y);
      y += 5;
      doc.text(`  • Systematic: ${(data.predictions.uncertaintyBudget.systematic * 100).toFixed(0)}%`, margin, y);
      y += 5;
      doc.text(`  • Model: ${(data.predictions.uncertaintyBudget.model * 100).toFixed(0)}%`, margin, y);
      y += 10;
    }
  }
  
  // ===== SCIENTIFIC EXPLANATION =====
  if (config.includeExplanation) {
    const explanation = generateScientificExplanation(data);
    const lines = explanation.split('\n');
    
    doc.addPage();
    y = margin;
    
    lines.forEach(line => {
      if (line.startsWith('## ')) {
        checkNewPage(15);
        doc.setFillColor(...colors.primary as [number, number, number]);
        doc.rect(margin, y, contentWidth, 10, 'F');
        doc.setTextColor(40, 40, 40);
        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.text(line.replace('## ', ''), margin + 3, y + 7);
        y += 15;
      } else if (line.startsWith('### ')) {
        checkNewPage(12);
        doc.setTextColor(...colors.accent as [number, number, number]);
        doc.setFontSize(11);
        doc.setFont('helvetica', 'bold');
        doc.text(line.replace('### ', ''), margin, y);
        y += 8;
      } else if (line.startsWith('- **')) {
        checkNewPage(6);
        const match = line.match(/- \*\*(.+?)\*\*:(.+)/);
        if (match) {
          doc.setFontSize(10);
          doc.setFont('helvetica', 'bold');
          doc.setTextColor(...colors.text as [number, number, number]);
          doc.text(`• ${match[1]}:`, margin + 3, y);
          doc.setFont('helvetica', 'normal');
          const restText = doc.splitTextToSize(match[2].trim(), contentWidth - 45);
          doc.text(restText, margin + 40, y);
          y += 5 * restText.length + 2;
        }
      } else if (line.match(/^\d+\. \*\*/)) {
        checkNewPage(6);
        const match = line.match(/^(\d+)\. \*\*(.+?)\*\*:(.+)/);
        if (match) {
          doc.setFontSize(10);
          doc.setFont('helvetica', 'bold');
          doc.setTextColor(...colors.text as [number, number, number]);
          doc.text(`${match[1]}. ${match[2]}:`, margin + 3, y);
          doc.setFont('helvetica', 'normal');
          doc.text(match[3].trim(), margin + 50, y);
          y += 6;
        }
      } else if (line.trim()) {
        checkNewPage(6);
        doc.setFontSize(10);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(...colors.text as [number, number, number]);
        const splitText = doc.splitTextToSize(line, contentWidth);
        doc.text(splitText, margin, y);
        y += 5 * splitText.length;
      } else {
        y += 3;
      }
    });
  }
  
  // ===== FOOTER ON ALL PAGES =====
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text(
      `Generated by Exoplanet Spectrum Analysis System | Page ${i} of ${totalPages}`,
      pageWidth / 2,
      pageHeight - 10,
      { align: 'center' }
    );
  }
  
  // Save the PDF
  const filename = `${data.metadata.targetName.replace(/\s+/g, '_')}_Analysis_Report.pdf`;
  doc.save(filename);
};

// ===== Plot Renderer Component =====
interface PlotRendererProps {
  data: ReportData;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  colorScheme: 'color' | 'grayscale';
}

const PlotRenderer: React.FC<PlotRendererProps> = ({ data, canvasRef, colorScheme }) => {
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 30, right: 30, bottom: 50, left: 60 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    
    // Clear
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    
    // Data ranges
    const wlMin = Math.min(...data.spectrum.wavelengths);
    const wlMax = Math.max(...data.spectrum.wavelengths);
    const fluxMin = Math.min(...data.spectrum.flux) * 0.9;
    const fluxMax = Math.max(...data.spectrum.flux) * 1.1;
    
    // Grid
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (plotHeight * i / 5);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      
      const x = padding.left + (plotWidth * i / 5);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
    }
    
    // Axes
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();
    
    // Axis labels
    ctx.fillStyle = '#333333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Wavelength (μm)', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Normalized Flux', 0, 0);
    ctx.restore();
    
    // Tick labels
    ctx.font = '10px Arial';
    for (let i = 0; i <= 5; i++) {
      const wl = wlMin + (wlMax - wlMin) * i / 5;
      const x = padding.left + (plotWidth * i / 5);
      ctx.fillText(wl.toFixed(1), x, height - padding.bottom + 15);
      
      const flux = fluxMax - (fluxMax - fluxMin) * i / 5;
      const y = padding.top + (plotHeight * i / 5);
      ctx.textAlign = 'right';
      ctx.fillText(flux.toFixed(2), padding.left - 5, y + 3);
      ctx.textAlign = 'center';
    }
    
    // Transform functions
    const toX = (wl: number) => padding.left + ((wl - wlMin) / (wlMax - wlMin)) * plotWidth;
    const toY = (flux: number) => padding.top + ((fluxMax - flux) / (fluxMax - fluxMin)) * plotHeight;
    
    // Uncertainty band
    if (data.spectrum.uncertainties) {
      ctx.fillStyle = colorScheme === 'color' ? 'rgba(0, 150, 200, 0.2)' : 'rgba(100, 100, 100, 0.2)';
      ctx.beginPath();
      
      for (let i = 0; i < data.spectrum.wavelengths.length; i++) {
        const x = toX(data.spectrum.wavelengths[i]);
        const y = toY(data.spectrum.flux[i] + (data.spectrum.uncertainties[i] || 0));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      
      for (let i = data.spectrum.wavelengths.length - 1; i >= 0; i--) {
        const x = toX(data.spectrum.wavelengths[i]);
        const y = toY(data.spectrum.flux[i] - (data.spectrum.uncertainties[i] || 0));
        ctx.lineTo(x, y);
      }
      
      ctx.closePath();
      ctx.fill();
    }
    
    // Spectrum line
    ctx.strokeStyle = colorScheme === 'color' ? '#0088cc' : '#333333';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    
    for (let i = 0; i < data.spectrum.wavelengths.length; i++) {
      const x = toX(data.spectrum.wavelengths[i]);
      const y = toY(data.spectrum.flux[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // Molecule markers
    const moleculeColors = colorScheme === 'color' 
      ? ['#ff6b6b', '#ffd700', '#00ff88', '#00d4ff', '#9b59b6']
      : ['#333', '#555', '#777', '#999', '#bbb'];
    
    data.molecules.filter(m => m.detected).forEach((mol, idx) => {
      const [wl1, wl2] = mol.wavelengthRange;
      if (wl1 >= wlMin && wl2 <= wlMax) {
        const x1 = toX(wl1);
        const x2 = toX(wl2);
        
        ctx.fillStyle = `${moleculeColors[idx % 5]}40`;
        ctx.fillRect(x1, padding.top, x2 - x1, plotHeight);
        
        ctx.fillStyle = moleculeColors[idx % 5];
        ctx.font = 'bold 9px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(mol.formula, (x1 + x2) / 2, padding.top - 5);
      }
    });
    
    // Title
    ctx.fillStyle = '#000000';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`${data.metadata.targetName} - Transmission Spectrum`, width / 2, 18);
    
  }, [data, colorScheme]);
  
  return (
    <canvas 
      ref={canvasRef} 
      width={800} 
      height={400} 
      className="spectrum-preview-canvas"
    />
  );
};

// ===== Main Component =====
const PDFReportGenerator: React.FC<{
  data?: ReportData;
  onClose?: () => void;
}> = ({ data: externalData, onClose }) => {
  const [reportData] = useState<ReportData>(externalData || generateDemoData);
  const [isGenerating, setIsGenerating] = useState(false);
  const [config, setConfig] = useState<ReportConfig>({
    title: 'Exoplanet Spectrum Analysis Report',
    author: 'Research Team',
    institution: 'Exoplanet Research Institute',
    includeRawSpectrum: true,
    includeProcessedSpectrum: true,
    includeMoleculeTable: true,
    includePredictions: true,
    includeExplanation: true,
    includeMetadata: true,
    includeUncertainty: true,
    colorScheme: 'color',
    pageSize: 'letter'
  });
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const handleGenerate = useCallback(async () => {
    setIsGenerating(true);
    try {
      await generatePDFContent(reportData, config, canvasRef.current);
    } catch (error) {
      console.error('PDF generation failed:', error);
      alert('Failed to generate PDF. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  }, [reportData, config]);
  
  const updateConfig = (key: keyof ReportConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };
  
  return (
    <div className="pdf-generator">
      <header className="pdf-header">
        <h2>📄 PDF Report Generator</h2>
        {onClose && (
          <button className="close-btn" onClick={onClose}>×</button>
        )}
      </header>
      
      <div className="pdf-content">
        {/* Preview Section */}
        <section className="preview-section">
          <h3>Report Preview</h3>
          <div className="preview-container">
            <div className="preview-page">
              <div className="preview-title-bar">
                <span className="preview-title">{config.title}</span>
                <span className="preview-target">{reportData.metadata.targetName}</span>
              </div>
              
              {config.includeMetadata && (
                <div className="preview-metadata">
                  <span>📡 {reportData.metadata.telescope}</span>
                  <span>🔬 {reportData.metadata.instrument}</span>
                  <span>📅 {reportData.metadata.observationDate}</span>
                </div>
              )}
              
              {(config.includeRawSpectrum || config.includeProcessedSpectrum) && (
                <div className="preview-plot">
                  <PlotRenderer 
                    data={reportData} 
                    canvasRef={canvasRef}
                    colorScheme={config.colorScheme}
                  />
                </div>
              )}
              
              {config.includeMoleculeTable && (
                <div className="preview-molecules">
                  <h4>Detected Molecules</h4>
                  <div className="molecule-chips">
                    {reportData.molecules.filter(m => m.detected).map(mol => (
                      <span key={mol.formula} className="molecule-chip">
                        {mol.formula} ({(mol.confidence * 100).toFixed(0)}%)
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {config.includePredictions && (
                <div className="preview-predictions">
                  <div className="prediction-item">
                    <span className="prediction-label">Classification</span>
                    <span className="prediction-value">{reportData.predictions.planetClass}</span>
                  </div>
                  <div className="prediction-item">
                    <span className="prediction-label">Habitability</span>
                    <span className="prediction-value habitability" style={{
                      color: reportData.predictions.habitabilityIndex > 0.7 ? '#00ff88' :
                             reportData.predictions.habitabilityIndex > 0.5 ? '#ffd700' : '#888'
                    }}>
                      {(reportData.predictions.habitabilityIndex * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
        
        {/* Configuration Section */}
        <section className="config-section">
          <h3>Report Configuration</h3>
          
          <div className="config-group">
            <label>Report Title</label>
            <input
              type="text"
              value={config.title}
              onChange={(e) => updateConfig('title', e.target.value)}
            />
          </div>
          
          <div className="config-row">
            <div className="config-group">
              <label>Author</label>
              <input
                type="text"
                value={config.author}
                onChange={(e) => updateConfig('author', e.target.value)}
              />
            </div>
            <div className="config-group">
              <label>Institution</label>
              <input
                type="text"
                value={config.institution}
                onChange={(e) => updateConfig('institution', e.target.value)}
              />
            </div>
          </div>
          
          <div className="config-group">
            <label>Include Sections</label>
            <div className="checkbox-grid">
              <label className="checkbox-item">
                <input
                  type="checkbox"
                  checked={config.includeMetadata}
                  onChange={(e) => updateConfig('includeMetadata', e.target.checked)}
                />
                Observation Metadata
              </label>
              <label className="checkbox-item">
                <input
                  type="checkbox"
                  checked={config.includeRawSpectrum}
                  onChange={(e) => updateConfig('includeRawSpectrum', e.target.checked)}
                />
                Spectrum Plot
              </label>
              <label className="checkbox-item">
                <input
                  type="checkbox"
                  checked={config.includeMoleculeTable}
                  onChange={(e) => updateConfig('includeMoleculeTable', e.target.checked)}
                />
                Molecule Detection Table
              </label>
              <label className="checkbox-item">
                <input
                  type="checkbox"
                  checked={config.includePredictions}
                  onChange={(e) => updateConfig('includePredictions', e.target.checked)}
                />
                Model Predictions
              </label>
              <label className="checkbox-item">
                <input
                  type="checkbox"
                  checked={config.includeUncertainty}
                  onChange={(e) => updateConfig('includeUncertainty', e.target.checked)}
                />
                Uncertainty Analysis
              </label>
              <label className="checkbox-item">
                <input
                  type="checkbox"
                  checked={config.includeExplanation}
                  onChange={(e) => updateConfig('includeExplanation', e.target.checked)}
                />
                Scientific Explanation
              </label>
            </div>
          </div>
          
          <div className="config-row">
            <div className="config-group">
              <label>Color Scheme</label>
              <div className="radio-group">
                <label>
                  <input
                    type="radio"
                    name="colorScheme"
                    checked={config.colorScheme === 'color'}
                    onChange={() => updateConfig('colorScheme', 'color')}
                  />
                  Color
                </label>
                <label>
                  <input
                    type="radio"
                    name="colorScheme"
                    checked={config.colorScheme === 'grayscale'}
                    onChange={() => updateConfig('colorScheme', 'grayscale')}
                  />
                  Grayscale (Print)
                </label>
              </div>
            </div>
            <div className="config-group">
              <label>Page Size</label>
              <div className="radio-group">
                <label>
                  <input
                    type="radio"
                    name="pageSize"
                    checked={config.pageSize === 'letter'}
                    onChange={() => updateConfig('pageSize', 'letter')}
                  />
                  US Letter
                </label>
                <label>
                  <input
                    type="radio"
                    name="pageSize"
                    checked={config.pageSize === 'a4'}
                    onChange={() => updateConfig('pageSize', 'a4')}
                  />
                  A4
                </label>
              </div>
            </div>
          </div>
          
          <button 
            className={`generate-btn ${isGenerating ? 'generating' : ''}`}
            onClick={handleGenerate}
            disabled={isGenerating}
          >
            {isGenerating ? (
              <>
                <span className="spinner"></span>
                Generating PDF...
              </>
            ) : (
              <>📥 Generate PDF Report</>
            )}
          </button>
        </section>
      </div>
    </div>
  );
};

export default PDFReportGenerator;
