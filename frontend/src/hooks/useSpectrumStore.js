import { create } from 'zustand'

export const useSpectrumStore = create((set) => ({
  // Current spectrum data
  spectrum: null,
  setSpectrum: (spectrum) => set({ spectrum }),
  
  // Analysis results
  analysisResults: null,
  setAnalysisResults: (results) => set({ analysisResults: results }),
  
  // Loading states
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),
  
  // Selected model
  selectedModel: 'spectrum_denoiser_v2',
  setSelectedModel: (model) => set({ selectedModel: model }),
  
  // View settings
  showErrorBars: true,
  toggleErrorBars: () => set((state) => ({ showErrorBars: !state.showErrorBars })),
  
  showModel: true,
  toggleModel: () => set((state) => ({ showModel: !state.showModel })),
  
  // Reset state
  reset: () => set({
    spectrum: null,
    analysisResults: null,
    isLoading: false,
  }),
}))
