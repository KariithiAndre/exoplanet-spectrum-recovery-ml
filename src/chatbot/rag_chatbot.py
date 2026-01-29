"""
RAG-based Spectrum Analysis Chatbot

Scientific explanation system using Retrieval-Augmented Generation:
- Astrophysics knowledge base
- Semantic search with embeddings
- Context-aware responses with scientific terminology
- Integration with analysis results
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import pickle

# Vector store and embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# LLM integration
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# Knowledge Base Documents
# ============================================================================

ASTROPHYSICS_KNOWLEDGE_BASE = [
    # Transmission Spectroscopy
    {
        "id": "transmission_spectroscopy_basics",
        "title": "Transmission Spectroscopy Fundamentals",
        "category": "methodology",
        "content": """
        Transmission spectroscopy is a powerful technique for characterizing exoplanet atmospheres. 
        During a planetary transit, starlight passes through the planet's atmospheric limb, and 
        atmospheric constituents absorb light at characteristic wavelengths. The transit depth 
        varies with wavelength according to the atmospheric opacity, creating a transmission spectrum.
        
        The effective planetary radius at wavelength λ is given by:
        R_eff(λ) = R_p + H × ln(τ(λ))
        
        where R_p is the reference radius, H is the atmospheric scale height, and τ is the optical depth.
        The scale height H = kT/(μg) depends on temperature T, mean molecular weight μ, and surface gravity g.
        
        Key considerations include:
        - Higher temperatures increase scale height, enhancing spectral features
        - Lower mean molecular weight (hydrogen-rich atmospheres) produces larger signals
        - Clouds and hazes can mute spectral features by increasing continuum opacity
        """
    },
    {
        "id": "molecular_spectroscopy",
        "title": "Molecular Absorption Features",
        "category": "spectroscopy",
        "content": """
        Atmospheric molecules produce characteristic absorption signatures through electronic, 
        vibrational, and rotational transitions. In the infrared, ro-vibrational bands dominate.
        
        Key molecular signatures in exoplanet atmospheres:
        
        Water (H₂O): Strong bands at 1.4, 1.9, 2.7, and 6.3 μm. The 2.7 μm band is particularly 
        diagnostic as it's less affected by telluric contamination in ground-based observations.
        
        Carbon Dioxide (CO₂): Prominent features at 4.3 μm (asymmetric stretch) and 15 μm 
        (bending mode). The 4.3 μm band is a key diagnostic for terrestrial atmospheres.
        
        Methane (CH₄): Absorption bands at 1.7, 2.3, 3.3, and 7.7 μm. Detection alongside 
        water may indicate disequilibrium chemistry or biological activity.
        
        Ozone (O₃): Strong 9.6 μm band. Ozone is a photochemical product of molecular oxygen 
        and considered a potential biosignature when found in conjunction with reducing gases.
        
        Oxygen (O₂): Weak features at 0.76 μm (A-band) and 1.27 μm. Difficult to detect but 
        highly significant as a potential biosignature.
        """
    },
    {
        "id": "biosignatures",
        "title": "Biosignature Gases and Habitability",
        "category": "astrobiology",
        "content": """
        Biosignature gases are atmospheric constituents that may indicate biological activity. 
        However, abiotic processes can also produce many of these gases, requiring careful 
        contextual analysis.
        
        Primary biosignature candidates:
        
        Oxygen (O₂) and Ozone (O₃): On Earth, molecular oxygen is primarily biogenic through 
        oxygenic photosynthesis. However, photolysis of water and CO₂ can produce abiotic O₂, 
        particularly around M-dwarf stars.
        
        Methane (CH₄): Biogenic methane (from methanogenesis) must be distinguished from 
        geological sources (serpentinization, volcanism). The CH₄/CO₂ ratio and presence of 
        other species help constrain origins.
        
        Nitrous Oxide (N₂O): Produced primarily by denitrifying bacteria on Earth. Has few 
        significant abiotic sources, making it a promising biosignature.
        
        The simultaneous detection of thermodynamically incompatible species (e.g., O₂ and CH₄) 
        suggests active replenishment, potentially by life. This "chemical disequilibrium" 
        approach was proposed by Lovelock (1965) and remains a cornerstone of biosignature science.
        
        Habitability assessment considers:
        - Surface temperature in the liquid water range (273-373 K at 1 bar)
        - Atmospheric pressure sufficient for liquid water stability
        - Radiation environment (UV flux, stellar activity)
        - Presence of essential volatiles (H₂O, CO₂, N₂)
        """
    },
    {
        "id": "planetary_classification",
        "title": "Exoplanet Classification Schemes",
        "category": "planetary_science",
        "content": """
        Exoplanets are classified based on mass, radius, density, and inferred composition:
        
        Terrestrial Planets (R < 1.5 R⊕): Rocky bodies with iron cores and silicate mantles. 
        May possess thin secondary atmospheres outgassed from interiors or delivered by impacts.
        Examples: Earth, Venus, Mars analogs.
        
        Super-Earths (1.5-2 R⊕): Scaled-up terrestrials with potentially thicker atmospheres. 
        The transition from rocky to volatile-rich composition occurs in this regime.
        
        Sub-Neptunes (2-4 R⊕): Possess significant hydrogen/helium envelopes or water-rich 
        compositions. The "radius valley" at ~1.8 R⊕ separates predominantly rocky from 
        volatile-rich planets.
        
        Neptune-like (4-6 R⊕): Ice giant analogs with substantial volatile envelopes. 
        Atmospheric compositions may range from H₂/He-dominated to water-rich.
        
        Gas Giants (>6 R⊕): Jupiter and Saturn analogs with deep hydrogen-helium atmospheres. 
        Hot Jupiters in close-in orbits show inflated radii due to stellar irradiation.
        
        The mass-radius relationship constrains bulk composition:
        ρ = M/(4πR³/3)
        
        Densities >5 g/cm³ suggest rocky composition; <2 g/cm³ indicates significant volatiles.
        """
    },
    {
        "id": "signal_to_noise",
        "title": "Signal-to-Noise Considerations",
        "category": "observations",
        "content": """
        The signal-to-noise ratio (SNR) fundamentally limits detection capabilities in 
        transmission spectroscopy. The atmospheric signal amplitude is:
        
        δ ≈ 2 R_p H / R_*²
        
        For an Earth-sized planet (R_p = R⊕) with H ≈ 8.5 km around a Sun-like star, 
        δ ≈ 10 ppm per scale height—extremely challenging to detect.
        
        SNR scales with:
        - Stellar brightness (photon noise ∝ 1/√N_photons)
        - Number of transits observed (SNR ∝ √N_transits)
        - Spectral resolution (higher R means fewer photons per bin)
        - Systematic noise floor (instrumental effects, stellar variability)
        
        JWST achieves SNR > 100 for bright targets (J < 10 mag) in single transits, 
        enabling atmospheric detection for sub-Neptune and larger planets. Terrestrial 
        planet characterization around M-dwarfs requires co-adding many transits.
        
        Confidence thresholds:
        - 3σ: Marginal detection, requires confirmation
        - 5σ: Significant detection, publishable result
        - 10σ+: High-confidence, robust characterization possible
        """
    },
    {
        "id": "atmospheric_retrieval",
        "title": "Atmospheric Retrieval Methods",
        "category": "methodology",
        "content": """
        Atmospheric retrieval inverts observed spectra to infer atmospheric properties 
        through Bayesian parameter estimation. The forward model computes synthetic spectra 
        given atmospheric parameters; the retrieval finds parameters that best match observations.
        
        Key retrieved parameters:
        - Molecular abundances (volume mixing ratios)
        - Temperature-pressure profile
        - Cloud/haze properties (pressure level, opacity, particle size)
        - Reference pressure and radius
        
        Retrieval frameworks (NEMESIS, CHIMERA, petitRADTRANS, TauREx) employ:
        - Nested sampling (MultiNest) for posterior exploration
        - MCMC methods for uncertainty quantification
        - Machine learning for rapid approximate inference
        
        Degeneracies and challenges:
        - Cloud-composition degeneracy: clouds can mimic low abundances
        - Temperature-abundance degeneracy: higher T with lower VMR can match spectra
        - Limited wavelength coverage may underconstrain solutions
        
        Results are reported as posterior distributions with credible intervals, 
        acknowledging uncertainties in the inference process.
        """
    },
    {
        "id": "jwst_capabilities",
        "title": "JWST Instrumentation for Exoplanets",
        "category": "instrumentation",
        "content": """
        The James Webb Space Telescope provides unprecedented capabilities for exoplanet 
        atmospheric characterization through multiple instrument modes:
        
        NIRSpec (0.6-5.3 μm):
        - PRISM mode (R~100): Broad wavelength coverage, lower resolution
        - G395H (R~2700): High resolution for detailed spectroscopy
        - Ideal for H₂O, CO₂, CO, CH₄ detection
        
        MIRI (5-28 μm):
        - LRS (R~100): Low-resolution spectroscopy
        - MRS (R~1500-3500): Medium resolution
        - Access to CO₂ 15 μm, O₃ 9.6 μm, NH₃ features
        
        NIRCam (0.6-5 μm):
        - Grism spectroscopy (R~1500)
        - Time-series observations
        
        NIRISS SOSS (0.6-2.8 μm):
        - Slitless spectroscopy (R~700)
        - Optimized for bright star time-series
        
        Typical observation strategies:
        - Single-transit reconnaissance with PRISM
        - Multi-transit campaigns for detailed characterization
        - Combined NIRSpec + MIRI for full spectral coverage
        
        JWST has detected H₂O, CO₂, SO₂, and other species in hot Jupiter and 
        sub-Neptune atmospheres, revolutionizing the field.
        """
    },
    {
        "id": "uncertainty_analysis",
        "title": "Uncertainty Quantification in Spectral Analysis",
        "category": "statistics",
        "content": """
        Rigorous uncertainty quantification is essential for credible scientific inference. 
        Spectral analysis uncertainties arise from multiple sources:
        
        Statistical Uncertainty:
        - Photon noise: σ_photon = √N_photons
        - Read noise: Detector-specific, typically subdominant for bright sources
        - Propagates through data reduction pipeline
        
        Systematic Uncertainty:
        - Instrumental effects (detector non-linearity, persistence, flat-fielding)
        - Stellar limb darkening models
        - Stellar contamination (spots, faculae)
        - Wavelength calibration errors
        
        Model Uncertainty:
        - Opacity database completeness and accuracy
        - Line list uncertainties at high temperatures
        - Cloud/haze parameterization choices
        - Atmospheric chemistry assumptions
        
        Reporting conventions:
        - 1σ (68% CI): Standard uncertainty
        - 2σ (95% CI): Expanded uncertainty
        - Detection significance in sigma (σ) units
        
        Proper uncertainty propagation and model comparison (Bayesian evidence, 
        information criteria) are required for robust conclusions.
        """
    },
    {
        "id": "spectral_features_interpretation",
        "title": "Interpreting Spectral Features",
        "category": "analysis",
        "content": """
        Spectral feature interpretation requires understanding the physical processes 
        that shape transmission spectra:
        
        Absorption Depth:
        The depth of molecular features scales with:
        - Molecular abundance (column density along slant path)
        - Absorption cross-section (temperature-dependent)
        - Atmospheric scale height (sets path length)
        
        Feature Width:
        - Natural line width (negligible for molecules)
        - Pressure broadening (Lorentzian wings, important at high pressures)
        - Doppler broadening (Gaussian core, temperature-dependent)
        - Instrumental resolution (convolution effect)
        
        Continuum Level:
        - Rayleigh scattering (λ⁻⁴ dependence, blue-ward slope)
        - Cloud/haze opacity (grey or wavelength-dependent)
        - Collision-induced absorption (H₂-H₂, H₂-He)
        
        Spectral Slope:
        - Positive slope (increasing toward blue): Rayleigh scattering, small hazes
        - Flat continuum: Large cloud particles, high-altitude clouds
        - Negative slope: Unusual, may indicate data systematics
        
        Absence of features may indicate:
        - Low molecular abundances
        - High-altitude clouds obscuring deeper atmosphere
        - Low atmospheric scale height (high mean molecular weight, low temperature)
        - High surface gravity compressing the atmosphere
        """
    },
    {
        "id": "model_confidence",
        "title": "Machine Learning Model Confidence",
        "category": "ml_methods",
        "content": """
        Deep learning models for spectral analysis must provide calibrated confidence 
        estimates to be scientifically useful:
        
        Confidence Interpretation:
        - Model confidence reflects the neural network's certainty, not absolute truth
        - High confidence with incorrect predictions indicates miscalibration
        - Ensemble methods improve calibration through model averaging
        
        Calibration Methods:
        - Temperature scaling: Adjusts softmax temperature post-training
        - Platt scaling: Logistic regression on validation predictions
        - Isotonic regression: Non-parametric calibration curve
        
        Uncertainty Types:
        - Aleatoric uncertainty: Inherent data noise, irreducible
        - Epistemic uncertainty: Model uncertainty, reducible with more data
        - Monte Carlo dropout estimates epistemic uncertainty
        
        Validation Approach:
        - Compare attention maps to known molecular absorption wavelengths
        - Test on out-of-distribution data (different stellar types, noise levels)
        - Cross-validate against traditional retrieval results
        
        Confidence thresholds for scientific conclusions:
        - >90%: High confidence, suitable for primary conclusions
        - 70-90%: Moderate confidence, requires supporting evidence
        - <70%: Low confidence, further observation recommended
        """
    }
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Document:
    """Knowledge base document."""
    id: str
    title: str
    category: str
    content: str
    embedding: Optional[np.ndarray] = None


@dataclass 
class AnalysisContext:
    """Context from spectrum analysis results."""
    target_name: str
    molecules_detected: List[Dict[str, Any]]
    planet_class: str
    habitability_index: float
    snr: float
    temperature: float
    confidence: float
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """Chat message with role and content."""
    role: str  # 'user', 'assistant', 'system'
    content: str


@dataclass
class RAGResponse:
    """Response from RAG system."""
    answer: str
    sources: List[str]
    confidence: float
    context_used: List[str]


# ============================================================================
# Vector Store
# ============================================================================

class VectorStore:
    """Simple vector store for document embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model = None
        self.model_name = model_name
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
    
    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        for doc_data in docs:
            doc = Document(
                id=doc_data["id"],
                title=doc_data["title"],
                category=doc_data["category"],
                content=doc_data["content"]
            )
            self.documents.append(doc)
        
        self._compute_embeddings()
    
    def _compute_embeddings(self) -> None:
        """Compute embeddings for all documents."""
        if self.model is None:
            return
        
        texts = [f"{doc.title}\n{doc.content}" for doc in self.documents]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        self.embeddings = np.array(embeddings)
        
        for i, doc in enumerate(self.documents):
            doc.embedding = embeddings[i]
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Search for relevant documents."""
        if self.model is None or self.embeddings is None:
            # Fallback to keyword matching
            return self._keyword_search(query, top_k)
        
        query_embedding = self.model.encode([query])[0]
        
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.documents[i], float(similarities[i])) for i in top_indices]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Fallback keyword-based search."""
        query_terms = set(query.lower().split())
        scores = []
        
        for doc in self.documents:
            text = f"{doc.title} {doc.content}".lower()
            score = sum(1 for term in query_terms if term in text)
            scores.append((doc, score / len(query_terms) if query_terms else 0))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """You are ExoSpectraBot, a scientific AI assistant specializing in exoplanet atmospheric analysis. You explain spectrum analysis results using precise astrophysical terminology while remaining accessible to researchers at various levels.

Your communication style:
- Use scientific terminology accurately (e.g., "transmission spectrum," "molecular opacity," "scale height")
- Cite specific wavelengths when discussing molecular features
- Quantify uncertainties and confidence levels
- Reference physical mechanisms underlying observations
- Maintain objectivity about detection significance
- Acknowledge limitations and alternative interpretations

When explaining analysis results:
1. Describe detected molecular species with their characteristic absorption bands
2. Interpret the planetary classification in context of mass-radius relationships
3. Assess habitability based on temperature, atmosphere, and biosignature potential
4. Discuss confidence levels and what additional observations might help
5. Connect findings to broader exoplanetary science context

You have access to retrieved knowledge about astrophysics, spectroscopy, and exoplanet science to provide accurate, well-grounded explanations."""


def build_context_prompt(
    analysis: Optional[AnalysisContext],
    retrieved_docs: List[Tuple[Document, float]]
) -> str:
    """Build context section of the prompt."""
    context_parts = []
    
    # Add analysis results if available
    if analysis:
        analysis_text = f"""
CURRENT ANALYSIS RESULTS:
Target: {analysis.target_name}
Planet Classification: {analysis.planet_class}
Habitability Index: {analysis.habitability_index:.1%}
Equilibrium Temperature: {analysis.temperature:.0f} K
Signal-to-Noise Ratio: {analysis.snr:.0f}
Model Confidence: {analysis.confidence:.1%}

Detected Molecules:
"""
        for mol in analysis.molecules_detected:
            if mol.get('detected', False):
                analysis_text += f"- {mol['formula']} ({mol['name']}): {mol['confidence']:.0%} confidence, {mol['significance']:.1f}σ significance"
                if mol.get('abundance'):
                    analysis_text += f", abundance ~{mol['abundance']:.1e}"
                analysis_text += "\n"
        
        context_parts.append(analysis_text)
    
    # Add retrieved knowledge
    if retrieved_docs:
        knowledge_text = "\nRELEVANT KNOWLEDGE:\n"
        for doc, score in retrieved_docs:
            knowledge_text += f"\n[{doc.title}] (relevance: {score:.2f})\n{doc.content[:800]}...\n"
        context_parts.append(knowledge_text)
    
    return "\n".join(context_parts)


# ============================================================================
# Response Generator
# ============================================================================

class ResponseGenerator:
    """Generate responses using LLM or rule-based fallback."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = None
        if HAS_OPENAI and api_key:
            self.client = OpenAI(api_key=api_key)
    
    def generate(
        self,
        query: str,
        context: str,
        chat_history: List[ChatMessage]
    ) -> str:
        """Generate response to user query."""
        if self.client:
            return self._generate_with_openai(query, context, chat_history)
        else:
            return self._generate_fallback(query, context)
    
    def _generate_with_openai(
        self,
        query: str,
        context: str,
        chat_history: List[ChatMessage]
    ) -> str:
        """Generate using OpenAI API."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        if context:
            messages.append({"role": "system", "content": f"CONTEXT:\n{context}"})
        
        for msg in chat_history[-6:]:  # Last 6 messages for context
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": query})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_fallback(self, query: str, context: str) -> str:
        """Rule-based fallback when no LLM is available."""
        query_lower = query.lower()
        
        # Check for molecular questions
        molecules = {
            'water': ('H₂O', '1.4, 1.9, 2.7, and 6.3 μm', 'essential for habitability'),
            'h2o': ('H₂O', '1.4, 1.9, 2.7, and 6.3 μm', 'essential for habitability'),
            'carbon dioxide': ('CO₂', '4.3 and 15 μm', 'greenhouse gas'),
            'co2': ('CO₂', '4.3 and 15 μm', 'greenhouse gas'),
            'methane': ('CH₄', '2.3, 3.3, and 7.7 μm', 'potential biosignature'),
            'ch4': ('CH₄', '2.3, 3.3, and 7.7 μm', 'potential biosignature'),
            'ozone': ('O₃', '9.6 μm', 'biosignature indicating oxygen'),
            'o3': ('O₃', '9.6 μm', 'biosignature indicating oxygen'),
            'oxygen': ('O₂', '0.76 and 1.27 μm', 'key biosignature'),
            'o2': ('O₂', '0.76 and 1.27 μm', 'key biosignature'),
        }
        
        for mol_key, (formula, bands, significance) in molecules.items():
            if mol_key in query_lower:
                return f"""**{formula} Detection Analysis**

{formula} exhibits characteristic absorption features at {bands} in the infrared transmission spectrum. These ro-vibrational bands arise from molecular transitions that are diagnostic of atmospheric composition.

From a scientific perspective, {formula} is {significance}. The detection confidence depends on:

1. **Signal-to-Noise Ratio**: Higher SNR enables more robust feature identification
2. **Spectral Resolution**: Higher resolution resolves individual absorption lines
3. **Atmospheric Scale Height**: Larger scale heights produce stronger spectral signatures

The significance is quantified in sigma (σ) units, where >3σ indicates marginal detection and >5σ represents high-confidence detection suitable for publication.

Would you like me to elaborate on the physical mechanisms of {formula} absorption or discuss its implications for habitability assessment?"""

        # Check for habitability questions
        if 'habitability' in query_lower or 'habitable' in query_lower:
            return """**Habitability Assessment Framework**

The habitability index quantifies the potential for a planetary environment to support liquid water and, by extension, life as we know it. This assessment integrates multiple factors:

1. **Temperature Factor**: Surface temperature must fall within the liquid water stability range (273-373 K at 1 bar). The equilibrium temperature is derived from stellar irradiance and planetary albedo.

2. **Atmospheric Factor**: Presence of a substantial atmosphere is required for pressure support and greenhouse warming. Composition affects both temperature regulation and biosignature potential.

3. **Water Presence**: Direct detection of H₂O absorption indicates volatile inventory. The 2.7 μm band is particularly diagnostic in transmission spectroscopy.

4. **Radiation Environment**: Stellar UV flux and activity affect atmospheric photochemistry and surface habitability. M-dwarf planets face enhanced challenges from stellar flares.

The composite habitability index represents a weighted combination of these factors, with uncertainties propagated from individual measurements. Values >70% suggest conditions potentially conducive to life; however, this remains a simplified metric requiring careful interpretation.

What specific aspect of the habitability assessment would you like me to elaborate on?"""

        # Check for confidence/uncertainty questions
        if 'confidence' in query_lower or 'uncertainty' in query_lower:
            return """**Uncertainty Quantification in Spectral Analysis**

Model confidence reflects the neural network's certainty in its predictions, quantified through several uncertainty sources:

**Statistical Uncertainty** (~12%): 
Arises from photon noise and detector characteristics. Scales as 1/√N with number of photons, improved by longer integrations or co-adding multiple transits.

**Systematic Uncertainty** (~18%):
Instrumental effects, stellar limb darkening models, and contamination from stellar activity. Often represents the noise floor that limits precision regardless of integration time.

**Model Uncertainty** (~25%):
Inherent to the deep learning architecture—epistemic uncertainty from limited training data coverage. Estimated through Monte Carlo dropout or ensemble methods.

The combined confidence accounts for all sources through quadrature summation. High confidence (>90%) indicates robust predictions; moderate confidence (70-90%) suggests additional observations would strengthen conclusions.

Detection significance in σ units specifically refers to how many standard deviations a spectral feature exceeds the noise level, directly quantifying false-positive probability.

Would you like more detail on any specific uncertainty component?"""

        # Check for classification questions
        if 'class' in query_lower or 'type' in query_lower or 'planet' in query_lower:
            return """**Planetary Classification from Spectroscopy**

Exoplanets are classified based on their bulk properties and inferred composition:

**Terrestrial (R < 1.5 R⊕)**: Rocky bodies analogous to Earth, Venus, and Mars. Thin secondary atmospheres if present, potentially habitable if in the temperate zone.

**Super-Earth (1.5-2 R⊕)**: Scaled-up terrestrials that may retain primary atmospheres. The "radius valley" at ~1.8 R⊕ marks the transition between rocky and volatile-rich compositions.

**Sub-Neptune (2-4 R⊕)**: Significant H₂/He envelopes or water-rich compositions. Atmospheric characterization reveals whether hydrogen-dominated or high mean molecular weight.

**Neptune-like (4-6 R⊕)**: Ice giant analogs with deep volatile envelopes. Spectroscopy probes upper atmospheric chemistry.

**Gas Giant (>6 R⊕)**: Jupiter/Saturn analogs. Hot Jupiters show inflated radii and inverted temperature profiles from intense stellar irradiation.

Classification confidence derives from consistency between observed spectral features and expected atmospheric signatures for each planetary type. The mass-radius relationship provides additional constraints when radial velocity data is available.

What aspects of the classification would you like me to explain further?"""

        # Default contextual response
        if context:
            return f"""Based on the current analysis context, I can provide scientific interpretation of the spectrum analysis results.

{context[:500]}...

The analysis reveals important constraints on the atmospheric composition and planetary properties. Detection confidences reflect both the signal strength in the data and model certainties.

For rigorous scientific conclusions, we consider:
- Statistical significance of spectral features (σ levels)
- Consistency with theoretical atmospheric models
- Comparison with similar characterized exoplanets

Please ask a specific question about the detected molecules, planetary classification, habitability assessment, or methodology, and I'll provide a detailed scientific explanation."""
        
        return """I'm ExoSpectraBot, your scientific assistant for exoplanet atmospheric analysis. I can help explain:

• **Molecular Detections**: Interpretation of H₂O, CO₂, CH₄, O₃, and other atmospheric species
• **Spectral Features**: Understanding absorption bands and their physical origins
• **Habitability Assessment**: Factors contributing to planetary habitability indices
• **Classification**: Planetary types and their characteristic properties
• **Confidence & Uncertainty**: Statistical interpretation of detection significance

Please provide a specific question about your spectrum analysis results, and I'll offer a detailed scientific explanation using appropriate astrophysical terminology."""


# ============================================================================
# RAG Chatbot
# ============================================================================

class SpectrumAnalysisChatbot:
    """RAG-based chatbot for spectrum analysis explanation."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.vector_store = VectorStore()
        self.vector_store.add_documents(ASTROPHYSICS_KNOWLEDGE_BASE)
        self.generator = ResponseGenerator(api_key)
        self.chat_history: List[ChatMessage] = []
        self.current_analysis: Optional[AnalysisContext] = None
    
    def set_analysis_context(self, analysis: AnalysisContext) -> None:
        """Set the current analysis results for context."""
        self.current_analysis = analysis
    
    def clear_history(self) -> None:
        """Clear chat history."""
        self.chat_history = []
    
    def chat(self, user_message: str) -> RAGResponse:
        """Process user message and generate response."""
        # Add user message to history
        self.chat_history.append(ChatMessage(role="user", content=user_message))
        
        # Retrieve relevant documents
        retrieved = self.vector_store.search(user_message, top_k=3)
        
        # Build context
        context = build_context_prompt(self.current_analysis, retrieved)
        
        # Generate response
        response_text = self.generator.generate(
            user_message,
            context,
            self.chat_history
        )
        
        # Add assistant response to history
        self.chat_history.append(ChatMessage(role="assistant", content=response_text))
        
        # Calculate confidence based on retrieval scores
        avg_relevance = np.mean([score for _, score in retrieved]) if retrieved else 0.5
        
        return RAGResponse(
            answer=response_text,
            sources=[doc.title for doc, _ in retrieved],
            confidence=float(avg_relevance),
            context_used=[doc.id for doc, _ in retrieved]
        )
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested follow-up questions based on context."""
        suggestions = [
            "What do the detected molecules tell us about this planet's atmosphere?",
            "How confident can we be in the habitability assessment?",
            "What additional observations would strengthen these conclusions?",
            "How does this planet compare to other characterized exoplanets?",
        ]
        
        if self.current_analysis:
            detected = [m['formula'] for m in self.current_analysis.molecules_detected 
                       if m.get('detected')]
            if 'H₂O' in detected and 'CH₄' in detected:
                suggestions.insert(0, "What are the implications of detecting both water and methane?")
            if self.current_analysis.habitability_index > 0.7:
                suggestions.insert(0, "What makes this planet a strong habitability candidate?")
        
        return suggestions[:5]


# ============================================================================
# FastAPI Integration
# ============================================================================

def create_chatbot_router():
    """Create FastAPI router for chatbot endpoints."""
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
    
    router = APIRouter(prefix="/api/chat", tags=["chatbot"])
    
    # Global chatbot instance
    chatbot = SpectrumAnalysisChatbot(api_key=os.getenv("OPENAI_API_KEY"))
    
    class ChatRequest(BaseModel):
        message: str
        analysis_context: Optional[Dict[str, Any]] = None
    
    class ChatResponse(BaseModel):
        answer: str
        sources: List[str]
        confidence: float
        suggested_questions: List[str]
    
    @router.post("/message", response_model=ChatResponse)
    async def send_message(request: ChatRequest):
        """Send a message to the chatbot."""
        try:
            # Update analysis context if provided
            if request.analysis_context:
                context = AnalysisContext(
                    target_name=request.analysis_context.get("target_name", "Unknown"),
                    molecules_detected=request.analysis_context.get("molecules", []),
                    planet_class=request.analysis_context.get("planet_class", "Unknown"),
                    habitability_index=request.analysis_context.get("habitability", 0.0),
                    snr=request.analysis_context.get("snr", 0.0),
                    temperature=request.analysis_context.get("temperature", 0.0),
                    confidence=request.analysis_context.get("confidence", 0.0)
                )
                chatbot.set_analysis_context(context)
            
            response = chatbot.chat(request.message)
            
            return ChatResponse(
                answer=response.answer,
                sources=response.sources,
                confidence=response.confidence,
                suggested_questions=chatbot.get_suggested_questions()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/clear")
    async def clear_chat():
        """Clear chat history."""
        chatbot.clear_history()
        return {"status": "cleared"}
    
    @router.get("/suggestions")
    async def get_suggestions():
        """Get suggested questions."""
        return {"suggestions": chatbot.get_suggested_questions()}
    
    return router


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for testing the chatbot."""
    print("=" * 60)
    print("ExoSpectraBot - Spectrum Analysis Chatbot")
    print("=" * 60)
    print("\nInitializing knowledge base...")
    
    chatbot = SpectrumAnalysisChatbot()
    
    # Set demo analysis context
    demo_analysis = AnalysisContext(
        target_name="TRAPPIST-1e",
        molecules_detected=[
            {"formula": "H₂O", "name": "Water", "detected": True, "confidence": 0.94, "significance": 8.2, "abundance": 1.2e-3},
            {"formula": "CO₂", "name": "Carbon Dioxide", "detected": True, "confidence": 0.89, "significance": 6.5, "abundance": 4.5e-4},
            {"formula": "CH₄", "name": "Methane", "detected": True, "confidence": 0.76, "significance": 4.1, "abundance": 2.1e-5},
            {"formula": "O₃", "name": "Ozone", "detected": True, "confidence": 0.82, "significance": 5.3},
            {"formula": "O₂", "name": "Oxygen", "detected": False, "confidence": 0.35, "significance": 1.8},
        ],
        planet_class="Super-Earth",
        habitability_index=0.73,
        snr=127,
        temperature=285,
        confidence=0.81
    )
    chatbot.set_analysis_context(demo_analysis)
    
    print("\nDemo analysis loaded for TRAPPIST-1e")
    print("Type 'quit' to exit, 'clear' to reset history")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            if user_input.lower() == 'clear':
                chatbot.clear_history()
                print("Chat history cleared.")
                continue
            
            response = chatbot.chat(user_input)
            
            print(f"\nExoSpectraBot: {response.answer}")
            print(f"\n[Sources: {', '.join(response.sources)}]")
            print(f"[Confidence: {response.confidence:.0%}]")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
