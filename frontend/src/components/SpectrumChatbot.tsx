/**
 * Spectrum Analysis Chatbot Component
 * 
 * RAG-based scientific explanation interface:
 * - Chat with astrophysics-aware AI
 * - Context from current analysis results
 * - Scientific terminology and citations
 * - Suggested follow-up questions
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import './SpectrumChatbot.css';

// ===== Types =====
interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  sources?: string[];
  confidence?: number;
}

interface AnalysisContext {
  targetName: string;
  molecules: MoleculeResult[];
  planetClass: string;
  habitability: number;
  snr: number;
  temperature: number;
  confidence: number;
}

interface MoleculeResult {
  formula: string;
  name: string;
  detected: boolean;
  confidence: number;
  significance: number;
  abundance?: number;
}

interface ChatResponse {
  answer: string;
  sources: string[];
  confidence: number;
  suggestedQuestions: string[];
}

// ===== Demo Knowledge Base =====
const KNOWLEDGE_RESPONSES: Record<string, string> = {
  water: `**H₂O Detection Analysis**

Water vapor exhibits characteristic absorption features at **1.4, 1.9, 2.7, and 6.3 μm** in the infrared transmission spectrum. These ro-vibrational bands arise from the bending and stretching modes of the H₂O molecule.

The 2.7 μm band is particularly diagnostic as it experiences minimal interference from other atmospheric species. Detection significance is quantified in sigma (σ) units:
- **>5σ**: High-confidence detection suitable for publication
- **3-5σ**: Marginal detection requiring confirmation
- **<3σ**: Upper limit only

Water is essential for habitability assessment as it indicates:
1. Volatile inventory from planetary formation
2. Potential for liquid water at the surface (given appropriate temperature/pressure)
3. Active hydrological cycle if combined with other indicators

The abundance is typically expressed as a volume mixing ratio (VMR) or column-integrated amount.`,

  methane: `**CH₄ Detection Analysis**

Methane shows absorption bands at **2.3, 3.3, and 7.7 μm**. The 3.3 μm feature is often the strongest and most diagnostic for transmission spectroscopy.

Methane is considered a potential **biosignature** when found in thermodynamic disequilibrium with oxidizing species like CO₂ or O₂. On Earth, ~90% of atmospheric methane is biogenic (methanogenesis), though abiotic sources exist:
- Serpentinization reactions
- Volcanic outgassing
- Fischer-Tropsch synthesis

The **CH₄/CO₂ ratio** provides constraints on atmospheric chemistry:
- High CH₄/CO₂: Reducing atmosphere, possible biological activity
- Low CH₄/CO₂: Oxidizing conditions, photochemical destruction

Detection alongside H₂O strengthens the case for a habitable environment with active carbon cycling.`,

  habitability: `**Habitability Assessment Framework**

The habitability index integrates multiple environmental factors weighted by their importance for Earth-like life:

**1. Temperature Factor (35% weight)**
Equilibrium temperature must allow liquid water stability (273-373 K at 1 bar). Calculated from stellar irradiance and planetary albedo:
\`T_eq = T_star × √(R_star/2a) × (1-A)^0.25\`

**2. Atmospheric Factor (25% weight)**
Presence of substantial atmosphere for:
- Surface pressure supporting liquid water
- Greenhouse warming to moderate temperature
- UV shielding for surface habitability

**3. Water Presence (25% weight)**
Direct H₂O detection in transmission spectrum indicates volatile inventory. The 2.7 μm band is most diagnostic.

**4. Radiation Environment (15% weight)**
- UV flux affects photochemistry and surface conditions
- Stellar activity (flares) can erode atmospheres
- M-dwarf planets face enhanced challenges

**Index Interpretation:**
- **>80%**: Excellent habitability candidate
- **60-80%**: Promising, warrants follow-up
- **40-60%**: Marginal conditions
- **<40%**: Unlikely to be habitable`,

  confidence: `**Uncertainty Quantification**

Model predictions include three uncertainty sources:

**Statistical Uncertainty (~12%)**
Arises from photon counting statistics and detector noise. Fundamental limit set by:
\`σ = √(N_photons + σ_read² + σ_dark²)\`

Improved by longer integration time or co-adding multiple transits (∝ √N).

**Systematic Uncertainty (~18%)**
- Instrumental calibration effects
- Stellar limb darkening model errors
- Contamination from stellar activity (spots, faculae)
- Wavelength calibration precision

Often represents the "noise floor" limiting precision regardless of integration.

**Model Uncertainty (~25%)**
Epistemic uncertainty from the neural network:
- Limited training data coverage
- Architecture assumptions
- Out-of-distribution inputs

Estimated via Monte Carlo dropout or ensemble disagreement.

**Combined Confidence:**
Quadrature summation: \`σ_total = √(σ_stat² + σ_sys² + σ_model²)\`

**Detection Significance (σ):**
Number of standard deviations above noise level. 5σ corresponds to <1 in 3.5 million false positive probability.`,

  classification: `**Planetary Classification**

Spectroscopic classification uses atmospheric signatures combined with bulk properties:

**Terrestrial (R < 1.5 R⊕)**
- Rocky composition with thin secondary atmospheres
- High mean molecular weight if atmosphere present
- Small scale height → weak spectral features
- Examples: Earth, Venus, Mars analogs

**Super-Earth (1.5-2 R⊕)**
- Transition regime between rocky and volatile-rich
- May retain primary H/He envelopes
- The "radius valley" at ~1.8 R⊕ marks compositional boundary

**Sub-Neptune (2-4 R⊕)**
- Significant volatile envelopes (H₂/He or H₂O-rich)
- Low mean molecular weight → large scale heights
- Strong spectral features readily detectable
- Examples: K2-18b, TOI-270d

**Neptune-like (4-6 R⊕)**
- Deep atmospheres with complex chemistry
- CH₄, NH₃, H₂O features prominent
- Internal heat may affect atmospheric structure

**Gas Giant (>6 R⊕)**
- Jupiter/Saturn analogs
- H₂/He dominated with trace molecules
- Hot Jupiters show thermal inversions, atomic species

Classification probability reflects consistency between observed spectral features and theoretical expectations.`,

  biosignature: `**Biosignature Assessment**

Biosignatures are atmospheric constituents potentially indicating biological activity. Key candidates detected in this analysis:

**Primary Biosignatures:**

*Oxygen (O₂) / Ozone (O₃)*
On Earth, O₂ is produced by oxygenic photosynthesis. O₃ forms photochemically from O₂ and is detectable at 9.6 μm. Abiotic production possible via:
- H₂O photolysis + hydrogen escape
- CO₂ photolysis around M-dwarfs

*Methane (CH₄)*
Biogenic sources: methanogenesis, fermentation
Abiotic sources: serpentinization, volcanism

**Chemical Disequilibrium:**
The simultaneous presence of O₂/O₃ and CH₄ is thermodynamically unstable—these species should react to form CO₂ and H₂O. Persistent coexistence requires active replenishment, potentially by life.

Lovelock (1965) proposed this disequilibrium as a remote life detection method.

**Contextual Requirements:**
- Biosignature must exceed expected abiotic production
- Multiple independent biosignatures strengthen case
- Planetary context (temperature, radiation) must be favorable
- False positive scenarios must be excluded

**This Analysis:**
Detection of H₂O + CH₄ + O₃ together represents a compelling biosignature cocktail requiring careful follow-up observations.`
};

// ===== Helper Functions =====
const generateId = () => Math.random().toString(36).substr(2, 9);

const findBestResponse = (query: string, context?: AnalysisContext): ChatResponse => {
  const queryLower = query.toLowerCase();
  
  // Check for specific topics
  const topicMatches: Array<{keywords: string[], topic: string}> = [
    { keywords: ['water', 'h2o', 'vapor'], topic: 'water' },
    { keywords: ['methane', 'ch4'], topic: 'methane' },
    { keywords: ['habitab', 'habitable', 'life'], topic: 'habitability' },
    { keywords: ['confidence', 'uncertain', 'error', 'sigma'], topic: 'confidence' },
    { keywords: ['class', 'type', 'terrestrial', 'neptune', 'giant'], topic: 'classification' },
    { keywords: ['biosign', 'oxygen', 'ozone', 'disequilibrium'], topic: 'biosignature' }
  ];
  
  for (const { keywords, topic } of topicMatches) {
    if (keywords.some(kw => queryLower.includes(kw))) {
      return {
        answer: KNOWLEDGE_RESPONSES[topic],
        sources: [`Astrophysics Knowledge Base: ${topic}`],
        confidence: 0.92,
        suggestedQuestions: getSuggestedQuestions(topic, context)
      };
    }
  }
  
  // Context-aware generic response
  if (context) {
    const detectedMols = context.molecules.filter(m => m.detected);
    return {
      answer: `Based on the analysis of **${context.targetName}**, I can provide scientific interpretation of the results.

**Summary:**
- **Classification**: ${context.planetClass} with ${(context.confidence * 100).toFixed(0)}% model confidence
- **Habitability Index**: ${(context.habitability * 100).toFixed(0)}%
- **Temperature**: ${context.temperature} K
- **SNR**: ${context.snr}

**Detected Molecules** (${detectedMols.length} species):
${detectedMols.map(m => `- ${m.formula} (${m.name}): ${(m.confidence * 100).toFixed(0)}% confidence, ${m.significance.toFixed(1)}σ`).join('\n')}

The detection of ${detectedMols.map(m => m.formula).join(', ')} provides important constraints on atmospheric composition. ${detectedMols.length >= 3 ? 'Multiple molecular detections enable cross-validation and strengthen confidence in the atmospheric characterization.' : ''}

What specific aspect would you like me to explain in more detail?`,
      sources: ['Current Analysis Results'],
      confidence: 0.85,
      suggestedQuestions: [
        'What are the implications of these molecular detections?',
        'How confident are these detections?',
        'What does the habitability index mean?',
        'How does this compare to other exoplanets?'
      ]
    };
  }
  
  return {
    answer: `I'm **ExoSpectraBot**, your scientific assistant for exoplanet atmospheric analysis. I can help explain:

• **Molecular Detections**: Interpretation of spectral features and their significance
• **Habitability Assessment**: Environmental factors and biosignature potential  
• **Classification**: Planetary types based on atmospheric signatures
• **Confidence & Uncertainty**: Statistical interpretation of results

Please ask a specific question about your spectrum analysis, or load analysis results for context-aware explanations.

*I use astrophysical terminology and provide quantitative interpretations grounded in the scientific literature.*`,
    sources: [],
    confidence: 0.7,
    suggestedQuestions: [
      'What molecules can be detected in exoplanet atmospheres?',
      'How is habitability assessed from spectra?',
      'What are biosignature gases?',
      'How do you quantify detection confidence?'
    ]
  };
};

const getSuggestedQuestions = (topic: string, context?: AnalysisContext): string[] => {
  const baseQuestions: Record<string, string[]> = {
    water: [
      'How does water abundance affect habitability?',
      'What other molecules are typically found with water?',
      'How is the detection significance calculated?'
    ],
    methane: [
      'How do we distinguish biogenic from abiotic methane?',
      'What does the CH₄/CO₂ ratio tell us?',
      'Why is methane considered a biosignature?'
    ],
    habitability: [
      'What temperature range is considered habitable?',
      'How does stellar radiation affect habitability?',
      'What additional observations would strengthen the assessment?'
    ],
    confidence: [
      'How can systematic uncertainties be reduced?',
      'What does 5-sigma detection mean?',
      'How does SNR affect detection limits?'
    ],
    classification: [
      'What distinguishes super-Earths from sub-Neptunes?',
      'How does atmospheric composition inform classification?',
      'What is the radius valley?'
    ],
    biosignature: [
      'What is chemical disequilibrium?',
      'Can these biosignatures be produced abiotically?',
      'What follow-up observations are needed?'
    ]
  };
  
  return baseQuestions[topic] || [
    'Tell me more about the detected molecules.',
    'How confident are these results?',
    'What are the implications for habitability?'
  ];
};

const formatMarkdown = (text: string): React.ReactNode => {
  // Simple markdown formatting
  const parts = text.split(/(\*\*.*?\*\*|\*.*?\*|`.*?`|\n)/g);
  
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith('*') && part.endsWith('*')) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={i}>{part.slice(1, -1)}</code>;
    }
    if (part === '\n') {
      return <br key={i} />;
    }
    return part;
  });
};

// ===== Components =====

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  return (
    <div className={`message-bubble ${message.role}`}>
      <div className="message-avatar">
        {message.role === 'user' ? '👤' : '🔭'}
      </div>
      <div className="message-content">
        <div className="message-text">
          {formatMarkdown(message.content)}
        </div>
        {message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <span className="sources-label">Sources:</span>
            {message.sources.map((source, i) => (
              <span key={i} className="source-tag">{source}</span>
            ))}
          </div>
        )}
        {message.confidence !== undefined && (
          <div className="message-confidence">
            Confidence: {(message.confidence * 100).toFixed(0)}%
          </div>
        )}
        <div className="message-time">
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

interface SuggestedQuestionsProps {
  questions: string[];
  onSelect: (question: string) => void;
}

const SuggestedQuestions: React.FC<SuggestedQuestionsProps> = ({ questions, onSelect }) => {
  if (questions.length === 0) return null;
  
  return (
    <div className="suggested-questions">
      <span className="suggestions-label">Suggested questions:</span>
      <div className="suggestions-list">
        {questions.map((q, i) => (
          <button key={i} className="suggestion-btn" onClick={() => onSelect(q)}>
            {q}
          </button>
        ))}
      </div>
    </div>
  );
};

// ===== Main Component =====
interface SpectrumChatbotProps {
  analysisContext?: AnalysisContext;
  onClose?: () => void;
  isCompact?: boolean;
}

const SpectrumChatbot: React.FC<SpectrumChatbotProps> = ({
  analysisContext,
  onClose,
  isCompact = false
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Initial greeting
  useEffect(() => {
    const greeting: Message = {
      id: generateId(),
      role: 'assistant',
      content: analysisContext 
        ? `Welcome! I'm **ExoSpectraBot**, ready to explain the spectrum analysis results for **${analysisContext.targetName}**.

I detected ${analysisContext.molecules.filter(m => m.detected).length} molecular species with a habitability index of ${(analysisContext.habitability * 100).toFixed(0)}%.

What would you like to know about these results?`
        : `Welcome! I'm **ExoSpectraBot**, your scientific assistant for exoplanet atmospheric analysis.

I can explain molecular detections, habitability assessments, and spectroscopic methodology using astrophysical terminology.

What would you like to learn about?`,
      timestamp: new Date()
    };
    
    setMessages([greeting]);
    setSuggestedQuestions([
      'What molecules were detected?',
      'Explain the habitability assessment',
      'How confident are these results?',
      'What are biosignature gases?'
    ]);
  }, [analysisContext]);
  
  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim()) return;
    
    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: text,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 700));
    
    const response = findBestResponse(text, analysisContext);
    
    const assistantMessage: Message = {
      id: generateId(),
      role: 'assistant',
      content: response.answer,
      timestamp: new Date(),
      sources: response.sources,
      confidence: response.confidence
    };
    
    setMessages(prev => [...prev, assistantMessage]);
    setSuggestedQuestions(response.suggestedQuestions);
    setIsTyping(false);
  }, [analysisContext]);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };
  
  const handleSuggestionClick = (question: string) => {
    sendMessage(question);
  };
  
  const clearChat = () => {
    setMessages([]);
    setSuggestedQuestions([]);
  };
  
  return (
    <div className={`spectrum-chatbot ${isCompact ? 'compact' : ''}`}>
      {/* Header */}
      <header className="chatbot-header">
        <div className="header-info">
          <span className="chatbot-icon">🔭</span>
          <div>
            <h3>ExoSpectraBot</h3>
            <span className="status">Scientific Analysis Assistant</span>
          </div>
        </div>
        <div className="header-actions">
          <button className="action-btn" onClick={clearChat} title="Clear chat">
            🗑️
          </button>
          {onClose && (
            <button className="action-btn close" onClick={onClose} title="Close">
              ✕
            </button>
          )}
        </div>
      </header>
      
      {/* Context Banner */}
      {analysisContext && (
        <div className="context-banner">
          <span className="context-target">📡 {analysisContext.targetName}</span>
          <span className="context-stats">
            {analysisContext.molecules.filter(m => m.detected).length} molecules • 
            {(analysisContext.habitability * 100).toFixed(0)}% habitability
          </span>
        </div>
      )}
      
      {/* Messages */}
      <div className="chatbot-messages">
        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        
        {isTyping && (
          <div className="typing-indicator">
            <div className="message-avatar">🔭</div>
            <div className="typing-dots">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      {/* Suggestions */}
      {!isTyping && suggestedQuestions.length > 0 && (
        <SuggestedQuestions 
          questions={suggestedQuestions} 
          onSelect={handleSuggestionClick}
        />
      )}
      
      {/* Input */}
      <form className="chatbot-input" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about the spectrum analysis..."
          disabled={isTyping}
        />
        <button type="submit" disabled={!input.trim() || isTyping}>
          ➤
        </button>
      </form>
    </div>
  );
};

export default SpectrumChatbot;
