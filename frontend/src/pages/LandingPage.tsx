import React, { useEffect, useRef } from 'react';
import './LandingPage.css';

/**
 * Landing Page - NASA/JWST Mission Dashboard Theme
 * 
 * Features:
 * - Animated galaxy background with stars and nebula effects
 * - Hero section with "Explore Alien Atmospheres" tagline
 * - Mission stats dashboard
 * - CTA to Spectrum Analyzer
 */

const LandingPage: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Animated star field background
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Star configuration
    interface Star {
      x: number;
      y: number;
      radius: number;
      opacity: number;
      twinkleSpeed: number;
      color: string;
      vx: number;
      vy: number;
    }

    const stars: Star[] = [];
    const numStars = 300;
    const colors = ['#ffffff', '#ffe4c4', '#add8e6', '#ffd700', '#ff6b6b'];

    // Initialize stars
    for (let i = 0; i < numStars; i++) {
      stars.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        radius: Math.random() * 1.5 + 0.5,
        opacity: Math.random(),
        twinkleSpeed: Math.random() * 0.02 + 0.005,
        color: colors[Math.floor(Math.random() * colors.length)],
        vx: (Math.random() - 0.5) * 0.1,
        vy: (Math.random() - 0.5) * 0.1,
      });
    }

    // Nebula clouds
    interface Nebula {
      x: number;
      y: number;
      radius: number;
      color: string;
      opacity: number;
    }

    const nebulae: Nebula[] = [
      { x: canvas.width * 0.2, y: canvas.height * 0.3, radius: 200, color: '#4a0080', opacity: 0.15 },
      { x: canvas.width * 0.7, y: canvas.height * 0.6, radius: 250, color: '#001a4a', opacity: 0.2 },
      { x: canvas.width * 0.5, y: canvas.height * 0.8, radius: 180, color: '#2d0a4e', opacity: 0.12 },
      { x: canvas.width * 0.85, y: canvas.height * 0.2, radius: 150, color: '#0a2a4a', opacity: 0.18 },
    ];

    // Animation loop
    let animationId: number;
    let time = 0;

    const animate = () => {
      time += 0.01;

      // Clear with gradient background
      const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, '#0a0a1a');
      gradient.addColorStop(0.5, '#0d1b2a');
      gradient.addColorStop(1, '#1a0a2e');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw nebulae
      nebulae.forEach((nebula) => {
        const grd = ctx.createRadialGradient(
          nebula.x, nebula.y, 0,
          nebula.x, nebula.y, nebula.radius
        );
        grd.addColorStop(0, nebula.color + '40');
        grd.addColorStop(0.5, nebula.color + '20');
        grd.addColorStop(1, 'transparent');
        ctx.fillStyle = grd;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      });

      // Draw and update stars
      stars.forEach((star) => {
        // Twinkle effect
        star.opacity += star.twinkleSpeed;
        if (star.opacity > 1 || star.opacity < 0.2) {
          star.twinkleSpeed *= -1;
        }

        // Slow drift
        star.x += star.vx;
        star.y += star.vy;

        // Wrap around
        if (star.x < 0) star.x = canvas.width;
        if (star.x > canvas.width) star.x = 0;
        if (star.y < 0) star.y = canvas.height;
        if (star.y > canvas.height) star.y = 0;

        // Draw star
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
        ctx.fillStyle = star.color + Math.floor(star.opacity * 255).toString(16).padStart(2, '0');
        ctx.fill();

        // Add glow for larger stars
        if (star.radius > 1) {
          const glow = ctx.createRadialGradient(
            star.x, star.y, 0,
            star.x, star.y, star.radius * 3
          );
          glow.addColorStop(0, star.color + '40');
          glow.addColorStop(1, 'transparent');
          ctx.fillStyle = glow;
          ctx.fillRect(star.x - star.radius * 3, star.y - star.radius * 3, star.radius * 6, star.radius * 6);
        }
      });

      // Shooting star occasionally
      if (Math.random() < 0.002) {
        const shootingStar = {
          x: Math.random() * canvas.width,
          y: 0,
          length: Math.random() * 100 + 50,
          speed: Math.random() * 10 + 5,
        };
        
        ctx.beginPath();
        ctx.moveTo(shootingStar.x, shootingStar.y);
        ctx.lineTo(shootingStar.x + shootingStar.length, shootingStar.y + shootingStar.length);
        const gradient = ctx.createLinearGradient(
          shootingStar.x, shootingStar.y,
          shootingStar.x + shootingStar.length, shootingStar.y + shootingStar.length
        );
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(1, 'transparent');
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <div className="landing-page">
      {/* Animated Background */}
      <canvas ref={canvasRef} className="galaxy-background" />

      {/* Navigation */}
      <nav className="mission-nav">
        <div className="nav-logo">
          <div className="logo-icon">
            <svg viewBox="0 0 100 100" className="jwst-icon">
              <polygon points="50,5 95,50 50,95 5,50" fill="none" stroke="currentColor" strokeWidth="2"/>
              <polygon points="50,20 80,50 50,80 20,50" fill="none" stroke="currentColor" strokeWidth="1.5"/>
              <circle cx="50" cy="50" r="10" fill="currentColor"/>
            </svg>
          </div>
          <span className="logo-text">EXOSPEC</span>
          <span className="logo-subtitle">Mission Control</span>
        </div>
        <div className="nav-links">
          <a href="#mission" className="nav-link">Mission</a>
          <a href="#discoveries" className="nav-link">Discoveries</a>
          <a href="#data" className="nav-link">Data Archive</a>
          <a href="/analyzer" className="nav-link nav-cta">Launch Analyzer</a>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="mission-badge">
            <span className="badge-icon">◈</span>
            <span>EXOPLANET SPECTRUM RECOVERY MISSION</span>
          </div>
          
          <h1 className="hero-title">
            <span className="title-line">Explore</span>
            <span className="title-line highlight">Alien Atmospheres</span>
          </h1>
          
          <p className="hero-description">
            Decode the chemical signatures of distant worlds using advanced machine learning 
            and JWST transmission spectroscopy. Discover water, methane, and the building 
            blocks of life across the cosmos.
          </p>

          <div className="hero-cta">
            <a href="/analyzer" className="cta-primary">
              <span className="cta-icon">▶</span>
              Launch Spectrum Analyzer
            </a>
            <a href="#mission" className="cta-secondary">
              <span className="cta-icon">◎</span>
              View Mission Brief
            </a>
          </div>

          <div className="hero-stats">
            <div className="stat-item">
              <div className="stat-value">5,500+</div>
              <div className="stat-label">Confirmed Exoplanets</div>
            </div>
            <div className="stat-divider"></div>
            <div className="stat-item">
              <div className="stat-value">14</div>
              <div className="stat-label">Detectable Molecules</div>
            </div>
            <div className="stat-divider"></div>
            <div className="stat-item">
              <div className="stat-value">0.6-28μm</div>
              <div className="stat-label">Wavelength Range</div>
            </div>
            <div className="stat-divider"></div>
            <div className="stat-item">
              <div className="stat-value">AI</div>
              <div className="stat-label">Powered Analysis</div>
            </div>
          </div>
        </div>

        {/* Floating JWST illustration */}
        <div className="hero-visual">
          <div className="jwst-model">
            <div className="mirror-segment"></div>
            <div className="mirror-segment"></div>
            <div className="mirror-segment"></div>
            <div className="mirror-segment"></div>
            <div className="mirror-segment"></div>
            <div className="mirror-segment"></div>
            <div className="sunshield"></div>
          </div>
          <div className="orbit-ring ring-1"></div>
          <div className="orbit-ring ring-2"></div>
          <div className="data-stream">
            {[...Array(10)].map((_, i) => (
              <div key={i} className="data-packet" style={{ animationDelay: `${i * 0.5}s` }}></div>
            ))}
          </div>
        </div>
      </section>

      {/* Mission Dashboard */}
      <section id="mission" className="mission-section">
        <div className="section-header">
          <div className="section-line"></div>
          <h2 className="section-title">Mission Dashboard</h2>
          <div className="section-line"></div>
        </div>

        <div className="dashboard-grid">
          {/* Status Panel */}
          <div className="dashboard-panel status-panel">
            <div className="panel-header">
              <span className="panel-icon">◉</span>
              <span className="panel-title">System Status</span>
              <span className="status-indicator active"></span>
            </div>
            <div className="status-grid">
              <div className="status-item">
                <span className="status-dot online"></span>
                <span>ML Models</span>
                <span className="status-text">ONLINE</span>
              </div>
              <div className="status-item">
                <span className="status-dot online"></span>
                <span>Data Pipeline</span>
                <span className="status-text">ACTIVE</span>
              </div>
              <div className="status-item">
                <span className="status-dot online"></span>
                <span>JWST Interface</span>
                <span className="status-text">CONNECTED</span>
              </div>
              <div className="status-item">
                <span className="status-dot processing"></span>
                <span>GPU Cluster</span>
                <span className="status-text">PROCESSING</span>
              </div>
            </div>
          </div>

          {/* Capabilities Panel */}
          <div className="dashboard-panel capabilities-panel">
            <div className="panel-header">
              <span className="panel-icon">◈</span>
              <span className="panel-title">Analysis Capabilities</span>
            </div>
            <div className="capabilities-list">
              <div className="capability-item">
                <div className="capability-icon">🔬</div>
                <div className="capability-info">
                  <h4>Molecule Detection</h4>
                  <p>H₂O, CO₂, CH₄, NH₃, O₃, and 9 more species</p>
                </div>
              </div>
              <div className="capability-item">
                <div className="capability-icon">🌍</div>
                <div className="capability-info">
                  <h4>Planet Classification</h4>
                  <p>Hot Jupiters to Earth-like worlds</p>
                </div>
              </div>
              <div className="capability-item">
                <div className="capability-icon">💧</div>
                <div className="capability-info">
                  <h4>Habitability Assessment</h4>
                  <p>AI-powered biosignature analysis</p>
                </div>
              </div>
              <div className="capability-item">
                <div className="capability-icon">📊</div>
                <div className="capability-info">
                  <h4>Uncertainty Quantification</h4>
                  <p>Bayesian confidence intervals</p>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Discoveries */}
          <div className="dashboard-panel discoveries-panel">
            <div className="panel-header">
              <span className="panel-icon">★</span>
              <span className="panel-title">Recent Discoveries</span>
            </div>
            <div className="discovery-feed">
              <div className="discovery-item">
                <div className="discovery-time">2h ago</div>
                <div className="discovery-text">
                  <strong>WASP-39b:</strong> CO₂ confirmed at 4.3μm with 99.7% confidence
                </div>
              </div>
              <div className="discovery-item">
                <div className="discovery-time">5h ago</div>
                <div className="discovery-text">
                  <strong>K2-18b:</strong> Potential dimethyl sulfide detection under review
                </div>
              </div>
              <div className="discovery-item">
                <div className="discovery-time">1d ago</div>
                <div className="discovery-text">
                  <strong>TRAPPIST-1e:</strong> Updated atmospheric model available
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="dashboard-panel actions-panel">
            <div className="panel-header">
              <span className="panel-icon">⚡</span>
              <span className="panel-title">Quick Launch</span>
            </div>
            <div className="action-buttons">
              <a href="/analyzer" className="action-btn primary">
                <span className="action-icon">📡</span>
                <span>Spectrum Analyzer</span>
              </a>
              <a href="/synthetic" className="action-btn">
                <span className="action-icon">🔮</span>
                <span>Simulate Spectrum</span>
              </a>
              <a href="/database" className="action-btn">
                <span className="action-icon">🗃️</span>
                <span>Exoplanet Database</span>
              </a>
              <a href="/docs" className="action-btn">
                <span className="action-icon">📖</span>
                <span>API Documentation</span>
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="mission-footer">
        <div className="footer-content">
          <div className="footer-logo">
            <span className="logo-text">EXOSPEC</span>
            <span className="footer-tagline">Exploring worlds beyond our solar system</span>
          </div>
          <div className="footer-links">
            <a href="/about">About</a>
            <a href="/privacy">Privacy</a>
            <a href="/api">API</a>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
          <div className="footer-credit">
            Powered by JWST, ML, and human curiosity • {new Date().getFullYear()}
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
