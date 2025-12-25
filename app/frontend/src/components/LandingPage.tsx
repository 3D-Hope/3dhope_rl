import { useNavigate } from 'react-router-dom';
import scene1 from '../assets/0017_SecondBedroom-19262.png';
import scene2 from '../assets/0021_MasterBedroom-7894.png';
import scene3 from '../assets/0028_MasterBedroom-4089.png';
import scene4 from '../assets/0046_Bedroom-18.png';
import scene5 from '../assets/0051_Bedroom-4098.png';
import scene6 from '../assets/0061_MasterBedroom-424.png';
import scene7 from '../assets/0062_SecondBedroom-32238.png';
import scene8 from '../assets/0064_SecondBedroom-19232.png';
import scene9 from '../assets/0066_SecondBedroom-24334.png';
import scene10 from '../assets/0075_SecondBedroom-86888.png';
import scene11 from '../assets/0076_MasterBedroom-9583.png';

const SCENE_ASSETS = [
  scene1, scene2, scene3, scene4, scene5, scene6,
  scene7, scene8, scene9, scene10, scene11
];

export function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      <div className="landing-hero">
        <h1 className="landing-title">Auto-Decor</h1>
        <p className="landing-subtitle">AI-Powered 3D Scene Generation</p>
        <p className="landing-description">
          Transform your floor plans into beautifully decorated 3D scenes using advanced AI models.
          Design custom layouts and watch as furniture is intelligently placed to create stunning interiors.
        </p>
        <button className="cta-button" onClick={() => navigate('/generate')}>
          Start Designing
        </button>
      </div>

      <div className="features-section">
        <h2>Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🎨</div>
            <h3>Custom Floor Plans</h3>
            <p>Draw your own floor plan polygon and generate scenes tailored to your space</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🤖</div>
            <h3>AI-Powered</h3>
            <p>Advanced machine learning models intelligently place furniture and decor</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📐</div>
            <h3>3D Visualization</h3>
            <p>View your generated scenes in interactive 3D with full camera controls</p>
          </div>
        </div>
      </div>

      <div className="gallery-section">
        <h2>Sample Scenes</h2>
        <p className="gallery-description">Explore examples of AI-generated bedroom scenes</p>
        <div className="gallery-grid">
          {SCENE_ASSETS.map((asset, index) => (
            <div key={index} className="gallery-item">
              <img src={asset} alt={`Generated scene ${index + 1}`} />
              <div className="gallery-overlay">
                <span>Scene {index + 1}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="landing-footer">
        <button className="cta-button-secondary" onClick={() => navigate('/generate')}>
          Get Started Now
        </button>
      </div>
    </div>
  );
}
