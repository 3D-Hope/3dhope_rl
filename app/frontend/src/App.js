import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Three.js GLB Viewer Component
function ThreeJSGLBViewer({ glbUrl }) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const animationFrameRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xcccccc); // Light gray background
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      50,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(5, 3, 5);

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls setup
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1;
    controls.maxDistance = 50;
    controlsRef.current = controls;

    // Lighting - Brighter setup for better visibility
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
    scene.add(ambientLight);

    // Hemisphere light for natural lighting
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x888888, 0.8);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    // Main directional light (sun-like, from top-right)
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 1.8);
    directionalLight1.position.set(10, 15, 10);
    scene.add(directionalLight1);

    // Fill light from the left
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight2.position.set(-10, 10, 5);
    scene.add(directionalLight2);

    // Back light for better depth
    const directionalLight3 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight3.position.set(0, 5, -10);
    scene.add(directionalLight3);

    // Additional top light for overall brightness
    const topLight = new THREE.DirectionalLight(0xffffff, 0.6);
    topLight.position.set(0, 20, 0);
    scene.add(topLight);

    // Load GLB model
    const loader = new GLTFLoader();
    loader.load(
      glbUrl,
      (gltf) => {
        const model = gltf.scene;
        
        // Center and scale the model
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 3 / maxDim;
        model.scale.multiplyScalar(scale);
        
        model.position.x = -center.x * scale;
        model.position.y = -center.y * scale;
        model.position.z = -center.z * scale;
        
        scene.add(model);
        
        // Adjust camera to fit the model
        camera.position.set(size.x * scale, size.y * scale, size.z * scale * 2);
        controls.target.set(0, 0, 0);
        controls.update();
      },
      undefined,
      (error) => {
        console.error('Error loading GLB model:', error);
      }
    );

    // Animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (controlsRef.current) {
        controlsRef.current.dispose();
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      if (containerRef.current && rendererRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
      }
    };
  }, [glbUrl]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        minHeight: '400px',
      }}
    />
  );
}

function App() {
  const [numScenes, setNumScenes] = useState(10);
  const [isRunning, setIsRunning] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [images, setImages] = useState([]);
  const [imageLimit, setImageLimit] = useState(50);
  const [totalImages, setTotalImages] = useState(0);
  const [selectedImage, setSelectedImage] = useState(null);
  const [glbUrl, setGlbUrl] = useState(null);
  const [useExistingOutput, setUseExistingOutput] = useState(false);
  const [existingOutputDir, setExistingOutputDir] = useState('');
  const logContainerRef = useRef(null);

  // Advanced parameters with defaults
  const [params, setParams] = useState({
    load: 'gtjphzpb',
    dataset: 'custom_scene',
    dataset_processed_scene_data_path: '/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/data/metadatas/custom_scene_metadata.json',
    dataset_max_num_objects_per_scene: 12,
    algorithm: 'scene_diffuser_midiffusion',
    algorithm_trainer: 'rl_score',
    experiment_find_unused_parameters: true,
    algorithm_classifier_free_guidance_use: false,
    algorithm_classifier_free_guidance_use_floor: true,
    algorithm_classifier_free_guidance_weight: 1,
    algorithm_custom_loss: true,
    algorithm_ema_use: true,
    algorithm_noise_schedule_scheduler: 'ddim',
    algorithm_noise_schedule_ddim_num_inference_timesteps: 150,
  });

  useEffect(() => {
    let interval = null;
    if (taskId && isRunning) {
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API_BASE_URL}/api/sampling/status/${taskId}`);
          const taskStatus = response.data;
          setStatus(taskStatus);
          
          if (taskStatus.status === 'completed' || taskStatus.status === 'failed') {
            setIsRunning(false);
            if (interval) clearInterval(interval);
            
            // Fetch images if task completed
            if (taskStatus.status === 'completed' && taskId) {
              // Images will be fetched by the useEffect when status changes
            }
          }
        } catch (err) {
          console.error('Error fetching status:', err);
          setError('Failed to fetch task status');
          setIsRunning(false);
          if (interval) clearInterval(interval);
        }
      }, 2000); // Poll every 2 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [taskId, isRunning]);

  // Auto-scroll to bottom when status updates
  useEffect(() => {
    if (logContainerRef.current && status) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [status]);

  // Fetch images function - uses current imageLimit state
  const fetchImages = useCallback(async (taskIdToFetch) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/images/${taskIdToFetch}`, {
        params: { limit: imageLimit }
      });
      setImages(response.data.images || []);
      setTotalImages(response.data.total || 0);
    } catch (err) {
      console.error('Error fetching images:', err);
      // Don't show error if images just aren't ready yet
      if (err.response?.status !== 404) {
        setImages([]);
      }
    }
  }, [imageLimit]);

  // Fetch images when limit changes or task completes
  useEffect(() => {
    if (taskId && status?.status === 'completed') {
      fetchImages(taskId);
    }
  }, [imageLimit, taskId, status?.status, fetchImages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsRunning(true);
    setError(null);
    setStatus(null);

    try {
      const requestData = {
        num_scenes: numScenes,
        ...params,
      };

      // Add existing output directory if using it
      if (useExistingOutput && existingOutputDir.trim()) {
        requestData.existing_output_dir = existingOutputDir.trim();
      }

      const response = await axios.post(`${API_BASE_URL}/api/sampling/run`, requestData);
      setTaskId(response.data.task_id);
      setStatus({
        status: 'running',
        message: useExistingOutput ? 'Checking existing output directory...' : 'Starting sampling process...',
      });
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to start sampling');
      setIsRunning(false);
    }
  };

  const handleParamChange = (key, value) => {
    setParams(prev => ({
      ...prev,
      [key]: value,
    }));
  };

  const getStatusColor = () => {
    if (!status) return '#666';
    switch (status.status) {
      case 'running':
        return '#2196F3';
      case 'completed':
        return '#4CAF50';
      case 'failed':
        return '#F44336';
      default:
        return '#666';
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🎨 Scene Generation</h1>
          <p>Generate 3D scenes using AI models</p>
        </header>

        <form onSubmit={handleSubmit} className="form">
          <div className="form-group checkbox-group">
            <label>
              <input
                type="checkbox"
                checked={useExistingOutput}
                onChange={(e) => setUseExistingOutput(e.target.checked)}
                disabled={isRunning}
              />
              Use Existing Output Directory (Skip Sampling/Rendering)
            </label>
          </div>

          {useExistingOutput && (
            <div className="form-group">
              <label htmlFor="existingOutputDir">
                Existing Output Directory <span className="required">*</span>
              </label>
              <input
                type="text"
                id="existingOutputDir"
                value={existingOutputDir}
                onChange={(e) => setExistingOutputDir(e.target.value)}
                disabled={isRunning}
                placeholder="e.g., outputs/2025-12-14/06-24-41 or /full/path/to/output/dir"
                required={useExistingOutput}
              />
              <small>Provide path to existing output directory with PNG and GLB files</small>
            </div>
          )}

          {!useExistingOutput && (
            <div className="form-group">
              <label htmlFor="numScenes">
                Number of Scenes <span className="required">*</span>
              </label>
              <input
                type="number"
                id="numScenes"
                min="1"
                max="1000"
                value={numScenes}
                onChange={(e) => setNumScenes(parseInt(e.target.value) || 1)}
                disabled={isRunning}
                required
              />
              <small>Enter the number of scenes you want to generate (1-1000)</small>
            </div>
          )}

          <button
            type="button"
            className="toggle-advanced"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? '▼' : '▶'} Advanced Parameters
          </button>

          {showAdvanced && (
            <div className="advanced-params">
              <div className="form-group">
                <label htmlFor="load">Model Checkpoint (load)</label>
                <input
                  type="text"
                  id="load"
                  value={params.load}
                  onChange={(e) => handleParamChange('load', e.target.value)}
                  disabled={isRunning}
                />
              </div>

              <div className="form-group">
                <label htmlFor="dataset">Dataset</label>
                <input
                  type="text"
                  id="dataset"
                  value={params.dataset}
                  onChange={(e) => handleParamChange('dataset', e.target.value)}
                  disabled={isRunning}
                />
              </div>

              <div className="form-group">
                <label htmlFor="dataset_processed_scene_data_path">Processed Scene Data Path</label>
                <input
                  type="text"
                  id="dataset_processed_scene_data_path"
                  value={params.dataset_processed_scene_data_path}
                  onChange={(e) => handleParamChange('dataset_processed_scene_data_path', e.target.value)}
                  disabled={isRunning}
                />
              </div>

              <div className="form-group">
                <label htmlFor="dataset_max_num_objects_per_scene">Max Objects Per Scene</label>
                <input
                  type="number"
                  id="dataset_max_num_objects_per_scene"
                  min="1"
                  value={params.dataset_max_num_objects_per_scene}
                  onChange={(e) => handleParamChange('dataset_max_num_objects_per_scene', parseInt(e.target.value) || 12)}
                  disabled={isRunning}
                />
              </div>

              <div className="form-group">
                <label htmlFor="algorithm">Algorithm</label>
                <input
                  type="text"
                  id="algorithm"
                  value={params.algorithm}
                  onChange={(e) => handleParamChange('algorithm', e.target.value)}
                  disabled={isRunning}
                />
              </div>

              <div className="form-group">
                <label htmlFor="algorithm_trainer">Algorithm Trainer</label>
                <input
                  type="text"
                  id="algorithm_trainer"
                  value={params.algorithm_trainer}
                  onChange={(e) => handleParamChange('algorithm_trainer', e.target.value)}
                  disabled={isRunning}
                />
              </div>

              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    checked={params.experiment_find_unused_parameters}
                    onChange={(e) => handleParamChange('experiment_find_unused_parameters', e.target.checked)}
                    disabled={isRunning}
                  />
                  Find Unused Parameters
                </label>
              </div>

              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    checked={params.algorithm_classifier_free_guidance_use}
                    onChange={(e) => handleParamChange('algorithm_classifier_free_guidance_use', e.target.checked)}
                    disabled={isRunning}
                  />
                  Use Classifier-Free Guidance
                </label>
              </div>

              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    checked={params.algorithm_classifier_free_guidance_use_floor}
                    onChange={(e) => handleParamChange('algorithm_classifier_free_guidance_use_floor', e.target.checked)}
                    disabled={isRunning}
                  />
                  Use Floor in Guidance
                </label>
              </div>

              <div className="form-group">
                <label htmlFor="algorithm_classifier_free_guidance_weight">Guidance Weight</label>
                <input
                  type="number"
                  id="algorithm_classifier_free_guidance_weight"
                  min="0"
                  value={params.algorithm_classifier_free_guidance_weight}
                  onChange={(e) => handleParamChange('algorithm_classifier_free_guidance_weight', parseInt(e.target.value) || 1)}
                  disabled={isRunning}
                />
              </div>

              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    checked={params.algorithm_custom_loss}
                    onChange={(e) => handleParamChange('algorithm_custom_loss', e.target.checked)}
                    disabled={isRunning}
                  />
                  Use Custom Loss
                </label>
              </div>

              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    checked={params.algorithm_ema_use}
                    onChange={(e) => handleParamChange('algorithm_ema_use', e.target.checked)}
                    disabled={isRunning}
                  />
                  Use EMA
                </label>
              </div>

              <div className="form-group">
                <label htmlFor="algorithm_noise_schedule_scheduler">Noise Schedule Scheduler</label>
                <input
                  type="text"
                  id="algorithm_noise_schedule_scheduler"
                  value={params.algorithm_noise_schedule_scheduler}
                  onChange={(e) => handleParamChange('algorithm_noise_schedule_scheduler', e.target.value)}
                  disabled={isRunning}
                />
              </div>

              <div className="form-group">
                <label htmlFor="algorithm_noise_schedule_ddim_num_inference_timesteps">DDIM Inference Timesteps</label>
                <input
                  type="number"
                  id="algorithm_noise_schedule_ddim_num_inference_timesteps"
                  min="1"
                  value={params.algorithm_noise_schedule_ddim_num_inference_timesteps}
                  onChange={(e) => handleParamChange('algorithm_noise_schedule_ddim_num_inference_timesteps', parseInt(e.target.value) || 150)}
                  disabled={isRunning}
                />
              </div>
            </div>
          )}

          <button
            type="submit"
            className="submit-button"
            disabled={isRunning}
          >
            {isRunning ? '⏳ Running...' : '🚀 Generate Scenes'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {status && (
          <div className="status-container">
            <div className="status-header">
              <h3>Status</h3>
              <span
                className="status-badge"
                style={{ backgroundColor: getStatusColor() }}
              >
                {status.status.toUpperCase()}
              </span>
            </div>
            <div className="status-message" ref={logContainerRef}>
              <pre>{status.message}</pre>
            </div>
            {status.output_dir && (
              <div className="output-info">
                <strong>Output Directory:</strong> {status.output_dir}
              </div>
            )}
          </div>
        )}

        {status?.status === 'completed' && (
          <div className="images-container">
            <div className="images-header">
              <h3>Rendered Images</h3>
              <div className="image-controls">
                <label htmlFor="imageLimit">
                  Show Images: 
                  <input
                    type="number"
                    id="imageLimit"
                    min="1"
                    max="1000"
                    value={imageLimit}
                    onChange={(e) => setImageLimit(parseInt(e.target.value) || 50)}
                    className="image-limit-input"
                  />
                </label>
                {totalImages > 0 && (
                  <span className="image-count">
                    ({images.length} of {totalImages} total)
                  </span>
                )}
              </div>
            </div>
            {images.length > 0 ? (
              <div className="images-grid">
                {images.map((imageUrl, index) => {
                  // Extract filename from URL
                  const filename = imageUrl.split('/').pop();
                  const handleImageClick = () => {
                    setSelectedImage(imageUrl);
                    // Convert PNG filename to GLB filename
                    const glbFilename = filename.replace('.png', '.glb');
                    setGlbUrl(`${API_BASE_URL}/api/models/${taskId}/file/${glbFilename}`);
                  };
                  
                  return (
                    <div key={index} className="image-item" onClick={handleImageClick}>
                      <img
                        src={`${API_BASE_URL}${imageUrl}`}
                        alt={`Rendered scene ${index + 1}`}
                        loading="lazy"
                      />
                      <div className="image-overlay">
                        <span>Click to view 3D</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="no-images">
                {totalImages === 0 ? (
                  <p>No images found. Rendering may still be in progress or failed.</p>
                ) : (
                  <p>Loading images...</p>
                )}
              </div>
            )}
          </div>
        )}

        {/* 3D Model Viewer Modal */}
        {selectedImage && glbUrl && (
          <div
            className="modal-overlay"
            onClick={() => {
              setSelectedImage(null);
              setGlbUrl(null);
            }}
            tabIndex={-1}
            aria-modal="true"
            role="dialog"
          >
            <div
              className="modal-content"
              onClick={(e) => e.stopPropagation()}
              tabIndex={0}
              role="document"
            >
              <div className="modal-header">
                <h3>3D Scene Viewer</h3>
                <button
                  className="close-button"
                  aria-label="Close"
                  onClick={() => {
                    setSelectedImage(null);
                    setGlbUrl(null);
                  }}
                  tabIndex={0}
                >
                  ×
                </button>
              </div>
              <div className="model-viewer">
                <ThreeJSGLBViewer glbUrl={glbUrl} />
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
