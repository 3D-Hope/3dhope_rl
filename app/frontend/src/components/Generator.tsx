import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, getImageUrl, getModelUrl } from '../utils/api';
import type { PolygonPoint, SamplingParams, TaskStatus } from '../types';
import { PolygonCanvas } from './PolygonCanvas';
import { ThreeJSGLBViewer } from './ThreeJSGLBViewer';

const DEFAULT_PARAMS: SamplingParams = {
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
};

export function Generator() {
  const navigate = useNavigate();
  const [numScenes, setNumScenes] = useState(5);
  const [isRunning, setIsRunning] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<TaskStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [images, setImages] = useState<string[]>([]);
  const [imageLimit, setImageLimit] = useState(50);
  const [totalImages, setTotalImages] = useState(0);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [glbUrl, setGlbUrl] = useState<string | null>(null);
  const [useExistingOutput, setUseExistingOutput] = useState(false);
  const [existingOutputDir, setExistingOutputDir] = useState('');
  const [usePolygonInput, setUsePolygonInput] = useState(false);
  const [polygonPoints, setPolygonPoints] = useState<PolygonPoint[]>([]);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [params, setParams] = useState<SamplingParams>(DEFAULT_PARAMS);

  // Poll task status
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (taskId && isRunning) {
      interval = setInterval(async () => {
        try {
          const taskStatus = await api.getTaskStatus(taskId);
          setStatus(taskStatus);

          if (taskStatus.status === 'completed' || taskStatus.status === 'failed') {
            setIsRunning(false);
            if (interval) clearInterval(interval);
          }
        } catch (err) {
          console.error('Error fetching status:', err);
          setError('Failed to fetch task status');
          setIsRunning(false);
          if (interval) clearInterval(interval);
        }
      }, 2000);
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

  // Fetch images
  const fetchImages = useCallback(async (taskIdToFetch: string) => {
    try {
      const data = await api.getImages(taskIdToFetch, imageLimit);
      setImages(data.images || []);
      setTotalImages(data.total || 0);
    } catch (err) {
      console.error('Error fetching images:', err);
      if ((err as any).response?.status !== 404) {
        setImages([]);
      }
    }
  }, [imageLimit]);

  // Fetch images when task completes
  useEffect(() => {
    if (taskId && status?.status === 'completed') {
      fetchImages(taskId);
    }
  }, [imageLimit, taskId, status?.status, fetchImages]);

  // Polygon canvas click handler
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!usePolygonInput || isRunning) return;

    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const z = (e.clientY - rect.top) * scaleY;

    const worldX = ((x / canvas.width) * 6) - 3;
    const worldZ = ((z / canvas.height) * 6) - 3;

    setPolygonPoints([...polygonPoints, { x: worldX, z: worldZ }]);
  };

  const clearPolygon = () => {
    setPolygonPoints([]);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsRunning(true);
    setError(null);
    setStatus(null);

    try {
      if (usePolygonInput) {
        if (polygonPoints.length < 3) {
          setError('Please draw at least 3 points to form a polygon');
          setIsRunning(false);
          return;
        }

        const response = await api.runPolygonSampling(polygonPoints, numScenes, params);
        setTaskId(response.task_id);
        setStatus({
          task_id: response.task_id,
          status: 'running',
          message: 'Starting polygon-based scene generation...',
        });
      } else {
        const response = await api.runSampling(
          numScenes,
          params,
          useExistingOutput && existingOutputDir.trim() ? existingOutputDir.trim() : undefined
        );
        setTaskId(response.task_id);
        setStatus({
          task_id: response.task_id,
          status: 'running',
          message: useExistingOutput
            ? 'Checking existing output directory...'
            : 'Starting sampling process...',
        });
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to start sampling');
      setIsRunning(false);
    }
  };

  const handleParamChange = (key: keyof SamplingParams, value: any) => {
    setParams((prev) => ({
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

  const handleImageClick = (imageUrl: string) => {
    setSelectedImage(imageUrl);
    // Extract filename from URL and get corresponding GLB
    const filename = imageUrl.split('/').pop()?.replace('.png', '') || '';
    if (taskId) {
      setGlbUrl(getModelUrl(taskId, filename));
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <div className="header-content">
            <h1>Auto-Decor</h1>
            <button className="back-button" onClick={() => navigate('/')}>
              ← Back to Home
            </button>
          </div>
          <p>Generate 3D scenes using AI models</p>
        </header>

        <form onSubmit={handleSubmit} className="form">
          <div className="form-group checkbox-group">
            <label>
              <input
                type="checkbox"
                checked={usePolygonInput}
                onChange={(e) => {
                  setUsePolygonInput(e.target.checked);
                  if (e.target.checked) {
                    setUseExistingOutput(false);
                  }
                }}
                disabled={isRunning}
              />
              Use Custom Polygon Floor Plan
            </label>
          </div>

          {usePolygonInput && (
            <div className="form-group">
              <label>
                Draw Floor Plan Polygon <span className="required">*</span>
              </label>
              <PolygonCanvas
                polygonPoints={polygonPoints}
                onCanvasClick={handleCanvasClick}
                onClear={clearPolygon}
                disabled={isRunning}
              />
            </div>
          )}

          <div className="form-group checkbox-group">
            <label>
              <input
                type="checkbox"
                checked={useExistingOutput}
                onChange={(e) => {
                  setUseExistingOutput(e.target.checked);
                  if (e.target.checked) {
                    setUsePolygonInput(false);
                  }
                }}
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
                <select
                  id="algorithm_noise_schedule_scheduler"
                  value={params.algorithm_noise_schedule_scheduler}
                  onChange={(e) => handleParamChange('algorithm_noise_schedule_scheduler', e.target.value)}
                  disabled={isRunning}
                >
                  <option value="ddim">DDIM</option>
                  <option value="ddpm">DDPM</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="algorithm_noise_schedule_ddim_num_inference_timesteps">
                  {params.algorithm_noise_schedule_scheduler === 'ddim' ? 'DDIM' : 'DDPM'} Inference Timesteps
                </label>
                <input
                  type="number"
                  id="algorithm_noise_schedule_ddim_num_inference_timesteps"
                  min="1"
                  max="1000"
                  value={params.algorithm_noise_schedule_ddim_num_inference_timesteps}
                  onChange={(e) => handleParamChange('algorithm_noise_schedule_ddim_num_inference_timesteps', parseInt(e.target.value) || 150)}
                  disabled={isRunning}
                />
                <small>Number of inference steps (typically 50-200 for DDIM, 1000 for DDPM)</small>
              </div>
            </div>
          )}

          <button
            type="submit"
            className="submit-button"
            disabled={isRunning}
          >
            {isRunning ? 'Generating...' : 'Generate Scenes'}
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
          </div>
        )}

        {status?.status === 'completed' && images.length > 0 && (
          <div className="images-container">
            <div className="images-header">
              <h3>Generated Scenes ({totalImages} total)</h3>
              <div className="image-controls">
                <label>
                  Show:
                  <input
                    type="number"
                    className="image-limit-input"
                    min="1"
                    max="100"
                    value={imageLimit}
                    onChange={(e) => setImageLimit(parseInt(e.target.value) || 50)}
                  />
                </label>
                <span className="image-count">
                  Showing {images.length} of {totalImages}
                </span>
              </div>
            </div>
            <div className="images-grid">
              {images.map((imageUrl, index) => (
                <div
                  key={index}
                  className="image-item"
                  onClick={() => handleImageClick(imageUrl)}
                >
                  <img src={imageUrl} alt={`Generated scene ${index + 1}`} />
                  <div className="image-overlay">
                    <span>Click to view 3D</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedImage && glbUrl && (
          <div className="modal-overlay" onClick={() => setSelectedImage(null)}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>3D Scene Viewer</h3>
                <button
                  className="close-button"
                  onClick={() => setSelectedImage(null)}
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
