import axios from 'axios';
import type { PolygonPoint, SamplingParams, SamplingResponse, TaskStatus } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  async runSampling(
    numScenes: number,
    params: SamplingParams,
    existingOutputDir?: string
  ): Promise<SamplingResponse> {
    const requestData = {
      num_scenes: numScenes,
      ...params,
      ...(existingOutputDir && { existing_output_dir: existingOutputDir }),
    };
    const response = await axios.post<SamplingResponse>(
      `${API_BASE_URL}/api/sampling/run`,
      requestData
    );
    return response.data;
  },

  async runPolygonSampling(
    polygonPoints: PolygonPoint[],
    numScenes: number,
    params: SamplingParams
  ): Promise<SamplingResponse> {
    const requestData = {
      polygon_points: polygonPoints,
      num_scenes: numScenes,
      ...params,
    };
    const response = await axios.post<SamplingResponse>(
      `${API_BASE_URL}/api/sampling/run-with-polygon`,
      requestData
    );
    return response.data;
  },

  async getTaskStatus(taskId: string): Promise<TaskStatus> {
    const response = await axios.get<TaskStatus>(
      `${API_BASE_URL}/api/sampling/status/${taskId}`
    );
    return response.data;
  },

  async getImages(taskId: string, limit: number = 50) {
    const response = await axios.get(`${API_BASE_URL}/api/images/${taskId}`, {
      params: { limit },
    });
    return response.data;
  },
};

export const getImageUrl = (taskId: string, filename: string) =>
  `${API_BASE_URL}/api/images/${taskId}/file/${filename}`;

export const getModelUrl = (taskId: string, filename: string) =>
  `${API_BASE_URL}/api/models/${taskId}/file/${filename}`;
