export interface PolygonPoint {
  x: number;
  z: number;
}

export interface SamplingParams {
  load: string;
  dataset: string;
  dataset_processed_scene_data_path: string;
  dataset_max_num_objects_per_scene: number;
  algorithm: string;
  algorithm_trainer: string;
  experiment_find_unused_parameters: boolean;
  algorithm_classifier_free_guidance_use: boolean;
  algorithm_classifier_free_guidance_use_floor: boolean;
  algorithm_classifier_free_guidance_weight: number;
  algorithm_custom_loss: boolean;
  algorithm_ema_use: boolean;
  algorithm_noise_schedule_scheduler: string;
  algorithm_noise_schedule_ddim_num_inference_timesteps: number;
}

export interface TaskStatus {
  task_id: string;
  status: 'running' | 'completed' | 'failed';
  message: string;
  output_dir?: string;
}

export interface SamplingResponse {
  status: string;
  message: string;
  task_id: string;
}
