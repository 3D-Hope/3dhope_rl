PYTHONPATH=. python main.py +name=continuous_mi_separated_loss \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
algorithm=scene_diffuser_midiffusion \
algorithm.trainer=ddpm \
algorithm.noise_schedule.scheduler=ddim \
algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.use_floor=True \
algorithm.classifier_free_guidance.weight=0 \
algorithm.custom.loss=true \
algorithm.ema.use=True \
dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm_no_prm \
algorithm.custom.old=False \
