clear
PYTHONPATH=. python scripts/custom_sample_and_render.py \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    +num_scenes=1000 \
    algorithm=scene_diffuser_flux_transformer \
    algorithm.trainer=rl_score \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.custom.loss=true \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.ema.use=True \
    experiment.test.batch_size=196 \
    algorithm.classifier_free_guidance.use_floor=False \
    load=xn1h20rz \
    checkpoint_version=40

# python ../ThreedFront/scripts/render_results.py  --without_floor /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-22/07-07-33/sampled_scenes_results.pkl  