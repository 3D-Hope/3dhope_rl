clear
source .venv/bin/activate
PYTHONPATH=. python -u main.py +name=continuous_mi_bedroom_floor_obj32 \
    dataset=custom_scene \
    dataset.data.dataset_directory=bedrooms_objfeats_32_64 \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.obj_vec_len=62 \
    algorithm.custom.obj_diff_vec_len=62 \
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm

