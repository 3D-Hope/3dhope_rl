PYTHONPATH=. python scripts/sample_and_render.py load=data/checkpoints/restaurant_low_clutter.ckpt \
dataset.processed_scene_data_path=data/metadatas/restaurant_low_clutter.json \
dataset.model_path_vec_len=28 \
dataset.max_num_objects_per_scene=107 \
+num_scenes=1


PYTHONPATH=. python scripts/sample_and_render.py load=data/checkpoints/restaurant_low_clutter.ckpt \
dataset.processed_scene_data_path=data/metadatas/restaurant_low_clutter.json \
dataset.model_path_vec_len=28 \
dataset.max_num_objects_per_scene=107 \
+num_scenes=5 \
algorithm.classifier_free_guidance.weight=1.5 \
algorithm.classifier_free_guidance.sampling.labels="a scene with 2 shelves chairs and 1 table."

python scripts/inference_time_search.py load=data/checkpoints/restaurant_low_clutter.ckpt \
dataset.processed_scene_data_path=data/metadatas/restaurant_low_clutter.json \
dataset.model_path_vec_len=28 \
dataset.max_num_objects_per_scene=107 \
+num_scenes=5

PYTHONPATH=. python scripts/sample_and_render.py load=data/checkpoints/restaurant_low_clutter.ckpt \
dataset.processed_scene_data_path=data/metadatas/restaurant_low_clutter.json \
dataset.model_path_vec_len=28 \
dataset.max_num_objects_per_scene=107 \
algorithm.postprocessing.apply_forward_simulation=True \
algorithm.postprocessing.apply_non_penetration_projection=True \
+num_scenes=5

PYTHONPATH=. python scripts/sample_and_render.py load=data/checkpoints/restaurant_low_clutter.ckpt \
dataset.model_path_vec_len=28 \
dataset.max_num_objects_per_scene=107

---
# train diffuscene
python main.py +name=first algorithm=scene_diffuser_flux_transformer algorithm.trainer=rl_score algorithm.ddpo.use_non_penetration_reward=True experiment.find_unused_parameters=True algorithm.ddpo.batch_size=2

<!-- dataset.processed_scene_data_path=data/metadatas/restaurant_low_clutter.json \
dataset.max_num_objects_per_scene=30 \
algorithm.classifier_free_guidance.max_length=30 \
dataset.model_path_vec_len=19 \ -->

## rl diffuscene
python main.py +name=first \
load=data/checkpoints/restaurant_low_clutter.ckpt \
dataset.processed_scene_data_path=nepfaff/steerable-scene-generation-restaurant-low-clutter \
dataset.model_path_vec_len=28 \
dataset.max_num_objects_per_scene=107 \
algorithm=scene_diffuser_flux_transformer \
algorithm.classifier_free_guidance.use=False \
algorithm.ema.use=False \
algorithm.trainer=rl_score \
algorithm.ddpo.use_non_penetration_reward=True \
+algorithm.model.scene_vec_len=256 \
algorithm.noise_schedule.scheduler=ddim \
algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
experiment.training.max_steps=230001 \
experiment.validation.limit_batch=1 \
experiment.validation.val_every_n_step=50 \
algorithm.ddpo.ddpm_reg_weight=200.0 \
experiment.reset_lr_scheduler=True \
experiment.training.lr=1e-6 \
experiment.lr_scheduler.num_warmup_steps=250 \
algorithm.ddpo.batch_size=32 \
experiment.training.checkpointing.every_n_train_steps=500 \
algorithm.num_additional_tokens_for_sampling=20 \
algorithm.ddpo.n_timesteps_to_sample=100 \
experiment.find_unused_parameters=True



---
# custom dataset


## training


python main.py +name=first algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.weight=0



PYTHONPATH=. python main.py +name=second \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset._name=custom_scene \
dataset.max_num_objects_per_scene=12 \
algorithm=scene_diffuser_flux_transformer \
algorithm.trainer=ddpm \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.use_floor=True \
algorithm.classifier_free_guidance.weight=1.5 \
algorithm.custom.loss=true

PYTHONPATH=. python /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/steerable_scene_generation/datasets/custom_scene/custom_scene.py \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset._name=custom_scene \
dataset.max_num_objects_per_scene=12 \
algorithm=scene_diffuser_flux_transformer \
algorithm.trainer=ddpm \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.weight=0 \
algorithm.custom.loss=true



dataset.model_path_vec_len=62 \
# model_path_vec_len = dim of ohe of class labels in steering code base


## RAN for overfit
PYTHONPATH=. python main.py +name=genz dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene dataset.max_num_objects_per_scene=12 algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true resume=bgdrozky \
experiment.training.max_steps=2e6


PYTHONPATH=. python main.py +name=test_diffuscene dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene dataset.max_num_objects_per_scene=12 algorithm=scene_diffuser_diffuscene algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true experiment.training.checkpointing.every_n_train_steps=1

PYTHONPATH=. python main.py +name=continuous_midiffusion_baseline dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene dataset.max_num_objects_per_scene=12 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true experiment.training.checkpointing.every_n_train_steps=1

enotaatr

no iou
/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-26/06-11-54/sampled_scenes_results.pkl

PYTHONPATH=. python scripts/custom_sample_and_render.py \
load=bgdrozky \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=1000 \
+num_scenes=256 \
algorithm=scene_diffuser_flux_transformer \
algorithm.trainer=ddpm \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.weight=0 \
algorithm.custom.loss=true algorithm.ema.use=False \
algorithm.noise_schedule.scheduler=ddpm \
algorithm.noise_schedule.ddim.num_inference_timesteps=150


iou
/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-26/06-12-27/sampled_scenes_results.pkl

PYTHONPATH=. python scripts/custom_sample_and_render.py \
load=t2kp99eo \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=256 \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.loss.use_iou_regularization=True resume=t2kp99eo


rl
PYTHONPATH=. python scripts/custom_sample_and_render.py \
load=ykwgctpr \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=256 \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=False

# num_scenes = batch size during inference, it predicts for each scene in test set


python scripts/download_checkpoint.py --run_id bgdrozky --entity 078bct021-ashok-d --project 3dhope_rl


PYTHONPATH=. python main.py +name=test_diffuscene \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset._name=custom_scene \
dataset.max_num_objects_per_scene=12 \
algorithm=scene_diffuser_diffuscene \
algorithm.trainer=ddpm \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.weight=0 \
algorithm.custom.loss=true \
experiment.training.checkpointing.every_n_train_steps=1 \
experiment.training.checkpointing.save_last=True \
experiment.training.max_steps=5


PYTHONPATH=. python -u main.py +name=first_rl \
    resume=1m3m432c \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_diffuscene\
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=False \
    algorithm.trainer=rl_score \
    algorithm.ddpo.use_iou_reward=False \
    algorithm.ddpo.use_has_sofa_reward=True \
    algorithm.ddpo.use_object_number_reward=False \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    experiment.training.max_steps=2e6 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=200.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=2 \
    experiment.training.checkpointing.every_n_train_steps=500 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.ddpo.n_timesteps_to_sample=100 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    experiment.training.precision=bf16-mixed

PYTHONPATH=. python main.py +name=diffuscene_baseline dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene dataset.max_num_objects_per_scene=12 algorithm=scene_diffuser_diffuscene algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true \
experiment.training.max_steps=1e6 resume=jfgw3io6


PYTHONPATH=. python main.py +name=livingroom_flux_baseline dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true \
experiment.training.max_steps=1e6
---
/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/latest-run/checkpoints/I=4999-step=10000.ckpt

---

in /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/configurations/algorithm/scene_diffuser_base.yaml change this to use our loss (because their scene repr does not have size, objfeat)

custom:
  loss: true # if false default loss
  obj_vec_len: 62 # with objfeat32
  obj_diff_vec_len: 62



[Ashok] bounds sizes (array([0.03998289, 0.02000002, 0.012772  ], dtype=float32), array([2.8682  , 1.770065, 1.698315], dtype=float32)), translations (array([-2.7625005,  0.045    , -2.75275  ], dtype=float32), array([2.7784417, 3.6248395, 2.8185427], dtype=float32))
<!-- /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-25/06-05-07/sampled_scenes_results.pkl -->

<!--  Render Results -->
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-14/11-08-41/sampled_scenes_results.pkl --no_texture --without_floor

python ../ThreedFront/scripts/render_results.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-14/13-32-38/sampled_scenes_results.pkl --no_texture




srun --partition=debug --gres=gpu:a6000:1 --time=04:00:00 --pty bash


rsync -avzP /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/checkpoints/078bct021-ashok-d/3dhope_rl/bgdrozky/model.ckpt insait:/home/pramish_paudel/3dhope_data/model.ckpt




---
{0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp', 4: 'chair', 5: 'children_cabinet', 6: 'coffee_table', 7: 'desk', 8: 'double_bed', 9: 'dressing_chair', 10: 'dressing_table', 11: 'kids_bed', 12: 'nightstand', 13: 'pendant_lamp', 14: 'shelf', 15: 'single_bed', 16: 'sofa', 17: 'stool', 18: 'table', 19: 'tv_stand', 20: 'wardrobe'}

beds: double_bed, single_bed, kids_bed = 8, 15, 11

before rl bgdrozky

[Ashok] number of scenes with 2 beds 1 out of 162                                                                                                               
[Ashok] number of scenes with sofa 14 out of 162

after rl finetuning




PYTHONPATH=. python scripts/reward_custom_sample_and_render.py \
load=jfgw3io6 \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=14 \
+num_scenes=256 \
algorithm=scene_diffuser_diffuscene algorithm.trainer=rl_score algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 algorithm.ddpo.n_timesteps_to_sample=100 experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True debug=True



iou reward
PYTHONPATH=. python scripts/reward_custom_sample_and_render.py \
load=ykwgctpr \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=14 \
+num_scenes=256 \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=rl_score algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 algorithm.ddpo.n_timesteps_to_sample=100 experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=False debug=True



---
PYTHONPATH=. python -u main.py +name=first_rl \
    load=bgdrozky \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_flux_transformer \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    algorithm.ddpo.use_iou_reward=False \
    algorithm.ddpo.use_has_sofa_reward=False \
    algorithm.ddpo.use_composite_reward=True \
    algorithm.ddpo.use_composite_plus_task_reward=False \
    algorithm.ddpo.use_object_number_reward=False \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    experiment.training.max_steps=2e6 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=200.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=2 \
    experiment.training.checkpointing.every_n_train_steps=500 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.ddpo.n_timesteps_to_sample=100 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0

<!-- Dynamic RL -->

python take_user_instruction.py dataset=custom_scene algorithm=scene_diffuser_flux_transformer

PYTHONPATH=. python -u main.py +name=test_dynamic_rl \
    load=bgdrozky \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_flux_transformer \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    experiment.training.max_steps=2e6 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=200.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=2 \
    experiment.training.checkpointing.every_n_train_steps=500 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.ddpo.n_timesteps_to_sample=100 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    algorithm.ddpo.dynamic_constraint_rewards.use=True


PYTHONPATH=. python scripts/reward_custom_sample_and_render.py \
load=bgdrozky \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=256 \
algorithm.num_additional_tokens_for_sampling=0 \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=rl_score algorithm.noise_schedule.scheduler=ddpm algorithm.noise_schedule.ddim.num_inference_timesteps=150 algorithm.ddpo.n_timesteps_to_sample=100 experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True 

baseline with composite reward

PYTHONPATH=. python scripts/reward_custom_sample_and_render.py \
load=juy0jvto \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=256 \
algorithm.num_additional_tokens_for_sampling=0 \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=rl_score algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 algorithm.ddpo.n_timesteps_to_sample=100 experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True 


baseline with ddim
PYTHONPATH=. python scripts/reward_custom_sample_and_render.py \
load=juy0jvto \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=256 \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=rl_score algorithm.noise_schedule.scheduler=ddpm algorithm.noise_schedule.ddim.num_inference_timesteps=150 algorithm.ddpo.n_timesteps_to_sample=100 experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True debug=True











---
        descale_trans = descale_to_origin(
            positions,
            torch.tensor([-2.7625005, 0.045, -2.75275], device=device),
            torch.tensor([2.7784417, 3.6248395, 2.8185427], device=device),
        )
        descale_sizes = descale_to_origin(
            sizes,
            torch.tensor([0.03998289, 0.02000002, 0.012772], device=device),
            torch.tensor([2.8682, 1.770065, 1.698315], device=device),
        )
{0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp', 4: 'chair', 5: 'children_cabinet', 6: 'coffee_table', 7: 'desk', 8: 'double_bed', 9: 'dressing_chair', 10: 'dressing_table', 11: 'kids_bed', 12: 'nightstand', 13: 'pendant_lamp', 14: 'shelf', 15: 'single_bed', 16: 'sofa', 17: 'stool', 18: 'table', 19: 'tv_stand', 20: 'wardrobe'}

    def descale_to_origin( x, minimum, maximum):
        """
        x shape : BxNx3
        minimum, maximum shape: 3
        """
        x = (x + 1) / 2
        x = x * (maximum - minimum)[None, None, :] + minimum[None, None, :]
        return x


EMA is updated every training step



PYTHONPATH=. python main.py +name=livingroom_flux_baseline dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 \
dataset._name=custom_scene \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true \
experiment.training.max_steps=1e6 \
dataset.data.dataset_directory=livingroom \
dataset.data.annotation_file=livingroom_threed_front_splits.csv \
dataset.data.room_type=livingroom


PYTHONPATH=. python main.py +name=livingroom_flux_baseline dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 \
dataset._name=custom_scene \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true \
experiment.training.max_steps=1e6 \
dataset.data.dataset_directory=bedroom \
dataset.data.annotation_file=bedroom_threed_front_splits_original.csv \
dataset.data.room_type=bedroom


PYTHONPATH=. python main.py +name=flux_transformer_floor_cond \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset._name=custom_scene \
dataset.max_num_objects_per_scene=12 \
algorithm=scene_diffuser_flux_transformer \
algorithm.trainer=ddpm \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.use_floor=True \
algorithm.classifier_free_guidance.weight=-1 \
algorithm.custom.loss=true



<!-- Overfit -->

PYTHONPATH=. python main.py +name=flux_transformer_floor_cond \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset._name=custom_scene \
dataset.max_num_objects_per_scene=12 \
algorithm=scene_diffuser_flux_transformer \
algorithm.trainer=ddpm \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.use_floor=True \
algorithm.classifier_free_guidance.weight=0.7 \
algorithm.custom.loss=true \
dataset.training.splits=["overfit"] \
dataset.validation.splits=["overfit"]

PYTHONPATH=. python scripts/custom_sample_and_render.py \
load=q4d3nkdd \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset._name=custom_scene \
dataset.max_num_objects_per_scene=12 \
+num_scenes=256 \
algorithm=scene_diffuser_flux_transformer \
algorithm.trainer=ddpm \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.use_floor=True \
algorithm.classifier_free_guidance.weight=1.0 algorithm.custom.loss=true \
dataset.training.splits=["overfit"] \
dataset.validation.splits=["overfit"]

PYTHONPATH=. python scripts/reward_custom_sample_and_render.py \
load=q4d3nkdd \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=5 \
experiment.test.batch_size=32 \
algorithm.num_additional_tokens_for_sampling=0 \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm algorithm.noise_schedule.scheduler=ddpm  experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.classifier_free_guidance.use_floor=True dataset.training.splits=["overfit"] dataset.validation.splits=["overfit"]





floor cond flux baseline

PYTHONPATH=. python scripts/reward_custom_sample_and_render.py \
load=eviaimru \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=5 \
experiment.test.batch_size=32 \
algorithm.num_additional_tokens_for_sampling=0 \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm algorithm.noise_schedule.scheduler=ddpm  experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.classifier_free_guidance.use_floor=True


# Sample 256 scenes from test split with batch size from config
PYTHONPATH=. python scripts/reward_custom_sample_and_render.py \
load=q4d3nkdd \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=5 \
experiment.test.batch_size=32 \
algorithm=scene_diffuser_flux_transformer \
algorithm.trainer=ddpm \
algorithm.noise_schedule.scheduler=ddpm \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.use_floor=True \
algorithm.classifier_free_guidance.weight=0 \
algorithm.custom.loss=true \
algorithm.ema.use=True

diffuscene
PYTHONPATH=. python scripts/custom_sample_and_render.py \
load=jfgw3io6 \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=256 \
algorithm.num_additional_tokens_for_sampling=0 \
algorithm=scene_diffuser_diffuscene algorithm.trainer=ddpm algorithm.noise_schedule.scheduler=ddpm  experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.classifier_free_guidance.use_floor=False experiment.test.batch_size=196


continuous mi
PYTHONPATH=. python scripts/custom_sample_and_render.py \
load=pfksynuz \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=256 \
algorithm.num_additional_tokens_for_sampling=0 \
algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm algorithm.noise_schedule.scheduler=ddpm  experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.classifier_free_guidance.use_floor=False experiment.test.batch_size=196
















PYTHONPATH=. python -u main.py +name=diffuscene_baseline \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_diffuscene \
    algorithm.trainer=ddpm \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.loss=true \
    experiment.training.max_steps=1e6 \
    resume=jfgw3io6 \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0

PYTHONPATH=. python -u main.py +name=continuous_midiffusion_baseline \
    resume=pfksynuz \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0

















# evals
edit pkl file path and run
./batch_eval.sh
