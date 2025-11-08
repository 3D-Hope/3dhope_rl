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
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-27/17-57-19/sampled_scenes_results.pkl --no_texture --without_floor

python ../ThreedFront/scripts/render_results.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-31/19-40-48/sampled_scenes_results.pkl --no_texture


zip -r bedroom_accessibility_cache.zip bedroom_accessibility_cache/
zip -r bedroom_sdf_cache.zip bedroom_sdf_cache/



srun --partition=debug --gres=gpu:a6000:1 --time=04:00:00 --pty bash


rsync -avzP /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/checkpoints/078bct021-ashok-d/3dhope_rl/bgdrozky/model.ckpt insait:/home/pramish_paudel/3dhope_data/model.ckpt



rsync -avzP /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/accessibility_cache.zip insait:/home/pramish_paudel/3dhope_data/accessibility_cache.zip

rsync -avzP /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/livingroom.zip insait:/home/pramish_paudel/3dhope_data/  




rsync -avzP /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_accessibility_cache.zip insait:/home/pramish_paudel/3dhope_data/ 

rsync -avzP /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_sdf_cache.zip insait:/home/pramish_paudel/3dhope_data/  
---
{0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp', 4: 'chair', 5: 'children_cabinet', 6: 'coffee_table', 7: 'desk', 8: 'double_bed', 9: 'dressing_chair', 10: 'dressing_table', 11: 'kids_bed', 12: 'nightstand', 13: 'pendant_lamp', 14: 'shelf', 15: 'single_bed', 16: 'sofa', 17: 'stool', 18: 'table', 19: 'tv_stand', 20: 'wardrobe'}

beds: double_bed, single_bed, kids_bed = 8, 15, 11

before rl bgdrozky

[Ashok] number of scenes with 2 beds 1 out of 162                                                                                                               
[Ashok] number of scenes with sofa 14 out of 162

after rl finetuning




PYTHONPATH=. python scripts/custom_sample_and_render.py \
load=ykwgctpr \
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

python dynamic_constraint_rewards/take_user_instruction.py dataset=custom_scene algorithm=scene_diffuser_midiffusion

<!-- SAUGAT DO THIS -->
python scripts/custom_sample_and_render.py load=rrudae6n dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=10000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150



python scripts/custom_sample_and_render.py load=fhfnf4xi dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=10000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 algorithm.custom.objfeat_dim=0 algorithm.custom.old=True


python scripts/custom_sample_and_render.py load=z708k43v dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150



python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-28/11-54-27/sampled_scenes_results.pkl --no_texture

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-28/11-56-31/sampled_scenes_results.pkl


PYTHONPATH=. python -u main.py +name=test_dynamic_rl \
    load=qbyilta9 \
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
    algorithm.ddpo.dynamic_constraint_rewards.use=True \
    algorithm.ddpo.dynamic_constraint_rewards.universal_weight=2


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
load=ykwgctpr \
dataset=custom_scene \
algorithm.trainer=rl_score \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=100 \
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

PYTHONPATH=. python scripts/custom_sample_and_render.py \
load=eviaimru \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=16 \
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




## dynamic reward sample

PYTHONPATH=. python scripts/reward_custom_sample_and_render.py \
load=ykwgctpr \
dataset=custom_scene \
algorithm.trainer=rl_score \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=100 \
algorithm.num_additional_tokens_for_sampling=0 \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=rl_score algorithm.noise_schedule.scheduler=ddpm algorithm.noise_schedule.ddim.num_inference_timesteps=150 algorithm.ddpo.n_timesteps_to_sample=100 experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True experiment.test.batch_size=196 algorithm.ema.use=False

PYTHONPATH=. python scripts/custom_sample_and_render.py \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    +num_scenes=50 \
    algorithm=scene_diffuser_flux_transformer \
    algorithm.trainer=rl_score \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.custom.loss=true \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    load=xn1h20rz \
    checkpoint_version=v83 \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.ema.use=True \
    experiment.test.batch_size=196 \
    algorithm.classifier_free_guidance.use_floor=False

    
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-19/14-10-15/sampled_scenes_results.pkl --no_texture --without_floor




PYTHONPATH=. python scripts/custom_sample_and_render.py \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=1000 \
algorithm=scene_diffuser_flux_transformer \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.weight=0 \
algorithm.num_additional_tokens_for_sampling=0 \
algorithm.custom.loss=true \
algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
algorithm.trainer=ddpo \
load=qbyilta9 \
algorithm.noise_schedule.scheduler=ddpm \
algorithm.ema.use=True \
experiment.test.batch_size=196 \
algorithm.classifier_free_guidance.use_floor=False



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





PYTHONPATH=. python -u main.py +name=universal_reward_continuous_mi \
    load=pfksynuz \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
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
    algorithm.ddpo.batch_size=8 \
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
    algorithm.ddpo.use_universal_reward=True \
    experiment.training.precision=bf16-mixed \
    experiment.validation.precision=bf16-mixed \
    experiment.test.precision=bf16-mixed \
    experiment.matmul_precision=medium










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
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0



PYTHONPATH=. python -u main.py +name=continuous_midiffusion_baseline     dataset=custom_scene     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json     dataset._name=custom_scene     dataset.max_num_objects_per_scene=12     algorithm=scene_diffuser_midiffusion     algorithm.trainer=ddpm     experiment.find_unused_parameters=True     algorithm.classifier_free_guidance.use=False     algorithm.classifier_free_guidance.use_floor=True     algorithm.classifier_free_guidance.weight=0     algorithm.custom.loss=true     algorithm.validation.num_samples_to_render=0     algorithm.validation.num_samples_to_visualize=0     algorithm.validation.num_directives_to_generate=0     algorithm.test.num_samples_to_render=0     algorithm.test.num_samples_to_visualize=0     algorithm.test.num_directives_to_generate=0     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0













# evals
edit pkl file path and run
./batch_eval.sh





--- 
# test dynamic reward

instruction: design a bedroom for a person who loves watching tv

{   "constraints": [     "the room must contain a bed and a tv_stand",     "the bed should face the tv_stand directly",     "there should be at least 2.0 meters distance between the bed and tv_stand for comfortable viewing"   ] }



# best flux baseline with rl non pen
juy0jvto


all info from boxes.npz
['uids', 'jids', 'scene_id', 'scene_uid', 'scene_type', 'json_path', 'room_layout', 'floor_plan_vertices', 'floor_plan_faces', 'floor_plan_centroid', 'class_labels', 'translations', 'sizes', 'angles', 'floor_plan_ordered_corners', 'floor_plan_boundary_points_normals']





PYTHONPATH=. python -u main.py +name=test_physcene_rl \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion\
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=False \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    experiment.training.max_steps=2e6 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=50.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=4 \
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
    experiment.training.precision=bf16-mixed \
    algorithm.ddpo.use_universal_reward=True \
    algorithm.ddpo.universal_reward.use_physcene_reward=True \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=1.0 \
    load=rrudae6n



    <!-- Physcene Metrics -->
python scripts/physcene_metrics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-04/19-02-41/sampled_scenes_results.pkl


<!-- LIVING ROOM -->

<!-- Training Midiffusion -->

PYTHONPATH=. python -u main.py +name=continuous_midiffusion_baseline_livingroom \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=21 \
    dataset.model_path_vec_len=25 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.loss=true \

    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.dataset_directory=livingroom \
    dataset.data.room_type=livingroom \
    algorithm.custom.objfeat_dim=0 \
    algorithm.custom.num_classes=25 \
    algorithm.custom.obj_vec_len=33 \
    algorithm.custom.obj_diff_vec_len=33 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
    dataset.data.annotation_file=livingroom_threed_front_splits.csv \
    experiment.training.checkpointing.every_n_train_steps=1000


PYTHONPATH=. python -u main.py +name=continuous_midiffusion_baseline_bedroom \
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
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    








python scripts/custom_sample_and_render.py load=olisydpx dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=21 dataset.model_path_vec_len=25 +num_scenes=10 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.data.dataset_directory=livingroom dataset.data.room_type=livingroom dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm dataset.data.annotation_file=livingroom_threed_front_splits.csv dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data

python scripts/custom_sample_and_render.py load=omzk7l7k dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=10 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data

python ../ThreedFront/scripts/render_results.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-30/18-08-32/sampled_scenes_results.pkl --no_texture

    



<!-- inpainting -->


python scripts/inpaint.py load=rrudae6n dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30 algorithm.custom.old=true algorithm.custom.objfeat_dim=0

python scripts/custom_sample_and_render.py load=rrudae6n dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=1 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30




python scripts/inpaint_and_render.py load=rrudae6n dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=10 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30

python scripts/inpaint_two_fixed_single_beds.py load=rrudae6n dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=5 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30



python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-02/19-31-02/sampled_scenes_results.pkl \
    --background 1,1,1,1 \
    --with_orthographic_projection \
    --window_size 1024,1024 \
    --retrieve_by_size



python scripts/python scripts/rearrange_complete_scenes.py load=rrudae6n dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=10 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=False algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30



<!-- train with objfeat -->
PYTHONPATH=. python -u main.py +name=continuous_mi_bedroom_floor_obj32 \
    dataset=custom_scene \
    dataset.data.dataset_directory=bedrooms_objfeats_32_64 \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    algorithm.custom.objfeat_dim=32 \
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



python scripts/custom_sample_and_render.py +num_scenes=1000 algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30 \
    load=ndt0yc0l \
    dataset=custom_scene \
    dataset.data.dataset_directory=bedroom \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    algorithm.custom.objfeat_dim=32 \
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
    dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data

python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-03/13-25-12/sampled_scenes_results.pkl --no_texture --retrieve_by_size

python scripts/custom_sample_and_render.py +num_scenes=32 algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30 \
    load=x6n93gvb \
    dataset=custom_scene \
    dataset.data.dataset_directory=livingroom \
    dataset.data.annotation_file=livingroom_threed_front_splits.csv \
    dataset.data.room_type=livingroom \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=21 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    algorithm.custom.objfeat_dim=32 \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.obj_vec_len=65 \
    algorithm.custom.obj_diff_vec_len=65 \
    algorithm.custom.loss=true \
    algorithm.custom.num_classes=25 \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data


<!-- bedroom objfeats -->
python scripts/custom_sample_and_render.py +num_scenes=1000 algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30 \
    load=ndt0yc0l \
    dataset=custom_scene \
    dataset.data.dataset_directory=bedroom \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    algorithm.custom.objfeat_dim=32 \
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
    dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data

<!-- living room objfeats -->
python scripts/custom_sample_and_render.py +num_scenes=1000 algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30 \
    load=c51e7unm \
    dataset=custom_scene \
    dataset.data.dataset_directory=livingroom \
    dataset.data.annotation_file=livingroom_threed_front_splits.csv \
    dataset.data.room_type=livingroom \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=21 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    algorithm.custom.objfeat_dim=32 \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.obj_vec_len=65 \
    algorithm.custom.obj_diff_vec_len=65 \
    algorithm.custom.loss=true \
    algorithm.custom.num_classes=25 \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data


<!-- No objfeats -->
python scripts/custom_sample_and_render.py +num_scenes=1000 algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30 \
    load=cu8sru1y \
    dataset=custom_scene \
    dataset.data.dataset_directory=livingroom \
    dataset.data.annotation_file=livingroom_threed_front_splits.csv \
    dataset.data.room_type=livingroom \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=21 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    algorithm.custom.objfeat_dim=0 \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.obj_vec_len=65 \
    algorithm.custom.obj_diff_vec_len=65 \
    algorithm.custom.loss=true \
    algorithm.custom.num_classes=25 \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-02/04-28-23/sampled_scenes_results.pkl --no_texture

PYTHONPATH=. python -u main.py +name=continuous_mi_livingroom_floor_obj32 \
    dataset=custom_scene \
    dataset.data.dataset_directory=livingrooms_objfeats_32_64 \
    dataset.data.annotation_file=livingroom_threed_front_splits.csv \
    dataset.data.room_type=livingroom \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=21 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.objfeat_dim=32 \
    algorithm.custom.obj_vec_len=65 \
    algorithm.custom.obj_diff_vec_len=65 \
    algorithm.custom.loss=true \
    algorithm.custom.num_classes=25 \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm




python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-02/04-01-54-better/sampled_scenes_results.pkl     --background 1,1,1,1     --with_orthographic_projection     --window_size 1920,1920     --retrieve_by_size --ortho_zoom 0.8

python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-02/10-02-44/sampled_scenes_results.pkl    --background 1,1,1,1     --with_orthographic_projection     --window_size 1920,1920     --retrieve_by_size --ortho_zoom 0.8


python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-02/10-02-44/sampled_scenes_results.pkl    --background 1,1,1,1     --with_orthographic_projection     --window_size 1920,1920     --retrieve_by_size --ortho_zoom 0.8


<!-- new dynamic rl  -->
python test.py dataset=custom_scene algorithm=scene_diffuser_midiffusion algorithm.custom.old=true dataset.data.room_type=livingroom


python scripts/inpaint.py load=rrudae6n dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 algorithm.custom.old=True algorithm.predict.inpaint_masks={'ceiling_lamp': 4}


PYTHONPATH=. python -u main.py +name=test_inpaint_rl \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion\
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=False \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    experiment.training.max_steps=2e6 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=50.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=4 \
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
    experiment.training.precision=bf16-mixed \
    algorithm.ddpo.use_universal_reward=True \
    algorithm.ddpo.universal_reward.use_physcene_reward=True \
    algorithm.classifier_free_guidance.use_floor=True \
    load=rrudae6n \
    algorithm.classifier_free_guidance.weight=1.0 \
    algorithm.custom.old=True \
    algorithm.predict.inpaint_masks='{ceiling_lamp: 4}' \
    algorithm.ddpo.use_inpaint=True \




python scripts/inpaint.py load=cu8sru1y dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 algorithm.predict.inpaint_masks='{dining_chair: 10}' dataset.data.room_type=livingroom dataset.data.dataset_directory=livingroom dataset.data.annotation_file=livingroom_threed_front_splits.csv dataset.max_num_objects_per_scene=21 algorithm.custom.objfeat_dim=0 algorithm.custom.obj_vec_len=65 algorithm.custom.obj_diff_vec_len=65 algorithm.custom.num_classes=25 algorithm.custom.old=True dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm









---



python scripts/inpaint.py load=cu8sru1y dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150  dataset.data.room_type=livingroom dataset.model_path_vec_len=30 dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData dataset.data.dataset_directory=livingroom dataset.data.annotation_file=livingroom_threed_front_splits.csv dataset.max_num_objects_per_scene=21 algorithm.custom.objfeat_dim=0 algorithm.custom.obj_vec_len=65 algorithm.custom.obj_diff_vec_len=65 algorithm.custom.num_classes=25 dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data \
    algorithm.predict.inpaint_masks='{dining_chair: 10}'

python dynamic_constraint_rewards/get_reward_stats.py load=w0gmpwep dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150  dataset.data.room_type=livingroom dataset.model_path_vec_len=30 dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData dataset.data.dataset_directory=livingroom dataset.data.annotation_file=livingroom_threed_front_splits.csv dataset.max_num_objects_per_scene=21 algorithm.custom.objfeat_dim=0 algorithm.custom.obj_vec_len=65 algorithm.custom.obj_diff_vec_len=65 algorithm.custom.num_classes=25 dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data \
    dataset.sdf_cache_dir=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/living_sdf_cache \
    dataset.accessibility_cache_dir=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/living_accessibility_cache
    



--- 

Agentic

python dynamic_constraint_rewards/run_llm_pipeline.py dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150  dataset.data.room_type=livingroom dataset.model_path_vec_len=30 dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData dataset.data.dataset_directory=livingroom dataset.data.annotation_file=livingroom_threed_front_splits.csv dataset.max_num_objects_per_scene=21 algorithm.custom.objfeat_dim=0 algorithm.custom.obj_vec_len=65 algorithm.custom.obj_diff_vec_len=65 algorithm.custom.num_classes=25 dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm algorithm.validation.num_samples_to_render=0 algorithm.validation.num_samples_to_visualize=0 algorithm.validation.num_directives_to_generate=0 algorithm.test.num_samples_to_render=0 algorithm.test.num_samples_to_visualize=0 algorithm.test.num_directives_to_generate=0 algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0




python dynamic_constraint_rewards/get_reward_stats.py load=cu8sru1y dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene +num_scenes=5 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150  dataset.data.room_type=livingroom dataset.model_path_vec_len=30 dataset.data.dataset_directory=livingroom dataset.data.annotation_file=livingroom_threed_front_splits.csv dataset.max_num_objects_per_scene=21 algorithm.custom.objfeat_dim=0 algorithm.custom.obj_vec_len=65 algorithm.custom.obj_diff_vec_len=65 algorithm.custom.num_classes=25 dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm algorithm.validation.num_samples_to_render=0 algorithm.validation.num_samples_to_visualize=0 algorithm.validation.num_directives_to_generate=0 algorithm.test.num_samples_to_render=0 algorithm.test.num_samples_to_visualize=0 algorithm.test.num_directives_to_generate=0 algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0







scp ajad@10.144.126.219:/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_sdf_cache.zip pulchowk@113.199.192.32:/home/pulchowk/3dhope_rl
scp ajad@10.144.126.219:/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_accessibility_cache.zip pulchowk@113.199.192.32:/home/pulchowk/3dhope_rl


PYTHONPATH=. python -u main.py +name=dynamic_bedroom \
    load=rrudae6n \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    experiment.training.max_steps=1020000 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=50.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=4 \
    experiment.training.checkpointing.every_n_train_steps=500 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.ddpo.n_timesteps_to_sample=100 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=True \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    algorithm.ddpo.dynamic_constraint_rewards.use=True \
    experiment.training.precision=bf16-mixed \
    experiment.validation.precision=bf16-mixed \
    experiment.test.precision=bf16-mixed \
    experiment.matmul_precision=medium \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.ddpo.dynamic_constraint_rewards.stats_path=./dynamic_constraint_rewards/stats.json \
    dataset.sdf_cache_dir=./bedroom_sdf_cache/ \
    dataset.accessibility_cache_dir=./bedroom_accessibility_cache/ \
    algorithm.custom.num_classes=22 \
    algorithm.custom.objfeat_dim=0 \
    algorithm.custom.obj_vec_len=30 \
    algorithm.custom.obj_diff_vec_len=30 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
    dataset.data.dataset_directory=bedroom \
    dataset.data.annotation_file=bedroom_threed_front_splits_original.csv \
    dataset.data.room_type=bedroom \
    algorithm.custom.old=True

<!-- Get Stats Bedroom -->
python dynamic_constraint_rewards/get_reward_stats.py load=rrudae6n \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/ \
    dataset.data.path_to_dataset_files=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/dataset_files/\
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    experiment.training.max_steps=1020000 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=50.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=128 \
    experiment.training.checkpointing.every_n_train_steps=500 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.ddpo.n_timesteps_to_sample=100 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=True \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    algorithm.ddpo.dynamic_constraint_rewards.use=True \
    experiment.training.precision=bf16-mixed \
    experiment.validation.precision=bf16-mixed \
    experiment.test.precision=bf16-mixed \
    experiment.matmul_precision=medium \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.ddpo.dynamic_constraint_rewards.stats_path=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/dynamic_constraint_rewards/stats.json \
    dataset.sdf_cache_dir=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_sdf_cache/ \
    dataset.accessibility_cache_dir=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_accessibility_cache/ \
    algorithm.custom.num_classes=22 \
    algorithm.custom.objfeat_dim=0 \
    algorithm.custom.obj_vec_len=30 \
    algorithm.custom.obj_diff_vec_len=30 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
    dataset.data.dataset_directory=bedroom \
    dataset.data.annotation_file=bedroom_threed_front_splits_original.csv \
    algorithm.custom.old=True \
    dataset.data.room_type=bedroom


python scripts/custom_sample_and_render.py +num_scenes=1000 algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30 \
    load=cu8sru1y \
    dataset=custom_scene \
    dataset.data.dataset_directory=livingroom \
    dataset.data.annotation_file=livingroom_threed_front_splits.csv \
    dataset.data.room_type=livingroom \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=21 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    algorithm.custom.objfeat_dim=0 \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.obj_vec_len=65 \
    algorithm.custom.obj_diff_vec_len=65 \
    algorithm.custom.loss=true \
    algorithm.custom.num_classes=25 \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm




<!-- universal bedroom -->
python scripts/custom_sample_and_render.py load=fhfnf4xi dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 algorithm.custom.objfeat_dim=0 algorithm.custom.old=True

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-07/17-08-40/sampled_scenes_results.pkl --no_texture --retrieve_by_size

<!-- agentic ceiling lamp -->
python scripts/custom_sample_and_render.py load=w0gmpwep dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=10 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 algorithm.custom.objfeat_dim=0 algorithm.custom.old=True

python scripts/inpaint.py load=w0gmpwep dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30 algorithm.custom.old=true algorithm.custom.objfeat_dim=0 algorithm.predict.inpaint_masks='{ceiling_lamp: 4}' algorithm.ddpo.use_inpaint=True

python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-07/17-12-03/sampled_scenes_results.pkl --no_texture --retrieve_by_size





---
python dynamic_constraint_rewards/get_reward_stats.py load=fhfnf4xi dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150  dataset.data.room_type=bedroom dataset.model_path_vec_len=30 dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData dataset.data.dataset_directory=bedroom dataset.data.annotation_file=bedroom_threed_front_splits_original.csv dataset.max_num_objects_per_scene=12 algorithm.custom.objfeat_dim=0 algorithm.custom.obj_vec_len=65 algorithm.custom.obj_diff_vec_len=65 algorithm.custom.num_classes=22 dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data \
    dataset.sdf_cache_dir=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_sdf_cache \
    dataset.accessibility_cache_dir=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_accessibility_cache \
    algorithm.ddpo.use_inpaint=True
    

python dynamic_constraint_rewards/run_llm_pipeline.py load=w0gmpwep dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150  dataset.data.room_type=bedroom dataset.model_path_vec_len=30 dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData dataset.data.dataset_directory=bedroom dataset.data.annotation_file=bedroom_threed_front_splits_original.csv dataset.max_num_objects_per_scene=12 algorithm.custom.objfeat_dim=0 algorithm.custom.obj_vec_len=65 algorithm.custom.obj_diff_vec_len=65 algorithm.custom.num_classes=22 dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data \
    dataset.sdf_cache_dir=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_sdf_cache \
    dataset.accessibility_cache_dir=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_accessibility_cache \
    algorithm.ddpo.use_inpaint=True \
    algorithm.custom.old=True











python scripts/custom_sample_and_render.py load=rrudae6n dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset.max_num_objects_per_scene=12 +num_scenes=1 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=1 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30




python scripts/custom_sample_and_render.py +num_scenes=1000 algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30 \
    load=cu8sru1y \
    dataset=custom_scene \
    dataset.data.dataset_directory=livingroom \
    dataset.data.annotation_file=livingroom_threed_front_splits.csv \
    dataset.data.room_type=livingroom \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=21 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    algorithm.custom.objfeat_dim=0 \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.obj_vec_len=65 \
    algorithm.custom.obj_diff_vec_len=65 \
    algorithm.custom.loss=true \
    algorithm.custom.num_classes=25 \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
    algorithm.custom.old=False




python steerable_scene_generation/datasets/custom_scene/custom_scene_final.py +num_scenes=1000 algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.model_path_vec_len=30 \
    load=cu8sru1y \
    dataset=custom_scene \
    dataset.data.dataset_directory=livingroom \
    dataset.data.annotation_file=livingroom_threed_front_splits.csv \
    dataset.data.room_type=livingroom \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=21 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    algorithm.custom.objfeat_dim=0 \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.obj_vec_len=65 \
    algorithm.custom.obj_diff_vec_len=65 \
    algorithm.custom.loss=true \
    algorithm.custom.num_classes=25 \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
    algorithm.custom.old=False

<!-- AGENTIC CEILING LAMPS -->
PYTHONPATH=. python -u main.py +name=agentic_4_ceiling_lamps_debug \
    load=w0gmpwep \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    experiment.training.max_steps=1020000 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=50.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=2 \
    experiment.training.checkpointing.every_n_train_steps=500 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.ddpo.n_timesteps_to_sample=100 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=True \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.ddpo.dynamic_constraint_rewards.use=True \
    algorithm.ddpo.dynamic_constraint_rewards.reward_base_dir=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/dynamic_constraint_rewards \
    algorithm.ddpo.dynamic_constraint_rewards.user_query=ceiling_lamps_above_bed_corners \
    dataset.sdf_cache_dir=./bedroom_sdf_cache/ \
    dataset.accessibility_cache_dir=./bedroom_accessibility_cache/ \
    algorithm.custom.num_classes=22 \
    algorithm.custom.objfeat_dim=0 \
    algorithm.custom.obj_vec_len=30 \
    algorithm.custom.obj_diff_vec_len=30 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
    dataset.data.dataset_directory=bedroom \
    dataset.data.annotation_file=bedroom_threed_front_splits_original.csv \
    dataset.data.room_type=bedroom \
    algorithm.custom.old=True \
    algorithm.ddpo.use_inpaint=True

<!-- AGENTIC CLASSROOM -->
PYTHONPATH=. python -u main.py +name=agentic_classroom_debug \
    load=cu8sru1y \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=21 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    experiment.training.max_steps=1020000 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=0.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-5 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=2 \
    experiment.training.checkpointing.every_n_train_steps=500 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.ddpo.n_timesteps_to_sample=100 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=True \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    algorithm.ddpo.dynamic_constraint_rewards.use=True \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.ddpo.dynamic_constraint_rewards.reward_base_dir=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/dynamic_constraint_rewards \
    algorithm.ddpo.dynamic_constraint_rewards.user_query=classroom \
    dataset.sdf_cache_dir=./living_sdf_cache/ \
    dataset.accessibility_cache_dir=./living_accessibility_cache/ \
    algorithm.custom.num_classes=25 \
    algorithm.custom.objfeat_dim=0 \
    algorithm.custom.obj_vec_len=33 \
    algorithm.custom.obj_diff_vec_len=33 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
    dataset.data.dataset_directory=livingroom \
    dataset.data.annotation_file=livingroom_threed_front_splits.csv \
    dataset.data.room_type=livingroom \
    algorithm.ddpo.use_inpaint=True \
    experiment.seed=1