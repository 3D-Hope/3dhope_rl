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
algorithm.classifier_free_guidance.weight=0.5 \
algorithm.classifier_free_guidance.sampling.labels="a scene with 6 chairs and 1 table."

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
python main.py +name=first algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True

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
algorithm.ddpo.use_object_number_reward=True \
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
algorithm.classifier_free_guidance.weight=0 \
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
PYTHONPATH=. python main.py +name=test dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene dataset.max_num_objects_per_scene=12 algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true resume=bgdrozky


PYTHONPATH=. python main.py +name=train_with_iou_loss dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene dataset.max_num_objects_per_scene=12 algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.loss.use_iou_regularization=True


no iou
/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-26/06-11-54/sampled_scenes_results.pkl

PYTHONPATH=. python scripts/custom_sample_and_render.py \
load=bgdrozky \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=256 \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true

iou
/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-26/06-12-27/sampled_scenes_results.pkl
PYTHONPATH=. python scripts/custom_sample_and_render.py \
load=t2kp99eo \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
+num_scenes=256 \
algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true


# num_scenes = batch size during inference, it predicts for each scene in test set




PYTHONPATH=. python main.py +name=first_rl \
load=bgdrozky \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
algorithm=scene_diffuser_flux_transformer \
algorithm.classifier_free_guidance.use=False \
algorithm.ema.use=False \
algorithm.trainer=rl_score \
algorithm.ddpo.use_object_number_reward=True \
algorithm.noise_schedule.scheduler=ddim \
algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
experiment.training.max_steps=2e6 \
experiment.validation.limit_batch=1 \
experiment.validation.val_every_n_step=50 \
algorithm.ddpo.ddpm_reg_weight=200.0 \
experiment.reset_lr_scheduler=True \
experiment.training.lr=1e-6 \
experiment.lr_scheduler.num_warmup_steps=250 \
algorithm.ddpo.batch_size=32 \
experiment.training.checkpointing.every_n_train_steps=500 \
algorithm.num_additional_tokens_for_sampling=2 \
algorithm.ddpo.n_timesteps_to_sample=100 \
experiment.find_unused_parameters=True \
algorithm.custom.loss=true


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
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-25/11-37-55/sampled_scenes_results.pkl --no_texture --without_floor

