from typing import Dict, Type

from steerable_scene_generation.algorithms.scene_diffusion import (
    create_scene_diffuser_flux_transformer,
    create_scene_diffuser_diffuscene,
    SceneDiffuserTrainerDDPM,
    SceneDiffuserTrainerScore,
    SceneDiffuserTrainerPPO,
)
from steerable_scene_generation.datasets.custom_scene import CustomSceneDataset
from steerable_scene_generation.experiments.scene_diffusion import SceneDiffusionExperiment

class CustomSceneDiffusionExperiment(SceneDiffusionExperiment):
    """A scene diffusion experiment using a custom dataset."""

    # Inherit all the algorithm factories and trainers from SceneDiffusionExperiment
    
    # Override the compatible_datasets to include our custom dataset
    compatible_datasets = dict(
        scene=CustomSceneDataset,  # The original key "scene" still points to our custom dataset
        custom_scene=CustomSceneDataset,  # Add a direct "custom_scene" key
    )