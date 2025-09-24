from typing import Dict, Type

from steerable_scene_generation.algorithms.scene_diffusion import (
    SceneDiffuserTrainerDDPM,
    SceneDiffuserTrainerPPO,
    SceneDiffuserTrainerScore,
    create_scene_diffuser_diffuscene,
    create_scene_diffuser_flux_transformer,
)
from steerable_scene_generation.datasets.custom_scene import CustomDataset
from steerable_scene_generation.experiments.scene_diffusion import (
    SceneDiffusionExperiment,
)


class CustomSceneDiffusionExperiment(SceneDiffusionExperiment):
    """A scene diffusion experiment using a custom dataset."""

    # Inherit all the algorithm factories and trainers from SceneDiffusionExperiment

    # Override the compatible_datasets to include our custom dataset
    # compatible_datasets = dict(
    #     scene=CustomSceneDataset,  # The original key "scene" still points to our custom dataset
    #     custom_scene=CustomSceneDataset,  # Add a direct "custom_scene" key
    # )
    compatible_datasets = dict(
        scene=CustomDataset,  # The original key "scene" still points to our custom dataset
        custom_scene=CustomDataset,  # Add a direct "custom_scene" key
    )
