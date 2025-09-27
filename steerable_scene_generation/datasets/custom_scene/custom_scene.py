import json
import os

from typing import Any, Union

import hydra
import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset as TorchDataset

from steerable_scene_generation.datasets.common import BaseDataset
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler
from steerable_scene_generation.utils.omegaconf import register_resolvers

try:
    from threed_front_encoding import get_encoded_dataset
except ImportError:
    from .threed_front_encoding import get_encoded_dataset

from omegaconf import ListConfig


def update_data_file_paths(config_data, config):
    config_data["dataset_directory"] = os.path.join(
        config["data"]["path_to_processed_data"], config_data["dataset_directory"]
    )
    config_data["annotation_file"] = os.path.join(
        config["data"]["path_to_dataset_files"], config_data["annotation_file"]
    )
    return config_data


@hydra.main(
    version_base=None, config_path="../../../configurations", config_name="config"
)
def main(cfg: DictConfig):
    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)
    config = cfg.dataset
    
    # print(f"[DEBUG] config: {config}")
    print(config["data"])
    # import sys; sys.exit()
    
    PATH_TO_PROCESSED_DATA = config["data"]["path_to_processed_data"]
    PATH_TO_DATASET_FILES = config["data"]["path_to_dataset_files"]

    print(config["training"].get("splits", ["train", "val"]))

    train_dataset = get_encoded_dataset(
        update_data_file_paths(config["data"], config),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"]),
        # split=["test"],
        max_length=config["network"]["sample_num_points"],
        include_room_mask=(config["network"]["room_mask_condition"] and \
                            config["feature_extractor"]["name"]=="resnet18")
    )
    print(len(train_dataset))
    print(train_dataset[0])
    print(train_dataset[0].keys())
    print(train_dataset[0]["class_labels"].shape)
    print(train_dataset[0]["translations"].shape)
    print(train_dataset[0]["sizes"].shape)
    print(train_dataset[0]["angles"].shape)
    print(train_dataset[0]["fpbpn"].shape)
    print(train_dataset[0]["length"])

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config["training"].get("batch_size", 128),
    #     num_workers=config["training"].get("n_processes", 1),
    #     collate_fn=train_dataset.collate_fn,
    #     shuffle=True
    # )
    # print("train loader")
    # print(len(train_loader))
    # item = next(iter(train_loader))
    # print(item)
    # print(item.keys())
    # print(item["class_labels"].shape)
    # print(item["translations"].shape)
    # print(item["sizes"].shape)
    # print(item["angles"].shape)
    # print(item["fpbpn"].shape)
    # print(item["length"])

    # train_dataset_custom = CustomSceneDataset(
    #     cfg=config,
    #     split=config["training"].get("splits", ["train", "val"]),
    #     ckpt_path=None
    # )
    # print("test dataset")
    # print(len(train_dataset_custom))
    # item2 = train_dataset_custom[0]
    # print(item2)
    # print(item2.keys())
    # print(item2["scenes"].shape)
    # print(item2["idx"])

    # custom_dataset = CustomDataset(
    #     cfg=config,
    #     split=config["training"].get("splits", ["train", "val"]),
    #     ckpt_path=None,
    # )
    # print("custom dataset")
    # print(len(custom_dataset))
    # item3 = custom_dataset[0]
    # print(item3)
    # print(item3.keys())
    # print(item3["scenes"].shape)
    # print(item3["idx"])

    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    # np.savez(
    #     path_to_bounds,
    #     sizes=train_dataset.bounds["sizes"],
    #     translations=train_dataset.bounds["translations"],
    #     angles=train_dataset.bounds["angles"],
    #     #add objfeats
    #     objfeats=train_dataset.bounds["objfeats"],
    # )

    # validation_dataset = get_encoded_dataset(
    #     update_data_file_paths(config["data"]),
    #     path_to_bounds=path_to_bounds,
    #     augmentations=None,
    #     split=config["validation"].get("splits", ["test"]),
    #     max_length=config["network"]["sample_num_points"],
    #     include_room_mask=(config["network"]["room_mask_condition"] and \
    #                         config["feature_extractor"]["name"]=="resnet18")
    # )


# Actual dataset class code using ThreedFront
class CustomDatasetOld(BaseDataset):
    """
    CustomDataset that uses get_dataset_raw_and_encoded from threedfront to load and encode the dataset.
    """

    def __init__(
        self, cfg: DictConfig, split: str | list, ckpt_path: str | None = None
    ):
        """
        Args:
            cfg: Configuration object
            split: One of "training", "validation", "test"
            ckpt_path: Optional checkpoint path (not used here)
        """
        self.cfg = cfg
        self.split = split

        # Import get_dataset_raw_and_encoded from threedfront_encoding
        try:
            from threed_front_encoding import get_dataset_raw_and_encoded
        except ImportError:
            from .threed_front_encoding import get_dataset_raw_and_encoded

        # Prepare arguments for get_dataset_raw_and_encoded
        data_cfg = cfg["data"] if "data" in cfg else cfg.data
        network_cfg = cfg["network"] if "network" in cfg else cfg.network

        print(f"split: {split}")
        # print(f"type(split): {type(split)}")

        # Use the split argument, or fallback to config
        if split == "training":
            split_names = cfg["training"].get("splits", ["train", "val"])
        elif split == "validation":
            split_names = cfg["validation"].get("splits", ["test"])
        elif type(split) == ListConfig or type(split) == list:
            print(f"split: {split}")
            split_names = split
            print(f"split_names: {split_names}")
        else:
            split_names = [split]

        # Call get_dataset_raw_and_encoded to get raw and encoded datasets
        raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
            update_data_file_paths(data_cfg, cfg),
            # data_cfg,
            split=split_names,
            max_length=network_cfg["sample_num_points"],
            include_room_mask=network_cfg.get("room_mask_condition", True),
        )

        self.raw_dataset = raw_dataset
        self.encoded_dataset = encoded_dataset

        # For compatibility with BaseDataset
        self.hf_dataset = None
        self.normalizer = None  # You may want to set up a normalizer if needed

        # Set up indices for the split
        self.subset_indices = list(range(len(self.encoded_dataset)))
        print(f"[Ashok] found scenes: {len(self.subset_indices)}")

        self.normalizer = self._setup_normalizer(ckpt_path)

        # Create dummy metadata that includes required fields
        self.metadata = {
            "rotation_parametrization": "procrustes",
            "translation_vec_len": 3,
            "model_path_vec_len": cfg.model_path_vec_len,  # e.g., 19
            "max_num_objects_per_scene": cfg.max_num_objects_per_scene,  # e.g., 12
            "model_paths": ["model_" + str(i) for i in range(cfg.model_path_vec_len)],
            "welded_object_model_paths": [],
        }

        # Set up data loaders
        self._setup_data_loaders()

    def __len__(self):
        return len(self.subset_indices)

    def __getitem__(self, idx):
        # Map idx to the corresponding index in the subset
        actual_idx = self.subset_indices[idx]
        item = self.encoded_dataset[actual_idx]

        # Convert to scenes format (class_labels, translations, sizes, angles, fpbpn)
        scenes = np.concatenate(
            [item["class_labels"], item["translations"], item["sizes"], item["angles"]],
            axis=1,
        )
        scenes = torch.from_numpy(scenes).float()

        # Optionally, you can add "idx" to the returned dict for compatibility
        if isinstance(scenes, dict):
            item = dict(scenes)
            item["idx"] = idx
        else:
            item = {"scenes": scenes, "idx": idx}
        return item

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0, pin_memory=False):
        from torch.utils.data import DataLoader, Subset

        subset = Subset(self.encoded_dataset, self.subset_indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=getattr(self.encoded_dataset, "collate_fn", None),
        )

    def normalize_scenes(self, scenes):
        """Normalize scene data using the fitted normalizer."""
        # return self.normalizer.transform(scenes)
        return scenes

    def inverse_normalize_scenes(self, scenes):
        """Inverse normalize scene data using the fitted normalizer."""
        # return self.normalizer.inverse_transform(scenes)
        return scenes

    def _collate_fn(self, batch):
        """
        Custom collate function to handle the batch formatting.
        """
        scenes = torch.stack([item["scenes"] for item in batch])
        indices = [item["idx"] for item in batch]
        return {
            "scenes": scenes,
            "idx": indices,
        }

    def _setup_normalizer(self, ckpt_path):
        """Set up a normalizer for scene vectors."""
        normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)

        # If checkpoint provided, try to load normalizer from it
        if ckpt_path is not None:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if "normalizer_state" in ckpt:
                    normalizer.load_state(ckpt["normalizer_state"])
                    return normalizer
            except Exception as e:
                print(f"Could not load normalizer from checkpoint: {e}")

        # Otherwise, fit a new normalizer on some dummy data with correct feature dimension (62)
        # Breakdown: 22 (class) + 3 (translation) + 3 (size) + 2 (angles) + 32 (objfeats)
        dummy_data = torch.ones(100, 62)
        normalizer.fit(dummy_data)

        return normalizer

    def _get_item_from_index(self, idx):
        """Get an item from the dataset by index."""
        item = self.encoded_dataset[idx]
        return item

    def get_scene_vec_description(self):
        """Return a dummy scene vector description for compatibility."""
        # This would typically come from the metadata, but we'll create it on-the-fly
        from steerable_scene_generation.algorithms.common.dataclasses import (
            RotationParametrization,
            SceneVecDescription,
        )

        # Try to import PyDrake, but handle the case when it's not available
        try:
            from pydrake.all import PackageMap

            package_map = PackageMap()
        except ImportError:
            # If PyDrake is not available, create a dummy PackageMap
            class DummyPackageMap:
                def __init__(self):
                    pass

                def AddMap(self, *args):
                    pass

            package_map = DummyPackageMap()

        return SceneVecDescription(
            drake_package_map=package_map,
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.PROCRUSTES,
            model_paths=self.metadata["model_paths"],
            model_path_vec_len=self.metadata["model_path_vec_len"],
            welded_object_model_paths=[],
        )

    def replace_cond_data(
        self, data: dict[str, Any], txt_labels: str | list[str]
    ) -> dict[str, Any]:
        """
        Replaces the conditioning data in the input data with the provided text labels.

        Args:
            data (dict[str, Any]): The input data.
            txt_labels (str | list[str]): The text labels to use for conditioning.

        Returns:
            dict[str, Any]: The data with replaced conditioning data.
        """
        if isinstance(txt_labels, str):
            txt_labels = [txt_labels] * len(data["scenes"])

        if len(txt_labels) != len(data["scenes"]):
            raise ValueError(
                "The number of text labels does not match the number of scenes."
            )

        if "text_cond" in data and self.tokenizer is not None:
            data["text_cond"] = self.tokenizer(txt_labels)

        if "text_cond_coarse" in data and self.tokenizer_coarse is not None:
            data["text_cond_coarse"] = self.tokenizer_coarse(txt_labels)

        return data

    def _setup_data_loaders(self):
        """
        Set up the data loaders for training, validation, and testing.
        This method is required by BaseDataset.
        """
        # For a single sample dataset, we'll use the same sample for all splits
        self.train_dataset_len = len(self.encoded_dataset)
        self.val_dataset_len = len(self.encoded_dataset)
        self.test_dataset_len = len(self.encoded_dataset)

        # All indices point to the same sample
        all_indices = list(range(len(self.encoded_dataset)))

        # Use all samples for all splits
        self.subset_indices = all_indices

        # For compatibility with parent class
        self.hf_dataset = None


class CustomSceneDataset(BaseDataset):
    def __init__(self, cfg: DictConfig, split: str, ckpt_path: str | None = None):
        """
        Custom dataset that loads sample data from numpy files.

        Args:
            cfg: Configuration object
            split: One of "training", "validation", "test"
            ckpt_path: Optional checkpoint path
        """
        self.cfg = cfg
        self.split = split

        # Create dummy metadata that includes required fields
        self.metadata = {
            "rotation_parametrization": "procrustes",
            "translation_vec_len": 3,
            "model_path_vec_len": cfg.model_path_vec_len,  # e.g., 19
            "max_num_objects_per_scene": cfg.max_num_objects_per_scene,  # e.g., 12
            "model_paths": ["model_" + str(i) for i in range(cfg.model_path_vec_len)],
            "welded_object_model_paths": [],
        }

        # Create a normalizer (required by the diffusion model)
        self.normalizer = self._setup_normalizer(ckpt_path)

        # Create the torch dataset - using your sample data
        self.dataset = NumpySampleDataset(
            data_dir="/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/ashok/sample_data",
            num_objects=cfg.max_num_objects_per_scene,
        )

        # Set up data loaders
        self._setup_data_loaders()

    def __len__(self):
        """Return the total length of the dataset based on the current split."""
        return len(self.subset_indices)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx: Index into the dataset

        Returns:
            Dictionary containing the scene data
        """
        # Map idx to the corresponding index in the subset
        actual_idx = self.subset_indices[idx]

        # Get the item from the underlying dataset
        item = self.dataset[actual_idx]

        # Apply normalization if needed
        # item["scenes"] = self.normalize_scenes(item["scenes"])
        # print(f"[Ashok] input scene: {item['scenes']}")
        return item  # {"scenes": scene, "idx": idx} this is actually what is fet to the model NOTE

    def normalize_scenes(self, scenes):
        """Normalize scene data using the fitted normalizer."""
        # return self.normalizer.transform(scenes)
        return scenes

    def inverse_normalize_scenes(self, scenes):
        """Inverse normalize scene data using the fitted normalizer."""
        # return self.normalizer.inverse_transform(scenes)
        return scenes

    def _setup_data_loaders(self):
        """
        Set up the data loaders for training, validation, and testing.
        This method is required by BaseDataset.
        """
        # For a single sample dataset, we'll use the same sample for all splits
        self.train_dataset_len = len(self.dataset)
        self.val_dataset_len = len(self.dataset)
        self.test_dataset_len = len(self.dataset)

        # All indices point to the same sample
        all_indices = list(range(len(self.dataset)))

        # Use all samples for all splits
        self.subset_indices = all_indices

        # For compatibility with parent class
        self.hf_dataset = None

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0, pin_memory=False):
        """
        Create a PyTorch DataLoader for the dataset.
        Required by the base class for training.
        """
        from torch.utils.data import DataLoader, Subset

        # Create a subset based on the current split
        subset = Subset(self.dataset, self.subset_indices)

        # Return a DataLoader using the subset
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """
        Custom collate function to handle the batch formatting.
        """
        scenes = torch.stack([item["scenes"] for item in batch])
        indices = [item["idx"] for item in batch]

        return {
            "scenes": scenes,
            "idx": indices,
        }

    def _setup_normalizer(self, ckpt_path):
        """Set up a normalizer for scene vectors."""
        normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)

        # If checkpoint provided, try to load normalizer from it
        if ckpt_path is not None:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if "normalizer_state" in ckpt:
                    normalizer.load_state(ckpt["normalizer_state"])
                    return normalizer
            except Exception as e:
                print(f"Could not load normalizer from checkpoint: {e}")

        # Otherwise, fit a new normalizer on some dummy data with correct feature dimension (62)
        # Breakdown: 22 (class) + 3 (translation) + 3 (size) + 2 (angles) + 32 (objfeats)
        dummy_data = torch.ones(100, 62)
        normalizer.fit(dummy_data)

        return normalizer

    def _get_item_from_index(self, idx):
        """Get an item from the dataset by index."""
        item = self.dataset[idx]
        return item

    def get_scene_vec_description(self):
        """Return a dummy scene vector description for compatibility."""
        # This would typically come from the metadata, but we'll create it on-the-fly
        from steerable_scene_generation.algorithms.common.dataclasses import (
            RotationParametrization,
            SceneVecDescription,
        )

        # Try to import PyDrake, but handle the case when it's not available
        try:
            from pydrake.all import PackageMap

            package_map = PackageMap()
        except ImportError:
            # If PyDrake is not available, create a dummy PackageMap
            class DummyPackageMap:
                def __init__(self):
                    pass

                def AddMap(self, *args):
                    pass

            package_map = DummyPackageMap()

        return SceneVecDescription(
            drake_package_map=package_map,
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.PROCRUSTES,
            model_paths=self.metadata["model_paths"],
            model_path_vec_len=self.metadata["model_path_vec_len"],
            welded_object_model_paths=[],
        )

    def replace_cond_data(
        self, data: dict[str, Any], txt_labels: str | list[str]
    ) -> dict[str, Any]:
        """
        Replaces the conditioning data in the input data with the provided text labels.

        Args:
            data (dict[str, Any]): The input data.
            txt_labels (str | list[str]): The text labels to use for conditioning.

        Returns:
            dict[str, Any]: The data with replaced conditioning data.
        """
        if isinstance(txt_labels, str):
            txt_labels = [txt_labels] * len(data["scenes"])

        if len(txt_labels) != len(data["scenes"]):
            raise ValueError(
                "The number of text labels does not match the number of scenes."
            )

        if "text_cond" in data and self.tokenizer is not None:
            data["text_cond"] = self.tokenizer(txt_labels)

        if "text_cond_coarse" in data and self.tokenizer_coarse is not None:
            data["text_cond_coarse"] = self.tokenizer_coarse(txt_labels)

        return data


class NumpySampleDataset(TorchDataset):
    """A dataset that loads a single sample from numpy files and repeats it."""

    def __init__(self, data_dir, num_objects=12):
        """
        Args:
            data_dir: Directory containing the numpy files
            num_objects: Number of objects per scene
        """
        self.data_dir = data_dir
        self.num_objects = num_objects

        # Load numpy files
        print(f"Loading sample data from {data_dir}")
        self.class_labels = np.load(
            os.path.join(data_dir, "class_labels.npy")
        )  # (1, 12, 22)
        self.translations = np.load(
            os.path.join(data_dir, "translations.npy")
        )  # (1, 12, 3)
        self.sizes = np.load(os.path.join(data_dir, "sizes.npy"))  # (1, 12, 3)
        self.angles = np.load(os.path.join(data_dir, "angles.npy"))  # (1, 12, 2)
        self.objfeats = np.load(
            os.path.join(data_dir, "objfeats_32.npy")
        )  # (1, 12, 32)

        # Print shapes for debugging
        print(f"Loaded data shapes:")
        print(f"  class_labels: {self.class_labels.shape}")
        print(f"  translations: {self.translations.shape}")
        print(f"  sizes: {self.sizes.shape}")
        print(f"  angles: {self.angles.shape}")
        print(f"  objfeats: {self.objfeats.shape}")

        # Prepare the scene representation by concatenating all features
        self.scene = self._prepare_scene()

        # Calculate total feature dimension
        self.total_feature_dim = (
            self.class_labels.shape[2]
            + self.translations.shape[2]  # class labels dimension (22)
            + self.sizes.shape[2]  # translations dimension (3)
            + self.angles.shape[2]  # sizes dimension (3)
            # + self.objfeats.shape[2]  # angles dimension (2)  # objfeats dimension (32)
        )
        print(f"Total feature dimension: {self.total_feature_dim}")  # Should be 62

        # We'll create 512 virtual samples by repeating the same scene
        # This helps with batch creation during training
        self.num_samples = 1  ##TODO: Change to 512 for overfit training

    def _prepare_scene(self):
        """Prepare the scene by concatenating all features."""
        # Since all arrays are already (1, 12, X), we just need to concatenate along the last dimension
        # First, extract the single sample (removing the batch dimension)
        class_labels = self.class_labels[0]  # (12, 22)
        translations = self.translations[0]  # (12, 3)
        sizes = self.sizes[0]  # (12, 3)
        angles = self.angles[0]  # (12, 2)
        objfeats = self.objfeats[0]  # (12, 32)

        # Now we need to concatenate for each object
        scene = np.zeros(
            (
                self.num_objects,
                class_labels.shape[1]
                + translations.shape[1]
                + sizes.shape[1]
                + angles.shape[1]
                # + objfeats.shape[1],
            )
        )  # (12, 62)

        # For each object, create the concatenated feature vector
        for i in range(self.num_objects):
            # Concatenate in order: class_labels, translations, sizes, angles, objfeats
            scene[i] = np.concatenate(
                [
                    class_labels[i],  # (22,)
                    translations[i],  # (3,)
                    sizes[i],  # (3,)
                    angles[i],  # (2,)
                    # objfeats[i],  # (32,)
                ]
            )

        # Convert to tensor with shape [num_objects, feature_dim]
        scene_tensor = torch.tensor(scene, dtype=torch.float32)
        print(f"Created scene tensor with shape {scene_tensor.shape}")

        return scene_tensor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Return the same scene for any index.

        Returns:
            Dictionary containing the scene data
        """
        return {
            "scenes": self.scene,  # B, 12, 30
            "idx": idx,
        }


if __name__ == "__main__":
    main()
    
# PYTHONPATH=. python steerable_scene_generation/datasets/custom_scene/custom_scene.py dataset=custom_scene
