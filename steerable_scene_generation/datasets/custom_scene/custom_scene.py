import json
import os

import numpy as np
import torch

from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

from steerable_scene_generation.datasets.common import BaseDataset
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler


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

        return item

    def normalize_scenes(self, scenes):
        """Normalize scene data using the fitted normalizer."""
        return self.normalizer.transform(scenes)

    def inverse_normalize_scenes(self, scenes):
        """Inverse normalize scene data using the fitted normalizer."""
        return self.normalizer.inverse_transform(scenes)

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
            + self.objfeats.shape[2]  # angles dimension (2)  # objfeats dimension (32)
        )
        print(f"Total feature dimension: {self.total_feature_dim}")  # Should be 62

        # We'll create 512 virtual samples by repeating the same scene
        # This helps with batch creation during training
        self.num_samples = 512

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
                + objfeats.shape[1],
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
                    objfeats[i],  # (32,)
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
            "scenes": self.scene,
            "idx": idx,
        }


# Original DummyTorchDataset implementation (commented out)
"""
class DummyTorchDataset(TorchDataset):
    A dummy PyTorch dataset that generates random scene data.
    
    def __init__(self, num_samples, num_objects, feature_dim):
        Args:
            num_samples: Number of samples in the dataset
            num_objects: Number of objects per scene
            feature_dim: Dimension of each object feature vector
        
        print(f"[Ashok] custom dumby dataset initialized with {num_samples} samples, ")
        self.num_samples = num_samples
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        
        # Generate random seed for reproducibility
        self.seed = 42
        np.random.seed(self.seed)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        Generate a random scene with objects.
        
        Structure of each object vector:
        - First 3 elements: Translation (x, y, z)
        - Next 9 elements: Rotation (Procrustes format)
        - Remaining elements: One-hot encoding of object type
        
        # Generate random seed based on index for deterministic behavior
        np.random.seed(self.seed + idx)
        
        # Create random scene data
        scene = np.random.randn(self.num_objects, self.feature_dim)
        
        # Create a one-hot encoding part for the last part of each vector
        # Assuming 18 dimensions are used for translation and rotation
        # and the rest are for object type
        model_path_vec_len = self.feature_dim - 12  # Subtract 3 for translation and 9 for rotation
        
        for i in range(self.num_objects):
            # Normalize translation to reasonable range (-2 to 2)
            scene[i, :3] = scene[i, :3] * 2
            
            # Normalize rotation part (this is a simplification)
            # In a real scenario, rotation should be a valid rotation representation
            rot_part = scene[i, 3:12]
            # Simple normalization for quaternion-like structure
            rot_norm = np.linalg.norm(rot_part)
            if rot_norm > 0:
                scene[i, 3:12] = rot_part / rot_norm
            
            # Create one-hot encoding for object type
            one_hot_start = 12
            one_hot_idx = np.random.randint(0, model_path_vec_len)
            scene[i, one_hot_start:] = 0
            scene[i, one_hot_start + one_hot_idx] = 1
        
        # Convert to tensor
        scene_tensor = torch.tensor(scene, dtype=torch.float32)
        
        # Return in the format expected by the model
        return {
            "scenes": scene_tensor,
            "idx": idx,
        }
"""
