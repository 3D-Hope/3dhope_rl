import json
import logging
import os
import random

from functools import cache
from typing import Any, Union

import torch

from datasets import Dataset
from omegaconf import DictConfig, ListConfig
from torch.utils.data import WeightedRandomSampler
from transformers import BatchEncoding

from steerable_scene_generation.algorithms.common.txt_encoding import (
    concat_batch_encodings,
    load_txt_encoder_from_config,
)
from steerable_scene_generation.datasets.common import BaseDataset
from steerable_scene_generation.datasets.common.shuffled_streaming_dataset import (
    InfiniteShuffledStreamingDataset,
)
from steerable_scene_generation.utils.hf_dataset import load_hf_dataset_with_metadata
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler

try:
    from .threed_front_encoding import get_dataset_raw_and_encoded
except Exception:
    # Fallback if relative import fails when run as a module
    from steerable_scene_generation.datasets.custom_scene.threed_front_encoding import (
        get_dataset_raw_and_encoded,
    )


def update_data_file_paths(config_data, config):
    config_data["dataset_directory"] = os.path.join(
        config["data"]["path_to_processed_data"], config_data["dataset_directory"]
    )
    config_data["annotation_file"] = os.path.join(
        config["data"]["path_to_dataset_files"], config_data["annotation_file"]
    )
    return config_data


console_logger = logging.getLogger(__name__)


class CustomDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        split: str | list | ListConfig,
        ckpt_path: str | None = None,
    ):
        """
        Args:
            cfg: a DictConfig object defined by `configurations/dataset/scene.yaml`.
            split: One of "training", "validation", "test".
            ckpt_path: The optional checkpoint path.
        """
        self.cfg = cfg

        # Prepare ThreedFront config pieces
        data_cfg = cfg["data"] if "data" in cfg else cfg.data
        network_cfg = cfg["network"] if "network" in cfg else cfg.network

        # Resolve split names for ThreedFront
        if split == "training":
            split_names = (
                cfg["training"].get("splits", ["train", "val"])
                if "training" in cfg
                else ["train", "val"]
            )
        elif split == "validation":
            split_names = (
                cfg["validation"].get("splits", ["test"])
                if "validation" in cfg
                else ["test"]
            )
        elif type(split) == list or type(split) == ListConfig:
            split_names = split
        else:
            split_names = [split]

        # print(f"[Ashok] max_num_objects_per_scene: {cfg.max_num_objects_per_scene}")
        # import sys; sys.exit();

        # Load ThreedFront raw and encoded datasets
        self.raw_dataset, self.encoded_dataset = get_dataset_raw_and_encoded(
            update_data_file_paths(data_cfg, cfg),
            split=split_names,
            max_length=cfg.max_num_objects_per_scene,
            include_room_mask=network_cfg.get("room_mask_condition", True),
        )

        # HF dataset features are not used in this ThreedFront path
        self.hf_dataset = None
        self.use_subdataset_sampling = False
        self.subdataset_ranges = None
        self.subdataset_names = None

        # Build a normalizer compatible with expected interface
        self.normalizer = self._setup_normalizer_threedfront(ckpt_path)

        # Tokenizers (kept for API parity; will only be used if configured)
        self._setup_tokenizers()

    def _subsample_dataset_if_enabled(self, hf_dataset: Dataset) -> Dataset:
        """
        Subsamples the dataset if the random dataset sampling is enabled.
        """
        if self.cfg.random_dataset_sampling.use and self.cfg.subdataset_sampling.use:
            raise NotImplementedError(
                "Cannot use both subdataset sampling and random dataset sampling!"
            )

        if self.cfg.random_dataset_sampling.use:
            num_samples = int(self.cfg.random_dataset_sampling.num_samples)
            if len(hf_dataset) < num_samples:
                raise ValueError(
                    f"Dataset size ({len(hf_dataset)}) is smaller than the number "
                    f"of samples to sample ({num_samples})!"
                )
            hf_dataset = hf_dataset.select(
                torch.randperm(len(hf_dataset))[:num_samples]
            )
            console_logger.info(
                f"Using random dataset sampling with {num_samples} samples."
            )

        return hf_dataset

    def _perform_train_test_validation_split(
        self, hf_dataset: Dataset, split: str
    ) -> Dataset:
        """
        Performs a train-test split on the dataset.
        """
        if self.use_subdataset_sampling:
            # Don't split the dataset as it would mess up the subdataset sampling
            # indices due to random shuffling.
            self.setup_subdataset_sampling(self.cfg.subdataset_sampling, hf_dataset)

            # Calculate what the split size would be to maintain consistent dataset
            # sizes.
            total_size = len(hf_dataset)
            if split == "training":
                split_size = int(
                    total_size * (1.0 - self.cfg.val_ratio - self.cfg.test_ratio)
                )
            elif split == "validation":
                split_size = int(total_size * self.cfg.val_ratio)
            elif split == "test":
                split_size = int(total_size * self.cfg.test_ratio)
            else:
                raise ValueError(f"Invalid split: {split}")

            self.sampling_dataset_length = split_size
            console_logger.info(
                f"Using subdataset sampling with {split} split "
                f"length: {self.sampling_dataset_length}"
            )
            return hf_dataset

        # Only perform train-test split when not using subdataset sampling.
        train_ratio = (
            1.0 - self.cfg.val_ratio - self.cfg.test_ratio
            if len(hf_dataset) > 1
            else 1.0
        )
        train_testval_split = hf_dataset.train_test_split(train_size=train_ratio)
        if split == "training":
            return train_testval_split["train"]
        else:
            # Split further into validation and test.
            val_test_split = train_testval_split["test"].train_test_split(
                train_size=self.cfg.val_ratio
                / (self.cfg.val_ratio + self.cfg.test_ratio)
            )
            if split == "validation":
                return val_test_split["train"]
            elif split == "test":
                return val_test_split["test"]
            else:
                raise ValueError(f"Invalid split: {split}")

    def _setup_normalizer_threedfront(self, ckpt_path: str | None) -> MinMaxScaler:
        """
        Sets up a normalizer for ThreedFront-encoded samples by inferring the
        feature dimension and fitting a placeholder scaler. ThreedFront inputs are
        generally pre-scaled; this keeps the API consistent.
        """
        normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)

        # Try to restore from checkpoint if available
        if ckpt_path is not None:
            # print(f"[DEBUG] Loading normalizer state from checkpoint: {ckpt_path} SUI4")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "normalizer_state" in ckpt:
                normalizer.load_state(ckpt["normalizer_state"])
                return normalizer

        # Infer feature dimension from a sample and fit a dummy scaler
        sample = self.encoded_dataset[0]
        scene_tensor = self._to_scene_tensor(sample)
        feature_dim = scene_tensor.shape[-1]
        dummy = torch.ones(128, feature_dim)
        normalizer.fit(dummy)
        return normalizer

    def _setup_tokenizers(self):
        """
        Sets up the tokenizers.
        """
        # Load the tokenizer if using classifier-free guidance.
        self.tokenizer, self.tokenizer_coarse = None, None
        if not self.cfg.classifier_free_guidance.use:
            return

        # Setup the primary tokenizer.
        self.masking_prop = self.cfg.classifier_free_guidance.masking_prob
        self.tokenizer, _ = load_txt_encoder_from_config(
            self.cfg, component="tokenizer"
        )

        # Setup the coarse tokenizer if configured.
        use_coarse = self.cfg.classifier_free_guidance.txt_encoder_coarse is not None
        if use_coarse:
            self.masking_prop_coarse = (
                self.cfg.classifier_free_guidance.masking_prob_coarse
            )
            self.tokenizer_coarse, _ = load_txt_encoder_from_config(
                self.cfg, is_coarse=True, component="tokenizer"
            )

        # Pre-cache empty encodings for masked prompts.
        self._setup_tokenization_caches()

        # Setup static prompt caching if configured.
        if self.cfg.static_subdataset_prompts.use and hasattr(self, "subdataset_names"):
            self._setup_static_prompt_caches()

    def _setup_tokenization_caches(self):
        """
        Sets up caches for tokenization to improve performance.
        This includes caching empty string tokenization for masked prompts.
        """
        # Pre-generate empty token encodings for masked prompts.
        if self.tokenizer is not None:
            self._empty_encoding = self.tokenizer([""])
            self._empty_encoding = BatchEncoding(
                {k: v.squeeze(0) for k, v in self._empty_encoding.items()}
            )
            console_logger.info("Pre-cached empty encoding for regular tokenizer.")

        if self.tokenizer_coarse is not None:
            self._empty_encoding_coarse = self.tokenizer_coarse([""])
            self._empty_encoding_coarse = BatchEncoding(
                {k: v.squeeze(0) for k, v in self._empty_encoding_coarse.items()}
            )
            console_logger.info("Pre-cached empty encoding for coarse tokenizer.")

    def _setup_static_prompt_caches(self):
        """
        Pre-caches tokenized versions of static subdataset prompts.
        """
        if self.tokenizer is None and self.tokenizer_coarse is None:
            return

        self._tokenized_prompts_cache = {}
        for prompt in self.cfg.static_subdataset_prompts.name_to_prompt.values():
            self._tokenized_prompts_cache[prompt] = {}

            if self.tokenizer is not None:
                text_cond = self.tokenizer([prompt])
                self._tokenized_prompts_cache[prompt]["regular"] = BatchEncoding(
                    {k: v.squeeze(0) for k, v in text_cond.items()}
                )

            if self.tokenizer_coarse is not None:
                text_cond_coarse = self.tokenizer_coarse([prompt])
                self._tokenized_prompts_cache[prompt]["coarse"] = BatchEncoding(
                    {k: v.squeeze(0) for k, v in text_cond_coarse.items()}
                )

        console_logger.info(
            f"Pre-cached tokenization for {len(self._tokenized_prompts_cache)} static "
            "prompts."
        )

    def normalize_scenes(self, scenes: torch.Tensor) -> torch.Tensor:
        """
        Normalize scenes to [-1, 1].

        Args:
            scenes: Scenes to normalize. Shape (B, N, O) where B is the number of scenes,
                N is the number of objects, and O is the object feature vector length.

        Returns:
            torch.Tensor: Normalized scenes.
        """
        # normalized_scenes = self.normalizer.transform(
        #     scenes.reshape(-1, scenes.shape[-1])
        # ).reshape(scenes.shape)
        # return normalized_scenes
        return scenes

    def inverse_normalize_scenes(self, scenes: torch.Tensor) -> torch.Tensor:
        """
        Inverse normalize scenes from [-1, 1].

        Args:
            scenes: Scenes to inverse normalize. Shape (B, N, O) where B is the number of
                scenes, N is the number of objects, and O is the object feature vector
                length.

        Returns:
            torch.Tensor: Inverse normalized scenes.
        """
        # unormalized_scenes = self.normalizer.inverse_transform(
        #     scenes.reshape(-1, scenes.shape[-1])
        # ).reshape(scenes.shape)

        # return unormalized_scenes
        return scenes

    def __len__(self) -> int:
        """
        Returns the length of the ThreedFront encoded dataset.
        """
        if hasattr(self, "encoded_dataset") and self.encoded_dataset is not None:
            return len(self.encoded_dataset)
        return 0

    def _get_item(self, idx: int) -> dict[str, torch.Tensor]:
        if self.encoded_dataset is None:
            raise ValueError("ThreedFront encoded dataset is not loaded!")
        return self.encoded_dataset[idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raw_item = self._get_item(idx)
        
        # ===== ANALYSIS CODE: Detect problematic scenes =====
        self._analyze_scene_quality(raw_item, idx)
        # ===================================================
        
        scene_tensor = self._to_scene_tensor(raw_item)

        item: dict[str, Any] = {
            "scenes": scene_tensor,
            "idx": idx,
            "fpbpn": raw_item["fpbpn"],
        }

        # if self.cfg.use_permutation_augmentation:
        #     perm = torch.randperm(len(scene_tensor))
        #     item["scenes"] = scene_tensor[perm]

        # # Optional text handling only if configured with static prompts
        # if (
        #     (self.tokenizer is not None or self.tokenizer_coarse is not None)
        #     and getattr(self.cfg, "static_subdataset_prompts", None)
        #     and self.cfg.static_subdataset_prompts.use
        #     and hasattr(self, "_tokenized_prompts_cache")
        # ):
        #     # Fall back to a single prompt if provided; otherwise skip
        #     prompts = list(self.cfg.static_subdataset_prompts.name_to_prompt.values())
        #     if len(prompts) > 0:
        #         prompt = prompts[0]
        #         if self.tokenizer is not None:
        #             if random.random() >= self.masking_prop:
        #                 item["text_cond"] = self._tokenized_prompts_cache[prompt][
        #                     "regular"
        #                 ]
        #             else:
        #                 item["text_cond"] = self._empty_encoding
        #         if self.tokenizer_coarse is not None:
        #             if random.random() >= self.masking_prop_coarse:
        #                 item["text_cond_coarse"] = self._tokenized_prompts_cache[
        #                     prompt
        #                 ]["coarse"]
        #             else:
        #                 item["text_cond_coarse"] = self._empty_encoding_coarse

        return item

    def get_floor_polygon_points(self, idx: int) -> torch.Tensor:
        raw_item = self._get_item(idx)
        return raw_item["floor_polygon_points"]

    def get_floor_plan_args(self, idx: int) -> dict[str, torch.Tensor]:
        raw_item = self._get_item(idx)
        return {
            "floor_plan_centroid": raw_item["floor_plan_centroid"],
            "floor_plan_faces": raw_item["floor_plan_faces"],
            "floor_plan_vertices": raw_item["floor_plan_vertices"],
            "room_outer_box": raw_item["room_outer_box"],
        }

    def _analyze_scene_quality(self, raw_item: dict, idx: int) -> None:
        """
        Analyzes a scene for quality issues including NaN, Inf, and out-of-range values.
        Logs warnings for problematic scenes and stores them in an analysis report.
        """
        import numpy as np
        
        # Initialize analysis storage on first call
        if not hasattr(self, '_scene_analysis_report'):
            self._scene_analysis_report = {
                'problematic_indices': [],
                'issue_summary': {}
            }
        
        issues = []
        issue_details = {}
        
        # Analyze each component
        components_to_check = {
            'class_labels': raw_item.get('class_labels'),
            'translations': raw_item.get('translations'),
            'sizes': raw_item.get('sizes'),
            'angles': raw_item.get('angles'),
            'objfeats_32': raw_item.get('objfeats_32'),
            'objfeats': raw_item.get('objfeats'),
            'floor_polygon_points': raw_item.get('floor_polygon_points'),
            'floor_plan_centroid': raw_item.get('floor_plan_centroid'),
            'floor_plan_vertices': raw_item.get('floor_plan_vertices'),
            'room_outer_box': raw_item.get('room_outer_box'),
        }
        
        for component_name, component_data in components_to_check.items():
            if component_data is None:
                continue
            
            # Convert to numpy if tensor
            if isinstance(component_data, torch.Tensor):
                data_np = component_data.cpu().numpy()
            else:
                data_np = np.asarray(component_data)
            
            # Check for NaN
            nan_count = np.isnan(data_np).sum()
            if nan_count > 0:
                issues.append(f"NaN in {component_name}")
                issue_details[component_name] = {'nan_count': int(nan_count), 'shape': data_np.shape}
            
            # Check for Inf
            inf_count = np.isinf(data_np).sum()
            if inf_count > 0:
                issues.append(f"Inf in {component_name}")
                issue_details[component_name] = {'inf_count': int(inf_count), 'shape': data_np.shape}
            
            # Check for extremely large values (potential overflow risk)
            max_val = np.nanmax(np.abs(data_np))
            if max_val > 1e6:
                issues.append(f"Extreme values in {component_name} (max abs: {max_val:.2e})")
                issue_details[component_name] = {'max_abs': float(max_val), 'shape': data_np.shape}
            
            # Check for all zeros
            if np.allclose(data_np, 0):
                issues.append(f"All zeros in {component_name}")
                issue_details[component_name] = {'status': 'all_zeros', 'shape': data_np.shape}
            
            # For positional data, check ranges
            if component_name in ['translations', 'sizes', 'floor_polygon_points', 'floor_plan_vertices']:
                if np.any(data_np < -1e3) or np.any(data_np > 1e3):
                    issues.append(f"Out-of-range values in {component_name}")
                    issue_details[component_name] = {
                        'min': float(np.nanmin(data_np)),
                        'max': float(np.nanmax(data_np)),
                        'shape': data_np.shape
                    }
        
        # Store the analysis report
        if issues:
            self._scene_analysis_report['problematic_indices'].append(idx)
            self._scene_analysis_report['issue_summary'][idx] = {
                'fpbpn': raw_item.get('fpbpn', 'unknown'),
                'issues': issues,
                'details': issue_details,
                'num_objects': len(raw_item.get('class_labels', []))
            }
            
            # Log warning
            console_logger.warning(
                f"[Scene {idx}] Problematic data detected:\n"
                f"  FPBPN: {raw_item.get('fpbpn', 'unknown')}\n"
                f"  Issues: {', '.join(issues)}\n"
                f"  Details: {issue_details}"
            )

    def _to_scene_tensor(self, item: dict) -> torch.Tensor:
        """
        Convert a ThreedFront-encoded sample dict to a scene tensor of shape (N, O).
        Concatenates class_labels, translations, sizes, angles and includes objfeats
        if present in the encoding.
        """
        components = [
            item["class_labels"],
            item["translations"],
            item["sizes"],
            item["angles"],
        ]
        if "objfeats_32" in item:
            components.append(item["objfeats_32"])  # prefer 32-dim if available
        elif "objfeats" in item:
            components.append(item["objfeats"])  # fallback 64-dim

        if isinstance(components[0], torch.Tensor):
            concat = torch.cat(components, dim=1)
        else:
            import numpy as np

            concat = torch.from_numpy(np.concatenate(components, axis=1)).float()
        return concat

    def add_classifier_free_guidance_uncond_data(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Adds classifier-free guidance unconditioned data to the input data. The
        resulting data will contain the original data along with the classifier-free
        guidance unconditioned data, concatenated such that the original data comes
        first.

        Args:
            data (dict[str, Any]): The input data.

        Returns:
            dict[str, Any]: The data with added classifier-free guidance unconditioned
            data.
        """
        if not self.cfg.classifier_free_guidance.use:
            return data

        if not data["scenes"].dim() == 3:
            raise NotImplementedError("Only batched data is supported right now.")

        uncond_txt = [""] * len(data["scenes"])

        if "language_annotation" in data:
            data["language_annotation"] = data["language_annotation"] + uncond_txt

        if "text_cond" in data:
            device = data["text_cond"]["input_ids"].device
            uncond_cond = self.tokenizer(uncond_txt).to(device)
            data["text_cond"] = concat_batch_encodings([data["text_cond"], uncond_cond])

        if "text_cond_coarse" in data:
            device = data["text_cond_coarse"]["input_ids"].device
            uncond_cond_coarse = self.tokenizer_coarse(uncond_txt).to(device)
            data["text_cond_coarse"] = concat_batch_encodings(
                [data["text_cond_coarse"], uncond_cond_coarse]
            )

        return data

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

    def get_analysis_report(self) -> dict:
        """
        Returns a comprehensive analysis report of all problematic scenes encountered
        during data loading. Should be called after iterating through the dataset.
        """
        if not hasattr(self, '_scene_analysis_report'):
            return {'status': 'No issues found', 'problematic_scenes': 0}
        
        report = self._scene_analysis_report.copy()
        report['num_problematic_scenes'] = len(report['problematic_indices'])
        report['percentage_problematic'] = (
            100.0 * report['num_problematic_scenes'] / len(self)
            if len(self) > 0 else 0.0
        )
        
        return report

    def save_analysis_report(self, output_path: str = None) -> str:
        """
        Saves the scene quality analysis report to a JSON file.
        
        Args:
            output_path: Path to save the report. If None, saves to ./scene_analysis_report.json
        
        Returns:
            str: Path where the report was saved
        """
        if output_path is None:
            output_path = "./scene_analysis_report.json"
        
        report = self.get_analysis_report()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj
        
        report = convert_to_serializable(report)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        console_logger.info(f"Analysis report saved to {output_path}")
        return output_path

    def print_analysis_summary(self) -> None:
        """
        Prints a human-readable summary of the analysis report.
        """
        report = self.get_analysis_report()
        
        print("\n" + "="*80)
        print("SCENE QUALITY ANALYSIS REPORT")
        print("="*80)
        print(f"Total scenes in dataset: {len(self)}")
        print(f"Problematic scenes: {report.get('num_problematic_scenes', 0)}")
        print(f"Percentage problematic: {report.get('percentage_problematic', 0):.2f}%")
        
        if report.get('num_problematic_scenes', 0) > 0:
            print("\nProblematic scene indices:", report.get('problematic_indices', []))
            print("\nDetailed issues by scene:")
            print("-" * 80)
            
            for idx, issue_info in report.get('issue_summary', {}).items():
                print(f"\nScene index: {idx}")
                print(f"  FPBPN: {issue_info.get('fpbpn', 'unknown')}")
                print(f"  Number of objects: {issue_info.get('num_objects', 'unknown')}")
                print(f"  Issues: {', '.join(issue_info.get('issues', []))}")
                if issue_info.get('details'):
                    print(f"  Details:")
                    for component, detail in issue_info['details'].items():
                        print(f"    - {component}: {detail}")
        else:
            print("\n✓ No issues detected!")
        
        print("="*80 + "\n")

    @staticmethod
    def sample_data_dict(data: dict[str, Any], num_items: int) -> dict[str, Any]:
        """
        Sample `num_items` from `data`. Sample with replacement if `data` contains less
        than `num_items` items.

        Args:
            data (dict[str, Any]): The data to sample.

        Returns:
            dict[str, Any]: The sampled data of length `num_items`.
        """
        total_items = len(data["scenes"])

        if num_items <= total_items:
            # Sample without replacement.
            sample_indices = torch.randperm(total_items)[:num_items]
        else:
            # Sample with replacement.
            sample_indices = torch.randint(0, total_items, (num_items,))

        # Create the sampled data dictionary.
        sampled_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                sampled_data[key] = value[sample_indices]
            elif isinstance(value, list):
                sampled_data[key] = [value[i] for i in sample_indices]
            elif isinstance(value, BatchEncoding):
                sampled_data[key] = BatchEncoding(
                    {k: v[sample_indices] for k, v in value.items()}
                )
            else:
                raise ValueError(
                    f"Unsupported data type '{type(value)}' for key '{key}'"
                )

        return sampled_data

    def get_sampler(self) -> Union[WeightedRandomSampler, None]:
        """
        Returns a sampler for weighted random sampling of the dataset based on the
        dataset labels.
        This is an alternative to the two-step sampling process used in
        `sample_subdataset_index` and `sample_from_subdataset`.
        """
        if not self.cfg.custom_data_batch_mix.use:
            return None

        if "labels" not in self.hf_dataset.column_names:
            raise ValueError("Dataset does not contain labels!")

        labels = self.hf_dataset["labels"]

        # Calculate the number of samples for each class.
        class_counts = torch.bincount(labels)

        # Calculate weights for each class.
        class_weights = [
            p / count if count > 0 else 0.0
            for p, count in zip(
                self.cfg.custom_data_batch_mix.label_probs, class_counts
            )
        ]

        # Create a list of weights for each sample in the dataset.
        weights = [class_weights[label] for label in labels]

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    @cache
    def get_all_data(
        self,
        normalized: bool = True,
        label: int = None,
        scene_indices: torch.Tensor | None = None,
        only_scenes: bool = False,
    ) -> dict[str, Any]:
        """
        Returns all data in the dataset, including scenes and additional attributes.

        Args:
            normalized (bool, optional): Whether to return normalized scenes.
            label (int, optional): If not None, only data that correspond to that
                label are returned. This option is ignored if the dataset does not
                contain labels.
            scene_indices (torch.Tensor, optional): If not None, only data at the
                specified indices are returned.
            only_scenes (bool, optional): If True, only the scenes are returned.

        Returns:
            dict[str, Any]: All data in the dataset, including "scenes" and
                additional attributes.
        """
        if self.encoded_dataset is None:
            raise ValueError("Dataset is not loaded!")

        if only_scenes and label is not None:
            raise ValueError("Cannot specify both 'only_scenes' and 'label'.")

        # Use indexing to fetch only the required data.
        # Aggregate from encoded dataset
        indices = (
            scene_indices.tolist()
            if scene_indices is not None
            else list(range(len(self)))
        )
        scenes_list = []
        for i in indices:
            sample = self.encoded_dataset[i]
            scenes_list.append(self._to_scene_tensor(sample))
        raw_scenes = torch.stack(scenes_list, dim=0)
        if only_scenes:
            return {"scenes": raw_scenes}
        return {"scenes": raw_scenes}

    def set_data(self, data: dict[str, torch.Tensor], normalized: bool = False) -> None:
        """
        Replaces the dataset with new data.

        Note that this will disable subdataset sampling if it was enabled.

        Args:
            data (dict[str, torch.Tensor]): The data dictionary containing "scenes" and
                optionally additional attributes such as "labels".
            normalized (bool): Whether the scenes are normalized.
        """
        # Not supported for ThreedFront-backed dataset
        raise NotImplementedError("set_data is not supported for ThreedFront dataset")

    def setup_subdataset_sampling(
        self, sampling_cfg: DictConfig, hf_dataset: Dataset
    ) -> None:
        """
        Sets up weighted subdataset sampling based on configuration.

        Args:
            sampling_cfg: Configuration for subdataset sampling.
            hf_dataset: HuggingFace dataset to sample from.
        """
        self.subdataset_probs = sampling_cfg.probabilities

        # Validate that all subdataset names have probabilities.
        if self.subdataset_probs is None:
            raise ValueError(
                "Subdataset sampling is enabled but no probabilities are specified."
            )
        for name in self.subdataset_names:
            if name not in self.subdataset_probs:
                raise ValueError(f"No probability specified for subdataset '{name}'")

        # Validate that probabilities approximately sum to 1.
        total_prob = sum(self.subdataset_probs.values())
        if not 0.98 <= total_prob <= 1.02:  # Allow for small floating point errors
            raise ValueError(
                f"Subdataset probabilities must sum to 1, got {total_prob}"
            )

        # Normalize probabilities to sum to 1.
        self.subdataset_probs = {
            name: prob / total_prob for name, prob in self.subdataset_probs.items()
        }

        # Create cumulative probabilities for efficient sampling
        self.subdataset_cum_probs = []
        cum_prob = 0.0
        for name in self.subdataset_names:
            cum_prob += self.subdataset_probs[name]
            self.subdataset_cum_probs.append(cum_prob)

        # Check if we should use infinite iterators.
        self.use_infinite_iterators = sampling_cfg.use_infinite_iterators
        if self.use_infinite_iterators:
            # Create infinite iterators for each subdataset.
            self.subdataset_iterators = []
            for start_idx, end_idx in self.subdataset_ranges:
                subdataset = hf_dataset.select(range(start_idx, end_idx))
                iterator = InfiniteShuffledStreamingDataset(
                    dataset=subdataset, buffer_size=sampling_cfg.buffer_size
                )
                self.subdataset_iterators.append(iterator)

            # Initialize iterators.
            self.subdataset_iterator_objects = [
                iter(iterator) for iterator in self.subdataset_iterators
            ]
            console_logger.info("Using infinite iterators for subdataset sampling.")

        console_logger.info(
            "Enabled weighted subdataset sampling with probabilities: "
            f"{self.subdataset_probs}"
        )

    def sample_subdataset_index(self) -> int:
        """
        Samples a subdataset index based on the configured probabilities.

        Returns:
            int: The index of the sampled subdataset.
        """
        if not self.use_subdataset_sampling:
            raise RuntimeError("Subdataset sampling is not enabled.")

        # Sample a random value between 0 and 1.
        r = random.random()

        # Find the subdataset whose cumulative probability range contains r.
        for i, cum_prob in enumerate(self.subdataset_cum_probs):
            if r <= cum_prob:
                return i

        # Something went wrong.
        raise RuntimeError("Failed to sample a subdataset index.")

    def sample_from_subdataset(self, subdataset_idx: int) -> int:
        """
        Samples an index from the specified subdataset.

        Args:
            subdataset_idx: Index of the subdataset to sample from.

        Returns:
            int: The global index of the sampled item.
        """
        # Use pre-computed indices for fast sampling.
        start_idx, end_idx = self.subdataset_ranges[subdataset_idx]
        return random.randint(start_idx, end_idx - 1)

    def get_subdataset_name_from_index(self, index: int) -> str:
        """
        Returns the subdataset name for the given dataset index.
        """
        if self.subdataset_ranges is None:
            raise ValueError("Subdataset ranges are not set!")

        for i, (start_idx, end_idx) in enumerate(self.subdataset_ranges):
            if index >= start_idx and index < end_idx:
                return self.subdataset_names[i]
        raise ValueError(f"Index {index} is out of range for any subdataset.")

    def _validate_subdataset_config(self) -> None:
        """
        Validates the subdataset configuration, ensuring that necessary metadata
        is available when using subdataset features.
        """
        # Check if static subdataset prompts are enabled but metadata is missing.
        if self.cfg.static_subdataset_prompts.use and (
            self.subdataset_ranges is None or self.subdataset_names is None
        ):
            raise ValueError(
                "Require subdataset ranges and names to be set when using static "
                "subdataset prompts!"
            )

        # Check if subdataset sampling is enabled but metadata is missing.
        if self.cfg.subdataset_sampling.use and (
            self.subdataset_ranges is None or self.subdataset_names is None
        ):
            raise ValueError(
                "Require subdataset ranges and names to be set when using "
                "subdataset sampling!"
            )

        # Check if static prompts are provided for all subdatasets.
        if self.cfg.static_subdataset_prompts.use and set(
            self.cfg.static_subdataset_prompts.name_to_prompt.keys()
        ) != set(self.subdataset_names):
            raise ValueError(
                "Require static subdataset prompts to be set for all subdatasets!\n"
                f"Subdataset names: {self.subdataset_names}\n"
                f"Prompts: {self.cfg.static_subdataset_prompts.name_to_prompt.keys()}"
            )

    def _validate_dataset_structure(
        self, hf_dataset: Dataset, metadata: dict[str, Any]
    ) -> None:
        # Not used in ThreedFront-backed implementation
        return


from omegaconf import DictConfig, OmegaConf
from steerable_scene_generation.utils.omegaconf import register_resolvers
import hydra

@hydra.main(version_base=None, config_path="../../../configurations", config_name="config")
#TODO: look at the living room dataset. is any scene corrupted why giving nan?

def main(cfg: DictConfig) -> None:
    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)
    
    
    dataset = CustomDataset(
        cfg=cfg.dataset,
        split=["train", "val", "test"],
        ckpt_path=None,
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # ===== ANALYSIS: Iterate through dataset to find problematic scenes =====
    print("\n" + "="*80)
    print("ANALYZING DATASET FOR QUALITY ISSUES")
    print("="*80)
    
    for i in range(len(dataset)):
        if i % max(1, len(dataset) // 10) == 0:
            print(f"Progress: {i}/{len(dataset)} scenes analyzed...")
        _ = dataset[i]  # This will trigger the analysis in __getitem__
    
    # Print and save the analysis report
    dataset.print_analysis_summary()
    report_path = dataset.save_analysis_report(
        output_path="./scene_quality_analysis.json"
    )
    print(f"\nAnalysis report saved to: {report_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
    
    
    
