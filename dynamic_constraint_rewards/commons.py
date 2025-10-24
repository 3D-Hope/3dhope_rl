import json
import os

import torch

from universal_constraint_rewards.commons import parse_and_descale_scenes


def import_dynamic_reward_functions(reward_code_dir: str):
    import os

    # Test each reward function individually.
    reward_functions = {}
    # Iterate through each file inside reward_code_dir and import the reward function.
    for file in os.listdir(reward_code_dir):
        if file.endswith(".py") and file != "__init__.py":
            # Dynamically import the module and extract get_reward and test_reward.
            import importlib

            module_name = f"dynamic_constraint_rewards.dynamic_reward_functions.{file.split('.')[0]}"
            module = importlib.import_module(module_name)

            get_reward = getattr(module, "get_reward")
            test_reward = getattr(module, "test_reward")

            # print(f"[SAUGAT] Imported reward function from {file.split('.')[0]}")

            reward_functions[file.split(".")[0]] = {
                "get_reward": get_reward,
                "test_reward": test_reward,
            }
    get_reward_functions = {}
    for key, value in reward_functions.items():
        get_reward_functions[key] = value["get_reward"]

    test_reward_functions = {}
    for key, value in reward_functions.items():
        test_reward_functions[key] = value["test_reward"]

    return get_reward_functions, test_reward_functions


def get_dynamic_reward(
    parsed_scene,
    reward_normalizer,
    get_reward_functions,
    num_classes=22,
    dynamic_importance_weights=None,
    config=None,
    **kwargs,
):
    """
    Entry point for computing dynamic reward from multiple reward functions.

    this function assumes, llm has already created variable number of rewards to follow user's instruction. these reward functions are runnable.

    llm should also give use a technique to normalize each reward to be bounded with in [0, 1] range.
    0 is worst 1 is the best scene.


    this function returns the sum of all those rewards weighted by importance weights. and the total should also be in range [0, 1].
    """
    rewards = {}
    room_type = config.ddpo.dynamic_constraint_rewards.room_type
    all_rooms_info = json.load(
        open(
            os.path.join(
                config.dataset.data.path_to_dataset_files, "all_rooms_info.json"
            )
        )
    )
    idx_to_label = all_rooms_info[room_type]["unique_values"]
    max_objects = all_rooms_info[room_type]["max_objects"]
    num_classes = all_rooms_info[room_type]["num_classes"]
    for key, value in get_reward_functions.items():
        reward = value(
            parsed_scene,
            idx_to_labels=idx_to_label,
            num_classes=num_classes,
            max_objects=max_objects,
            **kwargs,
        )
        print(f"[Ashok] Raw reward for {key}: {reward}")
        rewards[key] = reward

    reward_components = {}
    if reward_normalizer is not None:
        for key, value in rewards.items():
            reward_components[key] = value
            rewards[key] = reward_normalizer.normalize(key, torch.tensor(value))
    else:
        for key, value in rewards.items():
            reward_components[key] = value
    rewards_sum = 0
    if dynamic_importance_weights is None:
        dynamic_importance_weights = {key: 1.0 for key in rewards.keys()}


    for key, value in rewards.items():
        importance = dynamic_importance_weights.get(key, 1.0)
        rewards_sum += importance * value
        

    return rewards_sum / sum(dynamic_importance_weights.values()), reward_components
