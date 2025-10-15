# takes the user input
import argparse
from call_agent import ConstraintGenerator, RewardGenerator
import os
import torch
from get_reward_stats_from_baseline import get_reward_stats_from_baseline
from scale_raw_rewards import RewardNormalizer
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from steerable_scene_generation.utils.omegaconf import register_resolvers
from commons import import_dynamic_reward_functions


@hydra.main(
    version_base=None, config_path="../configurations", config_name="config"
)
def main(cfg: DictConfig):
    register_resolvers()
    OmegaConf.resolve(cfg)
    algorithm_config = cfg.algorithm
    user_input = algorithm_config.ddpo.dynamic_constraint_rewards.user_query
    reward_code_dir = algorithm_config.ddpo.dynamic_constraint_rewards.reward_code_dir
    room_type = algorithm_config.ddpo.dynamic_constraint_rewards.room_type
    os.makedirs(reward_code_dir, exist_ok=True)

    # TODO: Uncomment this when we have the llm working.
    # Call llm to parse the user input and create a list of reward functions to follow the user instruction.
    # constraint_generator = ConstraintGenerator()
    # reward_generator = RewardGenerator()

    # constraints = constraint_generator.generate_constraints({"room_type": room_type, "query": user_input})
    # print(constraints)

    # for i in range(len(constraints["constraints"])):
    #     constraint = constraints["constraints"][i]
    #     print(f"[SAUGAT] Constraint {i+1}: {constraint}")
    #     result = reward_generator.generate_reward({"room_type": room_type, "query": user_input, "constraint": constraint})
    #     print(f"[SAUGAT] Reward code {i+1}: {result["raw_response"]}")
    #     with open(f"{reward_code_dir}/{i}_code.py", "w") as f:
    #         f.write(result["raw_response"])

    # Test each reward function individually.

    
    get_reward_functions, test_reward_functions = import_dynamic_reward_functions(reward_code_dir)


    # Test each reward function individually.
    for file in test_reward_functions:
        test_reward_functions[file]()

    stats = get_reward_stats_from_baseline(get_reward_functions, num_scenes=162, config=cfg)
    print("Stats: ", stats)


    with open("stats.json", "w") as f:
        json.dump(stats, f)


    # Testing normaizer
    # reward_normalizer = RewardNormalizer(stats)

    # for key, value in get_reward_functions.items():
    #     normalized_reward = reward_normalizer.normalize(key, torch.tensor([1.0, 2.0, 3.0]))
    #     print(f"Normalized reward for {key}: {normalized_reward}")

if __name__ == "__main__":
    main()

# Run Command:
# python take_user_instruction.py dataset=custom_scene algorithm=scene_diffuser_flux_transformer