"""
Test script to verify composite reward integration with DDPO helpers.
"""

import torch

from physical_constraint_rewards.commons import get_composite_reward


def test_composite_reward_integration():
    """Test that composite reward works with the expected format."""

    print("=" * 60)
    print("Testing Composite Reward Integration")
    print("=" * 60)

    device = "cpu"
    num_classes = 22
    batch_size = 2
    num_objects = 12

    # Create sample scenes (B=2, N=12, V=30)
    # Format: one_hot(22) + position(3) + size(3) + orientation(2)
    scenes = torch.zeros(batch_size, num_objects, 30, device=device)

    # Scene 1: Bedroom with chair on ground
    scenes[0, 0, 4] = 1.0  # Chair class
    scenes[0, 0, num_classes : num_classes + 3] = torch.tensor(
        [0.0, -0.98, 0.0]
    )  # Position (on ground)
    scenes[0, 0, num_classes + 3 : num_classes + 6] = torch.tensor(
        [0.1, 0.1, 0.1]
    )  # Small size
    scenes[0, 0, num_classes + 6 : num_classes + 8] = torch.tensor(
        [1.0, 0.0]
    )  # Orientation

    # Scene 2: Living room with sofa floating
    scenes[1, 0, 16] = 1.0  # Sofa class
    scenes[1, 0, num_classes : num_classes + 3] = torch.tensor(
        [0.0, 0.5, 0.0]
    )  # Position (floating)
    scenes[1, 0, num_classes + 3 : num_classes + 6] = torch.tensor(
        [0.2, 0.15, 0.2]
    )  # Larger size
    scenes[1, 0, num_classes + 6 : num_classes + 8] = torch.tensor(
        [1.0, 0.0]
    )  # Orientation

    # Fill rest with empty objects
    for b in range(batch_size):
        for i in range(1, num_objects):
            scenes[b, i, 21] = 1.0  # Empty class

    print(f"\nScene tensor shape: {scenes.shape}")
    print(f"Device: {device}")

    # Test 1: Default importance weights (from commons.py)
    print("\n" + "-" * 60)
    print("Test 1: Default importance weights")
    print("-" * 60)

    total_rewards_1, components_1 = get_composite_reward(
        scenes=scenes,
        num_classes=num_classes,
        room_type="bedroom",
    )

    print(f"\nTotal rewards: {total_rewards_1}")
    print(f"Reward components:")
    for name, values in components_1.items():
        print(f"  {name}: {values}")

    # Test 2: Custom importance weights
    print("\n" + "-" * 60)
    print("Test 2: Custom importance weights")
    print("-" * 60)

    custom_weights = {
        "must_have_furniture": 2.0,  # Double importance
        "gravity": 1.5,
        "non_penetration": 1.0,
        "object_count": 0.5,
    }

    total_rewards_2, components_2 = get_composite_reward(
        scenes=scenes,
        num_classes=num_classes,
        importance_weights=custom_weights,
        room_type="living_room",
    )

    print(f"\nTotal rewards: {total_rewards_2}")
    print(f"Reward components:")
    for name, values in components_2.items():
        print(f"  {name}: {values}")

    # Verify shapes
    assert total_rewards_1.shape == (
        batch_size,
    ), f"Expected shape ({batch_size},), got {total_rewards_1.shape}"
    assert total_rewards_2.shape == (
        batch_size,
    ), f"Expected shape ({batch_size},), got {total_rewards_2.shape}"

    # Verify all components are present
    expected_components = {
        "gravity",
        "non_penetration",
        "must_have_furniture",
        "object_count",
    }
    assert (
        set(components_1.keys()) == expected_components
    ), f"Missing components: {expected_components - set(components_1.keys())}"

    print("\n" + "="*60)
    print("✓ All integration tests passed!")
    print("="*60)


def test_composite_plus_task_integration():
    """Test composite + task reward integration."""
    
    print("\n" + "="*60)
    print("Testing Composite + Task Reward Integration")
    print("="*60)
    
    device = 'cpu'
    num_classes = 22
    batch_size = 3
    num_objects = 12
    
    # Create sample scenes
    scenes = torch.zeros(batch_size, num_objects, 30, device=device)
    
    # Scene 1: Bedroom with sofa (task satisfied)
    scenes[0, 0, 16] = 1.0  # Sofa class
    scenes[0, 0, num_classes:num_classes+3] = torch.tensor([0.0, -0.98, 0.0])
    scenes[0, 0, num_classes+3:num_classes+6] = torch.tensor([0.2, 0.15, 0.2])
    scenes[0, 0, num_classes+6:num_classes+8] = torch.tensor([1.0, 0.0])
    
    # Scene 2: Bedroom without sofa (task not satisfied)
    scenes[1, 0, 4] = 1.0  # Chair class
    scenes[1, 0, num_classes:num_classes+3] = torch.tensor([0.0, -0.98, 0.0])
    scenes[1, 0, num_classes+3:num_classes+6] = torch.tensor([0.1, 0.1, 0.1])
    scenes[1, 0, num_classes+6:num_classes+8] = torch.tensor([1.0, 0.0])
    
    # Scene 3: Living room with sofa floating (physics violation + task satisfied)
    scenes[2, 0, 16] = 1.0  # Sofa class
    scenes[2, 0, num_classes:num_classes+3] = torch.tensor([0.0, 0.5, 0.0])  # Floating!
    scenes[2, 0, num_classes+3:num_classes+6] = torch.tensor([0.2, 0.15, 0.2])
    scenes[2, 0, num_classes+6:num_classes+8] = torch.tensor([1.0, 0.0])
    
    # Fill rest with empty
    for b in range(batch_size):
        for i in range(1, num_objects):
            scenes[b, i, 21] = 1.0
    
    # Mock config
    from omegaconf import OmegaConf
    
    cfg = OmegaConf.create({
        'custom': {'num_classes': num_classes},
        'ddpo': {
            'composite_plus_task': {
                'task_reward_type': 'has_sofa',
                'task_weight': 2.0,
                'room_type': 'living_room',
                'importance_weights': {
                    'must_have_furniture': 1.0,
                    'gravity': 1.0,
                    'non_penetration': 1.0,
                    'object_count': 0.5,
                }
            }
        }
    })
    
    # Test composite + task reward
    from steerable_scene_generation.algorithms.scene_diffusion.ddpo_helpers import composite_plus_task_reward
    
    print("\n" + "-"*60)
    print("Test: Composite + Task Reward (has_sofa)")
    print("-"*60)
    
    total_rewards, components = composite_plus_task_reward(
        scenes=scenes,
        scene_vec_desc=None,  # Not used in this test
        cfg=cfg,
    )
    
    print(f"\nTotal rewards: {total_rewards}")
    print(f"\nReward components:")
    for name, values in components.items():
        print(f"  {name}: {values}")
    
    # Verify shapes
    assert total_rewards.shape == (batch_size,), f"Expected shape ({batch_size},), got {total_rewards.shape}"
    
    # Verify task reward component
    task_rewards = components['task_reward']
    assert task_rewards[0] == 1.0, "Scene 1 should have sofa (task satisfied)"
    assert task_rewards[1] == 0.0, "Scene 2 should not have sofa (task not satisfied)"
    assert task_rewards[2] == 1.0, "Scene 3 should have sofa (task satisfied)"
    
    # Scene 1 (sofa on ground) should have best total reward
    # Scene 2 (no sofa) should have worst total reward due to missing task
    # Scene 3 (sofa floating) should be in between (task ok, physics bad)
    assert total_rewards[0] > total_rewards[2], "Grounded sofa should beat floating sofa"
    
    print("\n" + "="*60)
    print("✓ Composite + Task integration tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_composite_reward_integration()
    test_composite_plus_task_integration()
