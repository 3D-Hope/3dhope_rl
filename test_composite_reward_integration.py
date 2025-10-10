"""
Test script to verify composite reward integration with DDPO helpers.
"""

import torch
from physical_constraint_rewards.commons import get_composite_reward


def test_composite_reward_integration():
    """Test that composite reward works with the expected format."""
    
    print("="*60)
    print("Testing Composite Reward Integration")
    print("="*60)
    
    device = 'cpu'
    num_classes = 22
    batch_size = 2
    num_objects = 12
    
    # Create sample scenes (B=2, N=12, V=30)
    # Format: one_hot(22) + position(3) + size(3) + orientation(2)
    scenes = torch.zeros(batch_size, num_objects, 30, device=device)
    
    # Scene 1: Bedroom with chair on ground
    scenes[0, 0, 4] = 1.0  # Chair class
    scenes[0, 0, num_classes:num_classes+3] = torch.tensor([0.0, -0.98, 0.0])  # Position (on ground)
    scenes[0, 0, num_classes+3:num_classes+6] = torch.tensor([0.1, 0.1, 0.1])  # Small size
    scenes[0, 0, num_classes+6:num_classes+8] = torch.tensor([1.0, 0.0])  # Orientation
    
    # Scene 2: Living room with sofa floating
    scenes[1, 0, 16] = 1.0  # Sofa class
    scenes[1, 0, num_classes:num_classes+3] = torch.tensor([0.0, 0.5, 0.0])  # Position (floating)
    scenes[1, 0, num_classes+3:num_classes+6] = torch.tensor([0.2, 0.15, 0.2])  # Larger size
    scenes[1, 0, num_classes+6:num_classes+8] = torch.tensor([1.0, 0.0])  # Orientation
    
    # Fill rest with empty objects
    for b in range(batch_size):
        for i in range(1, num_objects):
            scenes[b, i, 21] = 1.0  # Empty class
    
    print(f"\nScene tensor shape: {scenes.shape}")
    print(f"Device: {device}")
    
    # Test 1: Default importance weights (from commons.py)
    print("\n" + "-"*60)
    print("Test 1: Default importance weights")
    print("-"*60)
    
    total_rewards_1, components_1 = get_composite_reward(
        scenes=scenes,
        num_classes=num_classes,
        room_type='bedroom',
    )
    
    print(f"\nTotal rewards: {total_rewards_1}")
    print(f"Reward components:")
    for name, values in components_1.items():
        print(f"  {name}: {values}")
    
    # Test 2: Custom importance weights
    print("\n" + "-"*60)
    print("Test 2: Custom importance weights")
    print("-"*60)
    
    custom_weights = {
        'must_have_furniture': 2.0,  # Double importance
        'gravity': 1.5,
        'non_penetration': 1.0,
        'object_count': 0.5,
    }
    
    total_rewards_2, components_2 = get_composite_reward(
        scenes=scenes,
        num_classes=num_classes,
        importance_weights=custom_weights,
        room_type='living_room',
    )
    
    print(f"\nTotal rewards: {total_rewards_2}")
    print(f"Reward components:")
    for name, values in components_2.items():
        print(f"  {name}: {values}")
    
    # Verify shapes
    assert total_rewards_1.shape == (batch_size,), f"Expected shape ({batch_size},), got {total_rewards_1.shape}"
    assert total_rewards_2.shape == (batch_size,), f"Expected shape ({batch_size},), got {total_rewards_2.shape}"
    
    # Verify all components are present
    expected_components = {'gravity', 'non_penetration', 'must_have_furniture', 'object_count'}
    assert set(components_1.keys()) == expected_components, f"Missing components: {expected_components - set(components_1.keys())}"
    
    print("\n" + "="*60)
    print("✓ All integration tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_composite_reward_integration()
