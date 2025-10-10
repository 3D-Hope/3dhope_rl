# Reward Scaling Strategy

## Philosophy: Normalize First, Then Weight by Importance

### Two-Step Process:

#### Step 1: Normalization (Technical)
Scale all rewards to **[-1, 0]** range where:
- `0` = perfect (no violation)
- `-1` = worst case violation

This makes all rewards technically comparable.

#### Step 2: Importance Weighting (Design)
Apply weights based on **interior design priorities**, not scale issues.

---

## Current Raw Reward Scales (Empirical Analysis)

| Reward | Best | Worst | Range | Notes |
|--------|------|-------|-------|-------|
| **Gravity** | -0.05 | -2.7 | 2.7 | Distance² penalty for floating |
| **Non-penetration** | 0 | -25 | 25 | **10x larger!** Sum of all overlaps |
| **Must-have** | 0 | -10 | 10 | Binary: has bed or not |
| **Object count (NLL)** | -1.2 | -9.2 | 8 | log(probability) |

---

## Strategy Comparison

### Option A: Current Approach (Scale-Based Weights)
```python
reward_weights = {
    'gravity': 5.0,              # Amplify small range
    'non_penetration': 0.5,      # Reduce large range
    'must_have_furniture': 1.0,  # Keep moderate
    'object_count': 1.0,         # Keep moderate
}
```

**Pros:**
- Simple, one-step
- Directly addresses scale imbalance

**Cons:**
- Conflates two concerns: normalization + importance
- Weights don't reflect design priorities (why is gravity "5x more important"?)
- Hard to interpret weights

---

### Option B: Normalize Then Weight (RECOMMENDED)

```python
# Step 1: Normalization factors (technical)
normalization = {
    'gravity': 1.0 / 2.7,           # Scale to [-1, 0]
    'non_penetration': 1.0 / 25.0,  # Scale to [-1, 0]
    'must_have_furniture': 1.0 / 10.0,  # Scale to [-1, 0]
    'object_count': 1.0 / 8.0,      # Scale to [-1, 0]
}

# Step 2: Importance weights (design decision)
importance_weights = {
    'gravity': 1.0,              # Critical: physics
    'non_penetration': 1.0,      # Critical: physics
    'must_have_furniture': 1.5,  # Very important: function
    'object_count': 0.7,         # Important: aesthetics
}

# Final weight = normalize * importance
final_weights = {
    'gravity': 1.0/2.7 * 1.0 = 0.37,
    'non_penetration': 1.0/25.0 * 1.0 = 0.04,
    'must_have_furniture': 1.0/10.0 * 1.5 = 0.15,
    'object_count': 1.0/8.0 * 0.7 = 0.09,
}
```

**Pros:**
- ✅ Clear separation: technical normalization vs design priorities
- ✅ Weights are interpretable (1.5 = 50% more important)
- ✅ Easy to adjust priorities without worrying about scales
- ✅ Self-documenting code

**Cons:**
- Slightly more complex (but clearer!)

---

## Interior Designer's Importance Ranking

### Tier 1: Critical (weight = 1.0)
- **Gravity following**: Floating furniture is impossible
- **Non-penetration**: Overlapping objects are impossible

### Tier 2: Very Important (weight = 1.2-1.5)
- **Must-have furniture**: Room must be functional
  - Bedroom without bed = not a bedroom
  - Weight: **1.5** (highest priority)

### Tier 3: Important (weight = 0.7-0.9)
- **Object count**: Realistic clutter level
  - Too empty = feels staged
  - Too cluttered = uncomfortable
  - Weight: **0.7** (lower than physics/function)

### Future Rewards (estimates)
- **Chair accessibility**: 0.8 (aesthetics/ergonomics)
- **Night tables placement**: 0.6 (nice-to-have)
- **Path clearance**: 0.9 (important for navigation)

---

## Recommended Implementation

### Option 1: Separate Normalization (Cleaner)
```python
def normalize_reward(reward, max_penalty):
    """Normalize reward to [-1, 0] range."""
    return reward / max_penalty

# In get_composite_reward():
NORMALIZATION_FACTORS = {
    'gravity': 2.7,
    'non_penetration': 25.0,
    'must_have_furniture': 10.0,
    'object_count': 8.0,
}

IMPORTANCE_WEIGHTS = {
    'gravity': 1.0,              # Critical
    'non_penetration': 1.0,      # Critical
    'must_have_furniture': 1.5,  # Very important
    'object_count': 0.7,         # Important
}

# Normalize then weight
for name, raw_reward in reward_components.items():
    normalized = raw_reward / NORMALIZATION_FACTORS[name]
    total_reward += normalized * IMPORTANCE_WEIGHTS[name]
```

### Option 2: Combined Weights (Current - Simpler)
```python
# Pre-computed: final_weight = (1/normalization) * importance
reward_weights = {
    'gravity': 0.37,           # (1/2.7) * 1.0
    'non_penetration': 0.04,   # (1/25) * 1.0
    'must_have_furniture': 0.15,  # (1/10) * 1.5
    'object_count': 0.09,      # (1/8) * 0.7
}
```

---

## Final Recommendation

**Use Option 1 (Separate Normalization)** for:
- 🎯 **Clarity**: Code documents both normalization AND importance
- 🔧 **Maintainability**: Easy to adjust importance without recalculating
- 📊 **Debugging**: Can see raw, normalized, and weighted values separately

**Implementation:**
1. Add `NORMALIZATION_FACTORS` dict to commons.py
2. Add `IMPORTANCE_WEIGHTS` dict to commons.py (defaults)
3. Allow users to override importance weights (not normalization)
4. Update get_composite_reward() to normalize first, then weight

**User-facing API:**
```python
# User only needs to think about importance, not scales!
total_reward, components = get_composite_reward(
    scenes,
    importance_weights={
        'must_have_furniture': 2.0,  # Double importance!
        'object_count': 0.5,         # Half importance
        # Others use defaults: gravity=1.0, non_penetration=1.0
    }
)
```

This is much more intuitive than the current:
```python
# Current: User has to know about internal scales
reward_weights={
    'gravity': 5.0,  # Why 5.0? ❓
    'non_penetration': 0.5,  # Why 0.5? ❓
}
```

---

## Example: All Equal Importance

If all rewards equally important (`importance = 1.0` for all):

```python
Effective weights = 1/normalization:
- gravity: 0.37  (1/2.7)
- non_penetration: 0.04  (1/25)
- must_have: 0.10  (1/10)
- count: 0.125  (1/8)

Total penalty for perfect scene: 0
Total penalty for worst scene: -4.0  (1+1+1+1)
```

## Example: Prioritize Function

```python
importance_weights = {
    'must_have_furniture': 2.0,  # 2x importance
    'gravity': 1.0,
    'non_penetration': 1.0,
    'object_count': 0.5,  # 0.5x importance
}

Total penalty for missing bed: -2.0  (normalized -1 * 2.0)
Total penalty for floating: -1.0
Total penalty for overlap: -1.0
Total penalty for bad count: -0.5
```

Much clearer! 🎉
