import torch
import json
class RewardNormalizer:
    def __init__(self, baseline_stats_path: str, alpha: float = 1.0, beta: float = 0.99, eps: float = 1e-8):
        """
        Reward normalizer with per-reward statistics and EMA updating.

        Args:
            baseline_stats_path (str): Path to the baseline stats file.
                {
                    "gravity": {"min": -1.2, "max": 0.8, "mean": -0.3, "stddev": 0.4},
                    "non_pen": {"min": -0.05, "max": 0.0, "mean": -0.02, "stddev": 0.01}
                }
            alpha (float): tanh sensitivity factor.
            beta (float): EMA smoothing factor for min/max updates.
            eps (float): Small value to prevent division by zero.
        """
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        # Initialize stats for each reward function
        self.stats = {}
        baseline_stats = json.load(open(baseline_stats_path))
        for name, s in baseline_stats.items():
            self.stats[name] = {
                "min": torch.tensor(s["min"], dtype=torch.float32),
                "max": torch.tensor(s["max"], dtype=torch.float32)
            }

    def normalize(self, name: str, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards for a specific reward component.

        Args:
            name (str): Reward function name (e.g. "gravity").
            rewards (torch.Tensor): Tensor of shape [B, 1].

        Returns:
            torch.Tensor: Normalized rewards in [0, 1].
        """
        if name not in self.stats:
            raise ValueError(f"Unknown reward name '{name}' — initialize first via baseline stats.")

        device = rewards.device
        r_min = self.stats[name]["min"].to(device)
        r_max = self.stats[name]["max"].to(device)

        # Step 1: Min-max normalization to [-1, 1]
        denom = torch.clamp(r_max - r_min, min=self.eps)
        scaled = 2 * ((rewards - r_min) / denom) - 1

        # Step 2: Apply tanh(alpha * x)
        bounded = torch.tanh(self.alpha * scaled)

        # Step 3: Map (-1, 1) → (0, 1)
        normalized = (bounded + 1) / 2

        # Step 4: Update EMA stats
        batch_min = rewards.min()
        batch_max = rewards.max()
        self.stats[name]["min"] = self.beta * r_min + (1 - self.beta) * batch_min
        self.stats[name]["max"] = self.beta * r_max + (1 - self.beta) * batch_max

        return normalized.to(device)

    def get_stats(self) -> dict:
        """Return current moving stats as a plain dict (for logging or checkpointing)."""
        return {
            name: {
                "min": float(v["min"].cpu().item()),
                "max": float(v["max"].cpu().item())
            }
            for name, v in self.stats.items()
        }
