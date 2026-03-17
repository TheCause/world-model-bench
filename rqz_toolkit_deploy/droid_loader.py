"""
DROID Dataset Loader for Track 1/Track 2 benchmarks.

Loads trajectories from HuggingFace `lerobot/droid_100` (or full droid).
Groups frames by episode. Extracts actions [7], states [7], images [H,W,C].

Contract reference: contract.md v1.0, section 4.2
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class Trajectory:
    """A single robot manipulation trajectory."""
    episode_id: int
    task_index: int
    actions: np.ndarray       # [T, 7] float32
    states: np.ndarray        # [T, 7] float32
    timestamps: np.ndarray    # [T] float32
    language: str             # task description
    n_steps: int
    # Images loaded separately (too large for memory)
    image_paths: Optional[List[str]] = None


def load_droid_trajectories(
    n_episodes: int = 20,
    min_steps: int = 80,
    seed: int = 42,
    dataset_name: str = "lerobot/droid_100",
) -> List[Trajectory]:
    """
    Load DROID trajectories from HuggingFace.

    Args:
        n_episodes: number of episodes to load
        min_steps: minimum trajectory length (contract: 80 steps)
        seed: for stratified selection
        dataset_name: HuggingFace dataset identifier

    Returns:
        List of Trajectory objects, sorted by episode_id
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")

    # Group frames by episode
    episodes: Dict[int, Dict] = {}
    for idx in range(len(ds)):
        row = ds[idx]
        ep_id = row["episode_index"]
        if ep_id not in episodes:
            episodes[ep_id] = {
                "actions": [], "states": [], "timestamps": [],
                "task_index": row["task_index"],
                "language": row.get("language_instruction", ""),
            }
        episodes[ep_id]["actions"].append(row["action"])
        episodes[ep_id]["states"].append(row["observation.state"])
        episodes[ep_id]["timestamps"].append(row["timestamp"])

    # Filter by min length
    valid = {k: v for k, v in episodes.items() if len(v["actions"]) >= min_steps}
    print(f"  {len(episodes)} episodes total, {len(valid)} >= {min_steps} steps")

    # Stratified selection by task (seed 42, contract section 4.5)
    rng = np.random.RandomState(seed)
    task_groups: Dict[int, List[int]] = {}
    for ep_id, ep in valid.items():
        t = ep["task_index"]
        if t not in task_groups:
            task_groups[t] = []
        task_groups[t].append(ep_id)

    # Select episodes: round-robin across tasks
    selected = []
    task_ids = sorted(task_groups.keys())
    rng.shuffle(task_ids)
    idx = 0
    while len(selected) < n_episodes and idx < len(task_ids) * 10:
        task = task_ids[idx % len(task_ids)]
        eps = task_groups[task]
        if eps:
            chosen = rng.choice(len(eps))
            selected.append(eps.pop(chosen))
        idx += 1

    print(f"  Selected {len(selected)} episodes from {len(task_groups)} tasks")

    # Build Trajectory objects
    trajectories = []
    for ep_id in sorted(selected):
        ep = valid[ep_id]
        trajectories.append(Trajectory(
            episode_id=ep_id,
            task_index=ep["task_index"],
            actions=np.array(ep["actions"], dtype=np.float32),
            states=np.array(ep["states"], dtype=np.float32),
            timestamps=np.array(ep["timestamps"], dtype=np.float32),
            language=ep["language"],
            n_steps=len(ep["actions"]),
        ))

    return trajectories


def make_synthetic_trajectories(
    n_episodes: int = 20,
    n_steps: int = 200,
    action_dim: int = 7,
    state_dim: int = 7,
    seed: int = 42,
) -> List[Trajectory]:
    """
    Create synthetic trajectories for dry-run testing (no HuggingFace needed).
    """
    rng = np.random.RandomState(seed)
    trajectories = []
    for ep_id in range(n_episodes):
        T = n_steps + rng.randint(-20, 20)
        trajectories.append(Trajectory(
            episode_id=ep_id,
            task_index=ep_id % 4,
            actions=rng.randn(T, action_dim).astype(np.float32),
            states=rng.randn(T, state_dim).astype(np.float32),
            timestamps=np.arange(T, dtype=np.float32) / 15.0,
            language=f"synthetic task {ep_id % 4}",
            n_steps=T,
        ))
    return trajectories
