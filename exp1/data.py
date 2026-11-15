"""Synthetic moving-shapes videos and a windowed next-frame dataset.

A training example is not a whole video. The model sees K=3 frames
in a row and tries to guess the next one. So one example is:

- input  x: shape (3, 32, 32)  — frames t, t+1, t+2 stacked as channels
- target y: shape (1, 32, 32)  — frame t+3

A 20-frame video therefore yields 17 windows (20 - 3). Those windows
overlap a lot: neighbors share two frames. If we put one window in
train and its neighbor in test, the test set would contain near-copies
of training data and the score would look better than it is. That is
why whole videos go to train, val, or test, never pieces of the same
video in two places.

The generator is deterministic: the same seed always draws the same
videos. Shapes move at constant velocity and bounce off the walls, so
when a prediction later fails we can usually say why (a bounce, a
crossing, or error compounding over a long rollout).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# These match the Step 1 spec and AGENTS.md (K, size, clip length).
K = 3
FRAME_SIZE = 32
SEQ_LEN = 20

# Default split is configurable; pass other counts to make_splits.
# AGENTS.md's later full-train recipe (1000/200) can be passed explicitly.
DEFAULT_N_TRAIN = 800
DEFAULT_N_VAL = 100
DEFAULT_N_TEST = 100

# Shape size and speed, in pixels and pixels-per-frame.
_MIN_RADIUS = 2.0
_MAX_RADIUS = 3.5
_MAX_SPEED = 2.0
_MIN_SPEED = 0.5

_CIRCLE = 0
_SQUARE = 1


@dataclass(frozen=True)
class VideoBatch:
    """A stack of videos plus a unique id for each one.

    videos:    (N, T, H, W) float32 pixels in [0, 1]
    video_ids: (N,) int64, unique within one make_splits() call
    """

    videos: np.ndarray
    video_ids: np.ndarray


def _as_generator(rng: int | np.random.Generator) -> np.random.Generator:
    """Accept either a seed or an already-started Generator.

    A seed always starts a fresh stream (same seed => same videos).
    A Generator is used as-is and advances, so make_splits can draw
    train, then val, then test without repeating the train videos.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _random_velocity(rng: np.random.Generator) -> np.ndarray:
    """A 2-D velocity that is not almost zero.

    If a shape barely moves, 'guess the next frame' is too easy:
    just copying the last frame already gets a near-perfect score.
    """
    for _ in range(32):
        velocity = rng.uniform(-_MAX_SPEED, _MAX_SPEED, size=2)
        if float(np.linalg.norm(velocity)) >= _MIN_SPEED:
            return velocity
    return np.array([_MIN_SPEED, 0.0], dtype=np.float64)


def _draw_shape(
    frame: np.ndarray,
    kind: int,
    cx: float,
    cy: float,
    radius: float,
    xx: np.ndarray,
    yy: np.ndarray,
) -> None:
    """Paint one filled circle or square onto a frame, adding ink.

    xx/yy are pixel coordinates (column, row). Overlapping shapes
    add together and are clipped to 1 later, so a crossing shows up
    as a brighter patch — useful when we study failures.
    """
    if kind == _CIRCLE:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    else:
        mask = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
    frame += mask.astype(np.float32)


def make_videos(
    n: int,
    rng: int | np.random.Generator,
    *,
    T: int = SEQ_LEN,
    S: int = FRAME_SIZE,
    id_start: int = 0,
) -> VideoBatch:
    """Draw n grayscale videos of 1–2 shapes bouncing around a box.

    Each video is (T, S, S) float32 in [0, 1]. Shapes are circles or
    axis-aligned squares. They move at constant velocity and reverse
    a component when that side of the shape hits a wall, so they stay
    on screen.

    `rng` may be an int seed or a numpy Generator. See `_as_generator`
    for why both are allowed.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if T < 1 or S < 1:
        raise ValueError(f"T and S must be >= 1, got T={T}, S={S}")

    generator = _as_generator(rng)
    # yy[row, col] is the y coordinate; xx[row, col] is the x coordinate.
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    videos = np.zeros((n, T, S, S), dtype=np.float32)

    for i in range(n):
        n_shapes = int(generator.integers(1, 3))  # 1 or 2
        for _ in range(n_shapes):
            kind = int(generator.integers(0, 2))  # circle or square
            radius = float(generator.uniform(_MIN_RADIUS, _MAX_RADIUS))
            # Keep the whole shape inside the frame at the start.
            low = radius
            high = S - 1 - radius
            if high <= low:
                raise ValueError(f"frame size S={S} is too small for radius={radius}")
            position = generator.uniform(low, high, size=2)
            velocity = _random_velocity(generator)

            for t in range(T):
                # Draw first, then step. Frame 0 uses the starting pose.
                _draw_shape(
                    videos[i, t],
                    kind,
                    float(position[0]),
                    float(position[1]),
                    radius,
                    xx,
                    yy,
                )
                position = position + velocity
                for dim in range(2):
                    lo = radius
                    hi = S - 1 - radius
                    if position[dim] < lo or position[dim] > hi:
                        velocity[dim] *= -1
                        position[dim] = np.clip(position[dim], lo, hi)

    videos = np.clip(videos, 0.0, 1.0)
    video_ids = np.arange(id_start, id_start + n, dtype=np.int64)
    return VideoBatch(videos=videos, video_ids=video_ids)


def make_splits(
    n_train: int = DEFAULT_N_TRAIN,
    n_val: int = DEFAULT_N_VAL,
    n_test: int = DEFAULT_N_TEST,
    *,
    seed: int,
    T: int = SEQ_LEN,
    S: int = FRAME_SIZE,
) -> tuple[VideoBatch, VideoBatch, VideoBatch]:
    """Build train/val/test by whole video, from one seeded generator.

    Videos are drawn in order (train, then val, then test) so the three
    sets cannot accidentally share clips. Ids are unique across the
    three batches: 0 .. n_train-1, then n_train .., and so on.
    """
    rng = np.random.default_rng(seed)
    train = make_videos(n_train, rng=rng, T=T, S=S, id_start=0)
    val = make_videos(n_val, rng=rng, T=T, S=S, id_start=n_train)
    test = make_videos(n_test, rng=rng, T=T, S=S, id_start=n_train + n_val)
    return train, val, test


class MovingShapesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Slice a VideoBatch into (K-frame input, next-frame target) windows.

    Windows are stored in video-major order: all 17 windows of video 0,
    then video 1, and so on. Window i of a video starts at frame i and
    predicts frame i+K.
    """

    def __init__(self, batch: VideoBatch, *, k: int = K) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if batch.videos.ndim != 4:
            raise ValueError(
                f"videos must have shape (N, T, H, W), got {batch.videos.shape}"
            )
        n_videos, n_frames, _, _ = batch.videos.shape
        if n_frames <= k:
            raise ValueError(f"need more than K={k} frames per video, got T={n_frames}")
        if len(batch.video_ids) != n_videos:
            raise ValueError("video_ids length must match the number of videos")

        self.videos = batch.videos
        self.video_ids = batch.video_ids
        self.k = k
        self.windows_per_video = n_frames - k
        self._length = n_videos * self.windows_per_video

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> tuple[int, int]:
        """Turn a flat window index into (video_index, start_frame)."""
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        video_index, start = divmod(index, self.windows_per_video)
        return video_index, start

    def video_id_at(self, index: int) -> int:
        """Return the source video id for the window at `index`."""
        video_index, _ = self._locate(index)
        return int(self.video_ids[video_index])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_index, start = self._locate(index)
        clip = self.videos[video_index]
        # torch.tensor(...) copies, so later training cannot mutate the source.
        x = torch.tensor(clip[start : start + self.k])
        y = torch.tensor(clip[start + self.k : start + self.k + 1])
        return x, y


def save_sample_window(
    path: str | Path,
    *,
    seed: int = 0,
    n_candidates: int = 8,
) -> Path:
    """Save a labeled picture of one window: 3 inputs and the target.

    We pick the window whose target differs most from the last input
    frame, so the motion is easy to see in the figure.
    """
    import matplotlib.pyplot as plt

    batch = make_videos(n_candidates, rng=seed)
    dataset = MovingShapesDataset(batch)
    best_index = 0
    best_delta = -1.0
    for i in range(len(dataset)):
        x, y = dataset[i]
        delta = float((x[-1] - y[0]).abs().mean())
        if delta > best_delta:
            best_delta = delta
            best_index = i

    x, y = dataset[best_index]
    _video_index, start = dataset._locate(best_index)
    video_id = dataset.video_id_at(best_index)
    frames = [x[0], x[1], x[2], y[0]]
    titles = [
        f"input t={start}",
        f"input t={start + 1}",
        f"input t={start + 2}",
        f"target t={start + 3}",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.6))
    for ax, frame, title in zip(axes, frames, titles, strict=True):
        ax.imshow(frame.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"One window (K=3): video id {video_id}, start frame {start}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    # Writes the Step 1 review figure next to this file's project root.
    out = Path(__file__).resolve().parents[1] / "results" / "step1_sample.png"
    written = save_sample_window(out)
    print(f"wrote {written}")
