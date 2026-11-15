"""Tests for the moving-shapes data pipeline.

These asserts are the contract for Step 1. They should fail until
exp1/data.py exists and does the right thing. Do not weaken them
to make a bad implementation pass.

Why these checks matter:
- Shape and dtype mistakes show up later as silent training bugs.
- A window-level train/test split would leak overlapping frames
  and inflate scores. We split by whole video and check ids.
- The target must be the frame immediately after the K=3 input,
  not one of the input frames itself.
"""

from __future__ import annotations

import numpy as np
import torch

from exp1.data import (
    DEFAULT_N_TEST,
    DEFAULT_N_TRAIN,
    DEFAULT_N_VAL,
    FRAME_SIZE,
    SEQ_LEN,
    K,
    MovingShapesDataset,
    VideoBatch,
    make_splits,
    make_videos,
)


def test_default_constants_match_the_step1_spec() -> None:
    # K=3 past frames, 32x32, about 20-frame clips, 800/100/100 split.
    assert K == 3
    assert FRAME_SIZE == 32
    assert SEQ_LEN == 20
    assert (DEFAULT_N_TRAIN, DEFAULT_N_VAL, DEFAULT_N_TEST) == (800, 100, 100)


def test_make_videos_returns_n_clips_with_expected_shape() -> None:
    batch = make_videos(5, rng=0)
    assert isinstance(batch, VideoBatch)
    assert batch.videos.shape == (5, SEQ_LEN, FRAME_SIZE, FRAME_SIZE)
    assert batch.video_ids.shape == (5,)
    assert len(np.unique(batch.video_ids)) == 5


def test_frames_are_float32_and_stay_in_unit_interval() -> None:
    videos = make_videos(8, rng=0).videos
    assert videos.dtype == np.float32
    assert videos.min() >= 0.0
    assert videos.max() <= 1.0
    # A real shape was drawn, not a blank canvas.
    assert videos.max() > 0.5
    # Every frame of every video has some ink (shapes stay on screen).
    assert (videos.max(axis=(2, 3)) > 0.5).all()


def test_same_seed_reproduces_identical_videos() -> None:
    a = make_videos(4, rng=0)
    b = make_videos(4, rng=0)
    assert np.array_equal(a.videos, b.videos)
    assert np.array_equal(a.video_ids, b.video_ids)


def test_different_seeds_produce_different_videos() -> None:
    a = make_videos(4, rng=0)
    b = make_videos(4, rng=1)
    assert not np.array_equal(a.videos, b.videos)


def test_shared_generator_advances_and_does_not_repeat() -> None:
    # make_splits will pass one Generator through each split. If we
    # accidentally re-seed for val/test, those videos would copy train.
    rng = np.random.default_rng(0)
    first = make_videos(3, rng=rng, id_start=0)
    second = make_videos(3, rng=rng, id_start=3)
    assert not np.array_equal(first.videos, second.videos)
    assert set(first.video_ids).isdisjoint(set(second.video_ids))


def test_splits_do_not_share_video_ids() -> None:
    train, val, test = make_splits(8, 4, 3, seed=0)
    train_ids = set(train.video_ids.tolist())
    val_ids = set(val.video_ids.tolist())
    test_ids = set(test.video_ids.tolist())
    assert len(train_ids) == 8
    assert len(val_ids) == 4
    assert len(test_ids) == 3
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    # All ids together are unique and cover the full range.
    assert train_ids | val_ids | test_ids == set(range(8 + 4 + 3))


def test_splits_do_not_share_video_content() -> None:
    # Belt and suspenders: unique ids are not enough if the generator
    # was accidentally re-seeded and drew the same pixels twice.
    train, val, test = make_splits(8, 4, 3, seed=0)

    def pixel_keys(batch: VideoBatch) -> set[bytes]:
        return {np.ascontiguousarray(video).tobytes() for video in batch.videos}

    train_keys = pixel_keys(train)
    val_keys = pixel_keys(val)
    test_keys = pixel_keys(test)
    assert len(train_keys) == 8
    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(test_keys)


def test_same_split_seed_is_reproducible() -> None:
    a_train, a_val, a_test = make_splits(5, 3, 2, seed=7)
    b_train, b_val, b_test = make_splits(5, 3, 2, seed=7)
    assert np.array_equal(a_train.videos, b_train.videos)
    assert np.array_equal(a_val.videos, b_val.videos)
    assert np.array_equal(a_test.videos, b_test.videos)


def test_each_video_yields_t_minus_k_windows() -> None:
    batch = make_videos(6, rng=0)
    dataset = MovingShapesDataset(batch)
    windows_per_video = SEQ_LEN - K  # 20 - 3 = 17
    assert windows_per_video == 17
    assert len(dataset) == 6 * windows_per_video


def test_window_shapes_dtypes_and_value_range() -> None:
    dataset = MovingShapesDataset(make_videos(2, rng=0))
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (K, FRAME_SIZE, FRAME_SIZE)
    assert y.shape == (1, FRAME_SIZE, FRAME_SIZE)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


def test_target_is_the_frame_right_after_the_input_window() -> None:
    batch = make_videos(3, rng=0)
    dataset = MovingShapesDataset(batch)
    windows_per_video = SEQ_LEN - K

    # First window of video 0: frames 0,1,2 predict frame 3.
    x0, y0 = dataset[0]
    expected_x0 = torch.from_numpy(batch.videos[0, 0:K])
    expected_y0 = torch.from_numpy(batch.videos[0, K : K + 1])
    assert torch.equal(x0, expected_x0)
    assert torch.equal(y0, expected_y0)

    # A later window in the same video: start t=5 -> frames 5,6,7 predict 8.
    t = 5
    x, y = dataset[t]
    assert torch.equal(x, torch.from_numpy(batch.videos[0, t : t + K]))
    assert torch.equal(y, torch.from_numpy(batch.videos[0, t + K : t + K + 1]))

    # First window of video 2, to check we index by video then time.
    idx = 2 * windows_per_video
    x, y = dataset[idx]
    assert torch.equal(x, torch.from_numpy(batch.videos[2, 0:K]))
    assert torch.equal(y, torch.from_numpy(batch.videos[2, K : K + 1]))


def test_dataset_keeps_source_video_ids_for_each_window() -> None:
    # Useful later if we ever want to audit which video a window came from.
    train, _, _ = make_splits(4, 2, 2, seed=0)
    dataset = MovingShapesDataset(train)
    windows_per_video = SEQ_LEN - K
    for video_index, video_id in enumerate(train.video_ids.tolist()):
        # Every window sliced from this video should report the same id.
        start = video_index * windows_per_video
        for offset in range(windows_per_video):
            assert dataset.video_id_at(start + offset) == video_id
