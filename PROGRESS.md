# PROGRESS.md
Updated: Sun Nov 15, 2026, ET    Phase: P0 Exp 1    Branch: feat/data

## Current goal
Step 1 data is in. Next is Step 2: copy-last and mean-frame baselines plus MSE and SSIM, on a new branch off `main` after this lands.

## Done
- Cursor setup files drafted (`AGENTS.md`, `.cursor/rules/exp1.mdc`, this file, ignore files).
- **Python env (Windows laptop, Fri Nov 13, 2026):** uv + Python 3.13 (`.python-version`), `pyproject.toml` / `uv.lock`, CPU `torch==2.11.0+cpu`, numpy, matplotlib, dev tools (ruff, pyright, pytest, hypothesis), `tests/test_env.py`. Proof:
  - `uv run pytest -q -s tests/test_env.py` → `torch.__version__=2.11.0+cpu`, `torch.cuda.is_available()=False`, 1 passed
  - `uv run ruff check .` ; `uv run ruff format --check .` → All checks passed
  - `uv run pyright` → 0 errors
  - `uv run pytest -q` → 1 passed
- **Step 1 data (box, Sun Nov 15, 2026, branch `feat/data`, not pushed):** package `exp1/data.py` — moving square/circle videos, video-level train/val/test split, `MovingShapesDataset` windows. Tests written first (`tests/test_data.py`), then the module. Proof:
  - `uv run pytest -q tests/test_data.py` before the module → collection error `ModuleNotFoundError: No module named 'exp1'`
  - `uv run pytest -q` after the module → 14 passed (13 data + 1 env)
  - `uv run ruff check .` ; `uv run ruff format --check .` → All checks passed
  - `uv run pyright` → 0 errors, 0 warnings
  - Sample window figure: `results/step1_sample.png` (untracked; also copied to `/workspace/exp1_step1_sample.png` on the box)
- Layout deviation: Step 1 uses package `exp1/data.py` rather than a single `exp1.py`. `pyproject.toml` now has a hatchling build so `import exp1` works under `uv run`.
- Generator deviation from the World Model Lab scaffold: hard-edged circles and squares, not soft Gaussian blobs. Same kinematics idea (constant velocity, bounce, 1–2 shapes, 32×32, T=20, K=3). Flagged so later numbers are not compared blindly to the old smoke run.
- Scaffold still also lives in the World Model Lab box copy (`build_nb.py`, generated notebook, `smoke_exp1.py`). Those files are not in this repo yet.

## Decisions and why
- Split by video, not by window: adjacent windows share K−1 frames, so a window split leaks.
- Residual from the last frame, then clamp to [0, 1]: copy-last is the zero-decoder path.
- Pin Python 3.13 and torch 2.11 to match Colab runtime 2026.10. Laptop system Python is 3.14; uv manages 3.13 per project.
- Torch on laptop: PyTorch CPU index in `pyproject.toml` (`explicit = true`). Colab uses its preinstalled torch, not this lockfile slice.
- GPU work on free Colab T4; home CPU for smoke and tests.
- Commit or push only when Stephen asks.
- Default split sizes are 800/100/100 videos (configurable). The later full-train recipe in AGENTS.md (1000/200) can be passed explicitly when training starts.
- Minimum shape speed 0.5 px/frame so copy-last is not already almost perfect.
- Overlapping shapes add and clip, so a crossing is a brighter patch.

## Open questions
- Does Cursor's command sandbox work on native Windows? Docs cover Mac and Linux only.
- MetaDrive / Python 3.13 is a later-phase risk (not in scope for Exp 1).

## Next step
Step 2 on a new branch off `main` (after `feat/data` is merged or applied): copy-last and mean-frame baselines, MSE and SSIM. Teach-first before writing code. Still no full training.

## Last verified results
- Date: Sun Nov 15, 2026 (box).
- What was run: `uv run pytest -q tests/test_data.py` (failing, then passing); `uv run pytest -q` ; `uv run ruff check .` ; `uv run ruff format --check .` ; `uv run pyright` ; `uv run python -m exp1.data`.
- Outcome: data contract tests green; sample window figure written. Torch 2.11.0+cpu installed fine on the box.
- Not yet verified: baselines, training loop, multi-seed, Colab full train, `smoke_exp1.py` on Windows path. No `feat/data` push.
