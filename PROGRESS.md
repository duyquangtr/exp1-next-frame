# PROGRESS.md
Updated: Fri Nov 13, 2026, ET    Phase: P0 Exp 1    Branch: chore/setup

## Current goal
Extract `exp1.py` from the box scaffold, add real tests beyond env smoke, and fix the Windows save-path bug in `smoke_exp1.py`. No full training until checkpoints 1–2 pass.

## Done
- Cursor setup files drafted (`AGENTS.md`, `.cursor/rules/exp1.mdc`, this file, ignore files).
- **Python env (Windows laptop, Fri Nov 13, 2026):** uv + Python 3.13 (`.python-version`), `pyproject.toml` / `uv.lock`, CPU `torch==2.11.0+cpu`, numpy, matplotlib, dev tools (ruff, pyright, pytest, hypothesis), `tests/test_env.py`. Proof:
  - `uv run pytest -q -s tests/test_env.py` → `torch.__version__=2.11.0+cpu`, `torch.cuda.is_available()=False`, 1 passed
  - `uv run ruff check .` ; `uv run ruff format --check .` → All checks passed
  - `uv run pyright` → 0 errors
  - `uv run pytest -q` → 1 passed
- Scaffold exists in the World Model Lab box copy (`build_nb.py`, generated notebook, `smoke_exp1.py`). Functions still live there; `exp1.py` not extracted yet.
- Smoke run only (seed 0, 200 train / 50 val, 2 epochs, CPU, box): conv val MSE 0.00087 vs copy-last 0.00557; conv val SSIM 0.968 vs 0.885. Not a result. Rollout-10 and mean-frame were not recorded.

## Decisions and why
- Split by video, not by window: adjacent windows share K−1 frames, so a window split leaks.
- Residual from the last frame, then clamp to [0, 1]: copy-last is the zero-decoder path.
- Pin Python 3.13 and torch 2.11 to match Colab runtime 2026.10. Laptop system Python is 3.14; uv manages 3.13 per project.
- Torch on laptop: PyTorch CPU index in `pyproject.toml` (`explicit = true`). Colab uses its preinstalled torch, not this lockfile slice.
- GPU work on free Colab T4; home CPU for smoke and tests.
- Commit or push only when Stephen asks.

## Open questions
- Does Cursor's command sandbox work on native Windows? Docs cover Mac and Linux only.
- MetaDrive / Python 3.13 is a later-phase risk (not in scope for Exp 1).

## Next step
Teach-first: explain `make_videos` and `windows` (shapes, dtypes, axes) and what would silently break if K went from 3 to 1. Then plan portable save path in `smoke_exp1.py` when copying scaffold in.

## Last verified results
- Date: Fri Nov 13, 2026 (Windows laptop).
- What was run: `uv run pytest -q -s tests/test_env.py` ; `uv run ruff check .` ; `uv run pyright` ; `uv run pytest -q`.
- Outcome: env and toolchain green; one env test only (no `exp1.py` yet).
- Not yet verified: `smoke_exp1.py` on Windows path, full pytest suite, multi-seed, mean-frame, Colab full train.
