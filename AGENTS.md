# Repository Guidelines

## Project Structure & Module Organization
- `localinferenceapi.py` runs the FastAPI backend exposing CLIP and SAM endpoints plus shared Pydantic models.
- Training utilities live under `tools/` (`clip_kmeans_and_regress.py`, `train_clip_regression_from_YOLO.py`) and write artifacts such as `my_logreg_model.pkl` and `my_label_list.pkl`.
- Annotation UI code is bundled in `ybat-master/`; open `ybat-master/ybat.html` directly to label/export crops or switch to the CLIP training tab.
- Vendor assets stay in `CLIP/` and model weights such as `sam_vit_h_4b8939.pth` reside at repo root. Generated data goes to `crops/` and `corrected_labels/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` (Windows: `.\.venv\Scripts\activate`) — create and activate an isolated environment before installing deps.
- `pip install -r requirements.txt` — install runtime dependencies (Python 3.10+ recommended).
- `pip install -r requirements-dev.txt && pre-commit install` — optional dev tooling, lints, and format hooks.
- `python -m uvicorn app:app --host 0.0.0.0 --port 8000` — launch the API; interactive docs at `http://localhost:8000/docs`.
- `python tools/clip_kmeans_and_regress.py` — train the CLIP regression model after exporting YOLO crops into `./crops/`.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Prefer explicit type hints and compact, single-purpose functions. Keep API schemas in `localinferenceapi.py` or move to dedicated modules if they grow.
- Black, Ruff, and Isort settings live in `pyproject.toml`; use `pre-commit run --all-files` before submitting changes.

## Testing Guidelines
- No formal suite yet; use `pytest` under `tests/` with `test_*.py` naming when adding coverage.
- Manual verification: 1) export crops via the UI, 2) run training scripts, 3) start the API, 4) exercise endpoints via the UI or `/docs`.

## Commit & Pull Request Guidelines
- Use imperative, descriptive commit subjects (e.g., `fix loading logic`). Group related changes and avoid committing large binaries (`*.pth`, dataset zips).
- PRs should explain motivation, summarize changes, list run steps, and include screenshots/GIFs for UI updates. Link issues, call out risk areas, and update `readme.md` when behavior changes.

## Security & Configuration Tips
- Configure runtime through `.env` (see `.env.example`). Set `ENABLE_METRICS=true` to expose `/metrics` to Prometheus. Use `SAM_VARIANT` (`sam1` today, `sam3` once enabled) and point `SAM_CHECKPOINT_PATH` / `SAM3_*` envs at your weights or Hugging Face IDs (see `sam3integration.txt` for sample configs).
- When enabling SAM3, request access to the [facebook/sam3](https://huggingface.co/facebook/sam3) repo, clone Meta’s implementation (`git clone https://github.com/facebookresearch/sam3.git && pip install -e . && pip install einops`), and run `hf auth login` (after creating a read token) so the official processors/checkpoints are available before switching the UI selector to “SAM 3”.
- The API enables wide-open CORS and lacks auth—do not expose publicly. Keep model weights (`sam_vit_h_4b8939.pth`) and regression artifacts local.
- GPU acceleration is optional; the app auto-detects CUDA. Ensure required weight files exist before launching the server.
