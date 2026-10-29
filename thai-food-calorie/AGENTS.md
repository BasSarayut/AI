# AGENTS.md

## Project
Thai food calorie estimator: YOLO (Ultralytics) detection trained on Google Colab, plus a Gradio demo for Hugging Face Spaces. Currently Level 1 (detection + standard-serving lookup).

## Setup & verification
- Env: `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`
- Tests: `.venv/bin/pytest -q` (core `calorie/` package is stdlib-only)
- Run demo: `MODEL_PATH=weights/best.pt .venv/bin/python app.py` (a COCO `yolo26n.pt` works for smoke tests)
- Notebook (`notebooks/train_yolo_thai_food.ipynb`) only runs on Colab. To check it locally, compile the code cells with IPython's `TransformerManager`.

## Conventions
- Keep `calorie/` free of ML dependencies so it stays unit-testable. Put ultralytics/gradio code in `app.py` and `scripts/`.
- Pinned versions (all at least 7 days old when pinned): ultralytics 8.4.181, gradio 6.30.0, huggingface_hub 2.2.0, roboflow 1.7.1. The notebook pins the same versions.
- Gradio 6: app-level options (theme/css) go in `launch()`, not `gr.Blocks()`.
- `data/nutrition.csv` `class_name` must exactly match the model's class names. Unverified values have `verified=false`.
- Never hardcode API keys. Use Colab Secrets (`ROBOFLOW_API_KEY`, `HF_TOKEN`) and Space Variables (`HF_MODEL_REPO`).
