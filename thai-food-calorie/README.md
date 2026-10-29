---
title: Thai Food Calorie Estimator
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: agpl-3.0
---

# Thai Food Calorie Estimator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BasSarayut/AI/blob/main/thai-food-calorie/notebooks/train_yolo_thai_food.ipynb)

Detect Thai dishes in a photo with a custom-trained YOLO model and estimate total calories, shown as a range rather than a single misleading number.

**Live demo:** `<link to your Hugging Face Space>`

<!-- Add a demo GIF here: upload photo -> boxes -> calorie table -->

## Problem

Calorie-tracking apps handle Thai food poorly. Dishes are rarely in their databases, and logging a plate of ข้าวมันไก่ by hand is tedious. This project asks: **how close can a phone photo get to the real calorie count?**

## How it works

```
photo ──► YOLO26 detector ──► dish labels + confidence
                                   │
              data/nutrition.csv ──┤  (kcal per standard serving)
                                   ▼
                         total kcal  ± uncertainty range
```

| Component | Choice | Why |
|---|---|---|
| Detector | Ultralytics YOLO26n, fine-tuned (transfer learning from COCO) | Small and fast, trains on a free Colab T4 |
| Data | [Thai_food (Roboflow Universe)](https://universe.roboflow.com/thaifood/thai_food-y98fg) + my own phone photos | Public data for training; own photos as a real-world test set |
| Nutrition | [Thai Food Composition Database, INMU Mahidol](https://inmu.mahidol.ac.th/thaifcd/) | Authoritative Thai source |
| Demo | Gradio on Hugging Face Spaces | Free, permanent link |

**Level 1 limitation:** detection identifies *what* is on the plate but not *how much*. Calories assume a standard serving, and the ±25% range reflects real variation in oil and portion size.

## Results

<!-- Fill these in from the notebook. Honest numbers beat impressive ones. -->

### Detection

| Model | Test set | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLO26n v1 | Roboflow test split | | | | |
| YOLO26n v1 | My own photos | | | | |

<!-- Add confusion_matrix_normalized.png and write 2-3 sentences on which dishes get confused and why -->

### Calorie estimation vs ground truth

Ground truth comes from convenience-store meals with kcal printed on the package, plus home-cooked meals weighed on a kitchen scale.

| Images | MAE (kcal) | MAPE | Range coverage |
|---|---|---|---|
| | | | |

### Error analysis

<!-- Where does it fail? Lighting, mixed plates, dishes not in the dataset, portion size... -->

## Run it

**Train (Google Colab):** open the notebook via the badge above and follow the steps. Everything runs on the free T4 GPU.

**Run the demo locally:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
mkdir -p weights && cp /path/to/best.pt weights/best.pt   # or: export HF_MODEL_REPO=<user>/thai-food-yolo
python app.py
```

**Evaluate calories against your own ground truth:**

```bash
python scripts/evaluate_calories.py --model weights/best.pt \
    --images eval/images --truth eval/ground_truth.csv
```

**Tests:** `pytest`

## Deploy to Hugging Face Spaces

1. Create a new Space with SDK **Gradio**.
2. In the Space settings, add a Variable `HF_MODEL_REPO` = `<user>/thai-food-yolo` (the model repo uploaded from the notebook).
3. This project lives in the `thai-food-calorie/` subfolder of the [BasSarayut/AI](https://github.com/BasSarayut/AI) repo, but a Space needs `app.py` at its root. Push only the subfolder, from the repo root:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/thai-food-calorie
   git subtree push --prefix thai-food-calorie space main
   ```

## Project structure

```
calorie/          core logic (pure Python, unit-tested)
  nutrition.py    load and validate the nutrition table
  estimator.py    detections -> calorie report with uncertainty range
  metrics.py      MAE, MAPE, range coverage
data/nutrition.csv
notebooks/        Colab training notebook
scripts/          calorie evaluation against ground truth
app.py            Gradio demo
```

## Roadmap

- [ ] Verify every row in `nutrition.csv` against INMU (`verified=true`)
- [ ] Level 2: instance segmentation (`yolo26n-seg`) + plate as size reference -> S/M/L portion
- [ ] Compare against a vision LLM on the same test set (accuracy / latency / cost)
- [ ] Level 3: depth estimation for volume-based portions

## Credits and licenses

- Dataset: Thai_food by Thaifood on Roboflow Universe
- Nutrition data: Institute of Nutrition, Mahidol University. Thai Food Composition Database, Online version 3 (2025)
- Ultralytics YOLO is AGPL-3.0, so this project is released under AGPL-3.0

*Calorie values are estimates and not medical or dietary advice.*
