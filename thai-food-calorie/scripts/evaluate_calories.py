"""Compare estimated calories against ground truth (e.g. kcal printed on convenience-store meal packages).

Usage:
    python scripts/evaluate_calories.py --model weights/best.pt \
        --images eval/images --truth eval/ground_truth.csv --out eval/results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calorie.estimator import detections_from_result, estimate_calories  # noqa: E402
from calorie.metrics import mean_absolute_error, mean_absolute_percentage_error, range_coverage  # noqa: E402
from calorie.nutrition import load_nutrition  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--nutrition", default=ROOT / "data" / "nutrition.csv", type=Path)
    parser.add_argument("--out", default=ROOT / "eval" / "results.csv", type=Path)
    parser.add_argument("--conf", default=0.4, type=float)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    table = load_nutrition(args.nutrition)
    with open(args.truth, newline="", encoding="utf-8") as f:
        truth_rows = list(csv.DictReader(f))
    if not truth_rows:
        sys.exit(f"no rows in {args.truth}")

    rows = []
    for row in truth_rows:
        result = model.predict(str(args.images / row["image"]), conf=args.conf, verbose=False)[0]
        report = estimate_calories(detections_from_result(result), table, min_confidence=args.conf)
        rows.append({
            "image": row["image"],
            "true_kcal": float(row["true_kcal"]),
            "pred_kcal": report.total_kcal,
            "low_kcal": report.low_kcal,
            "high_kcal": report.high_kcal,
            "detected": "; ".join(i.label for i in report.items),
            "unknown": "; ".join(report.unknown_labels),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    true = [r["true_kcal"] for r in rows]
    pred = [r["pred_kcal"] for r in rows]
    for r in rows:
        print(f"{r['image']:<40} true={r['true_kcal']:>6.0f}  pred={r['pred_kcal']:>6.0f}  [{r['detected']}]")
    print(f"\nimages: {len(rows)}")
    print(f"MAE:  {mean_absolute_error(true, pred):.1f} kcal")
    print(f"MAPE: {mean_absolute_percentage_error(true, pred):.1%}")
    print(f"Range coverage: {range_coverage(true, [r['low_kcal'] for r in rows], [r['high_kcal'] for r in rows]):.1%}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
