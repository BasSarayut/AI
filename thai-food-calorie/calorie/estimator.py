from collections.abc import Iterable
from dataclasses import dataclass

from .nutrition import NutritionEntry


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float


@dataclass(frozen=True)
class FoodItem:
    label: str
    name_th: str
    kcal: float
    confidence: float


@dataclass(frozen=True)
class CalorieReport:
    items: tuple[FoodItem, ...]
    unknown_labels: tuple[str, ...]
    total_kcal: float
    low_kcal: float
    high_kcal: float


def detections_from_result(result) -> list[Detection]:
    names = result.names
    classes = result.boxes.cls.tolist()
    confidences = result.boxes.conf.tolist()
    return [Detection(names[int(c)], float(p)) for c, p in zip(classes, confidences)]


def estimate_calories(
    detections: Iterable[Detection],
    table: dict[str, NutritionEntry],
    min_confidence: float = 0.4,
    uncertainty: float = 0.25,
) -> CalorieReport:
    if not 0 <= uncertainty < 1:
        raise ValueError(f"uncertainty must be in [0, 1), got {uncertainty}")

    items: list[FoodItem] = []
    unknown: set[str] = set()
    for d in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if d.confidence < min_confidence:
            continue
        entry = table.get(d.label)
        if entry is None:
            unknown.add(d.label)
            continue
        items.append(FoodItem(d.label, entry.name_th, entry.kcal, d.confidence))

    total = sum(i.kcal for i in items)
    return CalorieReport(
        items=tuple(items),
        unknown_labels=tuple(sorted(unknown)),
        total_kcal=total,
        low_kcal=total * (1 - uncertainty),
        high_kcal=total * (1 + uncertainty),
    )
