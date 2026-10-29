import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NutritionEntry:
    class_name: str
    name_th: str
    kcal: float
    source: str
    verified: bool


def load_nutrition(path: str | Path) -> dict[str, NutritionEntry]:
    table: dict[str, NutritionEntry] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["class_name"].strip()
            if name in table:
                raise ValueError(f"duplicate class_name in {path}: {name!r}")
            kcal = float(row["kcal"])
            if kcal < 0:
                raise ValueError(f"kcal must be >= 0 for {name!r}, got {kcal}")
            table[name] = NutritionEntry(
                class_name=name,
                name_th=row["name_th"].strip(),
                kcal=kcal,
                source=row["source"].strip(),
                verified=row["verified"].strip().lower() == "true",
            )
    return table


def missing_classes(class_names: Iterable[str], table: dict[str, NutritionEntry]) -> list[str]:
    return sorted(set(class_names) - table.keys())
