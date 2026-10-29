from types import SimpleNamespace

import pytest

from calorie.estimator import Detection, detections_from_result, estimate_calories
from calorie.nutrition import NutritionEntry

TABLE = {
    "Rice": NutritionEntry("Rice", "ข้าวสวย", 240, "test", False),
    "Fried Egg": NutritionEntry("Fried Egg", "ไข่ดาว", 150, "test", False),
}


def test_sums_calories_of_each_detected_item():
    report = estimate_calories([Detection("Rice", 0.9), Detection("Fried Egg", 0.8)], TABLE)

    assert report.total_kcal == 390
    assert [i.name_th for i in report.items] == ["ข้าวสวย", "ไข่ดาว"]


def test_counts_repeated_items_separately():
    report = estimate_calories([Detection("Fried Egg", 0.9), Detection("Fried Egg", 0.85)], TABLE)

    assert report.total_kcal == 300


def test_ignores_low_confidence_detections():
    report = estimate_calories([Detection("Rice", 0.9), Detection("Fried Egg", 0.2)], TABLE, min_confidence=0.4)

    assert report.total_kcal == 240


def test_reports_unknown_labels_instead_of_failing():
    report = estimate_calories([Detection("Pizza", 0.9), Detection("Pizza", 0.8), Detection("Rice", 0.9)], TABLE)

    assert report.unknown_labels == ("Pizza",)
    assert report.total_kcal == 240


def test_range_reflects_uncertainty():
    report = estimate_calories([Detection("Rice", 0.9)], TABLE, uncertainty=0.25)

    assert report.low_kcal == pytest.approx(180)
    assert report.high_kcal == pytest.approx(300)


def test_empty_detections_give_zero():
    report = estimate_calories([], TABLE)

    assert report.items == ()
    assert report.total_kcal == report.low_kcal == report.high_kcal == 0


@pytest.mark.parametrize("uncertainty", [-0.1, 1.0])
def test_rejects_invalid_uncertainty(uncertainty):
    with pytest.raises(ValueError):
        estimate_calories([], TABLE, uncertainty=uncertainty)


def test_detections_from_ultralytics_like_result():
    tensor = lambda values: SimpleNamespace(tolist=lambda: values)
    result = SimpleNamespace(
        names={0: "Rice", 1: "Fried Egg"},
        boxes=SimpleNamespace(cls=tensor([1.0, 0.0]), conf=tensor([0.7, 0.95])),
    )

    assert detections_from_result(result) == [Detection("Fried Egg", 0.7), Detection("Rice", 0.95)]
