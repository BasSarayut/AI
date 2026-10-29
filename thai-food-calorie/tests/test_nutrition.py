from pathlib import Path

import pytest

from calorie.nutrition import load_nutrition, missing_classes

HEADER = "class_name,name_th,kcal,source,verified\n"
PROJECT_CSV = Path(__file__).resolve().parents[1] / "data" / "nutrition.csv"


def write_csv(tmp_path, body: str) -> Path:
    path = tmp_path / "nutrition.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_loads_entries_keyed_by_class_name(tmp_path):
    table = load_nutrition(write_csv(tmp_path, "Fried Egg,ไข่ดาว,150,INMU,true\n"))

    entry = table["Fried Egg"]
    assert entry.name_th == "ไข่ดาว"
    assert entry.kcal == 150
    assert entry.verified is True


def test_unverified_flag(tmp_path):
    table = load_nutrition(write_csv(tmp_path, "Rice,ข้าวสวย,240,estimate,false\n"))

    assert table["Rice"].verified is False


def test_rejects_duplicate_class_names(tmp_path):
    path = write_csv(tmp_path, "Rice,ข้าวสวย,240,a,false\nRice,ข้าว,250,b,false\n")

    with pytest.raises(ValueError, match="duplicate"):
        load_nutrition(path)


def test_rejects_negative_kcal(tmp_path):
    with pytest.raises(ValueError, match="kcal"):
        load_nutrition(write_csv(tmp_path, "Rice,ข้าวสวย,-1,a,false\n"))


def test_missing_classes_reports_labels_without_nutrition(tmp_path):
    table = load_nutrition(write_csv(tmp_path, "Rice,ข้าวสวย,240,a,false\n"))

    assert missing_classes(["Rice", "Papaya Salad", "Fried Egg"], table) == ["Fried Egg", "Papaya Salad"]


def test_project_nutrition_csv_is_valid():
    table = load_nutrition(PROJECT_CSV)

    assert table
    assert all(e.name_th and e.source for e in table.values())
