import pytest

from calorie.metrics import mean_absolute_error, mean_absolute_percentage_error, range_coverage


def test_mae():
    assert mean_absolute_error([500, 300], [450, 360]) == 55


def test_mape():
    assert mean_absolute_percentage_error([500, 200], [400, 250]) == pytest.approx(0.225)


def test_mape_rejects_non_positive_truth():
    with pytest.raises(ValueError):
        mean_absolute_percentage_error([0, 100], [10, 100])


def test_range_coverage_counts_truth_inside_bounds():
    assert range_coverage([500, 300, 700], [400, 310, 600], [600, 400, 700]) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "args",
    [([], []), ([1, 2], [1])],
)
def test_rejects_empty_or_mismatched_inputs(args):
    with pytest.raises(ValueError):
        mean_absolute_error(*args)
