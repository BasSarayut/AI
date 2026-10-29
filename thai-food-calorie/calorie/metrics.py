from collections.abc import Sequence


def _check(*series: Sequence[float]) -> None:
    if not series[0]:
        raise ValueError("inputs must not be empty")
    if len({len(s) for s in series}) != 1:
        raise ValueError("inputs must have the same length")


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _check(y_true, y_pred)
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def mean_absolute_percentage_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _check(y_true, y_pred)
    if any(t <= 0 for t in y_true):
        raise ValueError("y_true values must be > 0 for MAPE")
    return sum(abs(t - p) / t for t, p in zip(y_true, y_pred)) / len(y_true)


def range_coverage(y_true: Sequence[float], lows: Sequence[float], highs: Sequence[float]) -> float:
    _check(y_true, lows, highs)
    return sum(lo <= t <= hi for t, lo, hi in zip(y_true, lows, highs)) / len(y_true)
