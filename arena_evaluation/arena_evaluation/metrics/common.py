from __future__ import annotations

import math
import typing


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def l2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def diff(values: list[float]) -> list[float]:
    return [values[i + 1] - values[i] for i in range(len(values) - 1)]


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0.0 or math.isnan(den) or math.isinf(den):
        return default
    return num / den


def finite_or_none(x: float | None) -> float | None:
    if x is None:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def as_float_list(values: typing.Iterable[float]) -> list[float]:
    return [float(v) for v in values]