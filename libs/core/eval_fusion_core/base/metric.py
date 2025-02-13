from __future__ import annotations

from abc import ABC, ABCMeta
from enum import Enum, EnumMeta


class Meta(ABCMeta, EnumMeta):
    pass


class EvalFusionBaseMetric(ABC, Enum, metaclass=Meta):
    pass
