"""Forecast models. ``seq2seq`` needs PyTorch and is imported lazily."""
from .baselines import Climatology, Persistence

__all__ = ["Persistence", "Climatology"]
