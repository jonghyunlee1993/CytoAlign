"""Cross-panel population evaluation."""

from .paired_curve import summarize_paired_curve
from .population_metrics import evaluate_matched_populations
from .summary import summarize_experiment

__all__ = [
    "evaluate_matched_populations",
    "summarize_experiment",
    "summarize_paired_curve",
]
