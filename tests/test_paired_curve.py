import json

from src.evaluation.paired_curve import summarize_paired_curve
from src.training.experiment import _paired_subset


def test_paired_subsets_are_nested():
    specimens = [f"S{index}" for index in range(10)]
    one = set(_paired_subset(specimens, 1, 7))
    four = set(_paired_subset(specimens, 4, 7))
    assert one < four


def test_paired_curve_summary_fits_improvement_trend(tmp_path):
    root = tmp_path / "curve"
    for seed, offset in ((1, 0.0), (2, 0.01)):
        run = root / "fold_0" / f"seed_{seed}"
        run.mkdir(parents=True)
        shared = {
            name: {"test": {"patient_first_normalized_wasserstein": value + offset}}
            for name, value in {"ridge_hl": 1.0, "knn_hl": 0.8}.items()
        }
        curve = {}
        for count, value in ((0, 1.0), (1, 0.9), (2, 0.8), (4, 0.7)):
            methods = {
                "ot_hl": {
                    "test": {
                        "patient_first_normalized_wasserstein": (
                            value + offset if count == 0 else value + 0.05
                        )
                    }
                },
                "cytoalign": {
                    "selected_alpha": 1.0 if count else 0.0,
                    "test": {"patient_first_normalized_wasserstein": value + offset},
                },
            }
            curve[str(count)] = {
                "paired_specimens": [f"S{i}" for i in range(count)],
                "methods": methods,
            }
        (run / "metrics.json").write_text(
            json.dumps({"shared_methods": shared, "paired_curve": curve})
        )

    result = summarize_paired_curve(root)

    assert result["paired_sets_are_nested"]
    assert result["trend"]["raw_count_linear_fit"]["slope"] > 0
    assert result["first_count_showing_x_value"] == 1
