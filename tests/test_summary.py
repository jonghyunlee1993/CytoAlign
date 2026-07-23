import json

from src.evaluation.summary import summarize_experiment


def test_summary_reports_baseline_comparisons(tmp_path):
    methods = {
        name: {
            "test": {
                "patient_first_normalized_wasserstein": value,
                "patient_first_normalized_median_error": value / 2,
            }
        }
        for name, value in {
            "ridge_hl": 1.0,
            "knn_hl": 0.9,
            "mlp_hl": 1.1,
            "ot_hl": 0.8,
            "cytoalign": 0.7,
        }.items()
    }
    run = tmp_path / "exp" / "fold_0" / "seed_1"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"methods": methods}))

    result = summarize_experiment(tmp_path / "exp")

    assert result["cytoalign_beats_all_baseline_means"]
    assert result["source_x_adds_value_over_ot_hl"]
    assert result["cytoalign_wins"]["knn_hl"] == 1
