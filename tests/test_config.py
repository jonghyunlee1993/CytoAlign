from src.config import load_config


def test_yaml_inheritance_merges_nested_values(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("data:\n  root: data\n  cells: 100\ntraining:\n  epochs: 30\n")
    child.write_text("extends: base.yaml\ndata:\n  cells: 10\n")

    config = load_config(child)

    assert config["data"] == {"root": "data", "cells": 10}
    assert config["training"]["epochs"] == 30
