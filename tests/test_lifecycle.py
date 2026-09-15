import pandas as pd
import pytest

from bankpulse.data import encode_target
from bankpulse.registry import serving_threshold, write_manifest


def test_numeric_and_text_labels_are_equivalent():
    assert encode_target(pd.Series(["yes", "no", 1, 0])).tolist() == [1, 0, 1, 0]
    with pytest.raises(ValueError):
        encode_target(pd.Series([None]))


def test_serving_rejects_tampered_artifact(tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"model")
    write_manifest(artifact, {"threshold": 0.37}, artifact.with_suffix(".json"))
    assert serving_threshold(artifact) == 0.37
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        serving_threshold(artifact)
