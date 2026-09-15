import pytest
from pydantic import ValidationError

from bankpulse.schema import PredictionRequest


def test_age_contract():
    with pytest.raises(ValidationError):
        PredictionRequest(age=10)
