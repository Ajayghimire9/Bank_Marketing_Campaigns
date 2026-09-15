from bankpulse.drift import drift_status, psi


def test_stable_distribution():
    assert drift_status(0.05) == "stable"
    assert psi([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0
