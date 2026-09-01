import numpy as np

from src.investigation import (
    contains_forbidden_ground_truth,
)
from src.scoring import (
    percentile_from_reference,
)


def test_percentile_from_reference():
    reference = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    scores = np.array(
        [1.0, 2.5, 4.0]
    )

    result = percentile_from_reference(
        reference,
        scores,
    )

    expected = np.array(
        [0.25, 0.50, 1.00]
    )

    assert np.allclose(
        result,
        expected,
    )


def test_ground_truth_guard():
    safe = {
        "transaction": {
            "amount": 100.0,
        }
    }

    unsafe = {
        "transaction": {
            "amount": 100.0,
        },
        "fraud": 1,
    }

    assert not contains_forbidden_ground_truth(
        safe
    )

    assert contains_forbidden_ground_truth(
        unsafe
    )
