import numpy as np
import pytest

from .synthetic import make_index


@pytest.fixture
def tiny_index():
    # Four chunks along the first two axes; a query at (1, 0) scores them
    # 1.0, 0.8, 0.6, 0.0 in that order.
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return make_index(
        ["rfc9110#1", "rfc9110#2", "rfc9111#3", "rfc9111#4"],
        vectors,
        question_ids=["qa", "qb"],
        question_vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
