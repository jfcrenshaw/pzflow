import pytest
from pzflow import Flow


@pytest.mark.parametrize(
    "input_dim,bijector,file",
    [
        (None, None, None),
        (2, None, "file"),
        (None, 2, "file"),
        (2, 2, "file"),
        (-1, None, None),
        (0, None, None),
        (1.1, None, None),
    ],
)
def test_bad_inputs(input_dim, bijector, file):
    with pytest.raises(ValueError):
        flow = Flow(input_dim, bijector, file)
