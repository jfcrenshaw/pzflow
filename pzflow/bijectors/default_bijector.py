from typing import Callable
from pzflow.bijectors.bijectors import Chain, Roll
from pzflow.bijectors.neural_splines import NeuralSplineCoupling
from typing import Callable


def DefaultBijector(input_dim: int) -> Callable:
    return Chain(*(NeuralSplineCoupling(), Roll()) * input_dim)
