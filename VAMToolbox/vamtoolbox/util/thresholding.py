import numpy as np
from numpy.typing import NDArray


def threshold(x: np.ndarray, thresh: float) -> NDArray[bool]:
    return (x >= thresh).astype(bool)
