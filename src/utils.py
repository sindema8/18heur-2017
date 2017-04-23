import numpy as np


def is_integer(a):
    df = a.dtype
    return df == np.int16 or df == np.int32 or df == np.int64


if __name__ == "__main__":
    pass
