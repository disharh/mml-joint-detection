import numpy as np
import yaml

def process_value(val):
    # Convert numpy arrays
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return float(val[0])
        elif val.size == 2:
            return {
                1: float(val[0]),
                2: float(val[1])
            }
        else:
            return val.tolist()
    return val


def convert_dict(d):
    return {k: process_value(v) for k, v in d.items()}