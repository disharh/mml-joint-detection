import numpy as np
import yaml

def convert_value(val):
    if isinstance(val, dict):
        return {k: convert_value(v) for k, v in val.items()}

    if isinstance(val, (list, tuple)):
        return [convert_value(v) for v in val]

    if isinstance(val, np.ndarray):
        if val.size == 1:
            return float(val.item())
        if val.size == 2:
            return {1: float(val[0]), 2: float(val[1])}
        return val.tolist()

    if isinstance(val, (np.floating, np.integer, np.bool_)):
        return val.item()

    return val


def convert_dict(d):
    return {k: convert_value(v) for k, v in d.items()}

