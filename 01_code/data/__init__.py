"""§4.1–4.2 datasets, chronological split, Z-score, scene features."""

from .features import build_time_features, build_q_scene_features
from .preprocess import make_windows, scale_zscore, time_split
from .q_real import load_real_q_subgraph, pick_bottleneck_node, pick_school_node

# Lazy wrappers so Q figure pipeline does not require h5py at import time
def load_dataset(*args, **kwargs):
    from .loaders import load_dataset as _f

    return _f(*args, **kwargs)


def load_adjacency(*args, **kwargs):
    from .loaders import load_adjacency as _f

    return _f(*args, **kwargs)


def load_q_coords(*args, **kwargs):
    from .loaders import load_q_coords as _f

    return _f(*args, **kwargs)


__all__ = [
    "load_dataset",
    "load_adjacency",
    "load_q_coords",
    "load_real_q_subgraph",
    "pick_bottleneck_node",
    "pick_school_node",
    "time_split",
    "scale_zscore",
    "make_windows",
    "build_time_features",
    "build_q_scene_features",
]
