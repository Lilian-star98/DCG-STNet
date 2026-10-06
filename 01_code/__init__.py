"""Paper-aligned package for DCG-STNet / STCausalNet (20260111 中文版).

Subpackages map to paper sections:
- model:      §3 method
- data:       §4.1–4.2 datasets and preprocessing
- scenarios:  Appendix B + experiments E3/E4/E5
- figures:    Fig.1–Fig.8
- tables:     Table 1–4
- experiments: training / evaluation entry points
"""

from .config import ModelConfig

__all__ = ["ModelConfig"]
