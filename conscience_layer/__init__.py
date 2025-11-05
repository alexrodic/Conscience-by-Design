"""
Conscience Layer — Ethical Awareness Core (v0.1.0)
Original concept and authorship: Aleksandar Rodić (2025)
"""
from .core import ConscienceCore, Metrics, EthicalScores
from .feature_extractor import HeuristicFeatureExtractor
from .adapter import PolicyGate

__all__ = [
    "ConscienceCore","Metrics","EthicalScores","HeuristicFeatureExtractor","PolicyGate",
]
__version__='0.1.0'
