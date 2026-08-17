"""
Deep Nonlinear Causality Framework
Author: Dr. Abdullah Mohammad Ghazi Al khatib
Affiliation: Al Sham Private University
Email: genius.275@hotmail.com
GitHub: https://github.com/ABDULLAH-AL-KHATIB
Date: 17-08-2026
"""
from .causality import CausalityTest, ConditionalCausalityTest
from .utils import prepare_data, prepare_sequential_data, block_permutation

__version__ = "1.0.0"
__author__ = "Dr. Abdullah Mohammad Ghazi Al khatib"
__email__ = "genius.275@hotmail.com"
__affiliation__ = "Al Sham Private University"
__github__ = "https://github.com/ABDULLAH-AL-KHATIB"
__date__ = "17-08-2026"

__all__ = [
    "CausalityTest",
    "ConditionalCausalityTest",
    "prepare_data",
    "prepare_sequential_data",
    "block_permutation",
]
