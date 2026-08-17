"""
Deep Nonlinear Causality Framework
Author: Dr. Abdullah Mohammad Ghazi Al khatib
Department: Department of Banking and Financial Institutions
Faculty: Faculty of Administrative Sciences
Affiliation: Al-Sham Private University, Damascus, Syrian Arab Republic
Academic Email: abduall.kh.foas@aspu.edu.sy
Personal Email: genius.275@hotmail.com
ORCID: https://orcid.org/0000-0002-1352-2348
GitHub: https://github.com/ABDULLAH-AL-KHATIB
Date: 17-08-2026
"""
from .causality import CausalityTest, ConditionalCausalityTest
from .utils import prepare_data, prepare_sequential_data, block_permutation

__version__ = "1.0.2"
__author__ = "Dr. Abdullah Mohammad Ghazi Al khatib"
__department__ = "Department of Banking and Financial Institutions"
__faculty__ = "Faculty of Administrative Sciences"
__affiliation__ = "Al-Sham Private University, Damascus, Syrian Arab Republic"
__academic_email__ = "abduall.kh.foas@aspu.edu.sy"
__email__ = "genius.275@hotmail.com"
__orcid__ = "https://orcid.org/0000-0002-1352-2348"
__github__ = "https://github.com/ABDULLAH-AL-KHATIB"
__date__ = "17-08-2026"

__all__ = [
    "CausalityTest",
    "ConditionalCausalityTest",
    "prepare_data",
    "prepare_sequential_data",
    "block_permutation",
]
