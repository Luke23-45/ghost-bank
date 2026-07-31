from src.methods.base import Method
from src.methods.baseline import BaselineMethod
from src.methods.uniform_herding import UniformHerdingMethod
from src.methods.static_bank import StaticBankMethod
from src.methods.icarl.method import iCaRLMethod

__all__ = [
    "Method",
    "BaselineMethod",
    "StaticBankMethod",
    "UniformHerdingMethod",
    "iCaRLMethod",
]
