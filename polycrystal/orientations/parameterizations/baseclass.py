"""Base class for parameterizations

This just uses the abstract base class and sets up the registry to use for
automatics association of a class with its name.
"""
from .parameterizationabc import ParameterizationABC
from ._registry import ParameterizationRegistry


class Parameterization(ParameterizationABC, metaclass=ParameterizationRegistry):
    pass
