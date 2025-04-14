"""Registry for parameterizations"""
import abc


registry = dict()


class ParameterizationRegistry(abc.ABCMeta):
    """Keep a dictionary of parameterizations by name"""

    def __init__(cls, name, bases, attrs):
        type.__init__(cls, name, bases, attrs)

        if hasattr(cls, 'parameterization'):
            registry[cls.parameterization] = cls
        else:
            if name != "Parameterization":
                raise RuntimeError(
                    'Parameterization subclasses need a "parameterization" '
		    'attribute'
                )
