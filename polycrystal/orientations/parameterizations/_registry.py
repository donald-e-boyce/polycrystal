"""Registry for parameterizations"""
import abc


registry = dict()


class ParameterizationRegistry(abc.ABCMeta):
    """Keep a dictionary of parameterizations by name"""

    def __init__(cls, name, bases, attrs):
        type.__init__(cls, name, bases, attrs)

        if hasattr(cls, 'name'):
            registry[cls.name] = cls
