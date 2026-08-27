"""Scripts and workflow tools for polycrystal simulations."""
from ..slip.solve import StateData, StressSequence


def run_sequence(*args, **kwargs):
    """Integrate a multi-step stress sequence and save step-NN.npz output files."""
    from .run_sequence import run_sequence as _run_sequence
    return _run_sequence(*args, **kwargs)


__all__ = [
    "StateData",
    "StressSequence",
    "run_sequence",
]
