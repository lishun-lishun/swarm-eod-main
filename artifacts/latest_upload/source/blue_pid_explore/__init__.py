"""Independent blue-team control stack.

The package is intentionally introduced one verified layer at a time.  The
current layer contains only the identified translational dynamics utilities;
navigation and exploration code must not depend on simulator ground truth.
"""

from .dynamics import DiscreteDynamicsModel, fit_discrete_dynamics
from .velocity_controller import (
    InverseDynamicsVelocityController,
    SteadyStateFeedbackVelocityController,
)

__all__ = [
    "DiscreteDynamicsModel",
    "fit_discrete_dynamics",
    "InverseDynamicsVelocityController",
    "SteadyStateFeedbackVelocityController",
]
