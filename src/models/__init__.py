"""Goals models behind a single interface.

Step 3 ships the baselines; step 4 adds Poisson regression and Dixon-Coles.
"""

from .base import GoalModel
from .baseline import FormBaseline, HomeAwayBaseline, LeagueAverageBaseline
from .dixon_coles import DixonColes
from .gbm_poisson import GBMPoisson
from .poisson_glm import PoissonGLM

__all__ = [
    "GoalModel",
    "LeagueAverageBaseline",
    "HomeAwayBaseline",
    "FormBaseline",
    "PoissonGLM",
    "DixonColes",
    "GBMPoisson",
]
