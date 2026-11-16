"""Model registry used by the shared runner. Each member adds their model here when it is ready."""
from .bspline_kan import BSplineKAN

REGISTRY = {"bkan": BSplineKAN}
