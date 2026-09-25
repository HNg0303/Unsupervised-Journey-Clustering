"""Reusable journey-clustering domain package.

The package contains preprocessing, feature construction, model fitting,
inference, and artifact helpers.  Repository-level commands live in
``scripts`` and should remain thin adapters around this package.
"""

from .pipelines import fit_global_journey_model, prepare_event_partition

__version__ = "0.1.0"

__all__ = ["prepare_event_partition", "fit_global_journey_model"]
