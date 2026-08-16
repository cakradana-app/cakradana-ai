"""Feature computation, defined once and shared by training and serving."""

from cakradana.features.definitions import (
    FeatureSpec,
    FeatureValue,
    catalogue,
    categorical_names,
    feature_names,
    numeric_names,
)
from cakradana.features.service import FeatureService, FeatureVector, feature_set_version

# Imported for its side effect of registering the network features. Without
# this the catalogue would depend on whether some other module happened to be
# imported first, and features would go missing silently rather than loudly.
from cakradana.features import graph as _graph  # noqa: F401

__all__ = [
    "FeatureService",
    "FeatureSpec",
    "FeatureValue",
    "FeatureVector",
    "catalogue",
    "categorical_names",
    "feature_names",
    "feature_set_version",
    "numeric_names",
]
