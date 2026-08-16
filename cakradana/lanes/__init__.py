"""Detection lanes.

Each lane produces a bounded share of the behavioural score and its own
reasons. They are kept separate because they rest on different kinds of
evidence and are not calibrated against one another, so pooling them would let
the weakest displace the strongest.
"""

from cakradana.lanes.anomaly import AnomalyLane, AnomalyModel, fit as fit_anomaly
from cakradana.lanes.classifier import ClassifierLane
from cakradana.lanes.graph import GraphLane
from cakradana.lanes.reputation import (
    CoverageIndex,
    CoverageItem,
    OperatingConditions,
    ReputationLane,
)

__all__ = [
    "AnomalyLane",
    "AnomalyModel",
    "ClassifierLane",
    "CoverageIndex",
    "CoverageItem",
    "GraphLane",
    "OperatingConditions",
    "ReputationLane",
    "fit_anomaly",
]
