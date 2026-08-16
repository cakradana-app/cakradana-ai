"""Detection lanes.

Each lane produces a bounded share of the behavioural score and its own
reasons. They are kept separate because they rest on different kinds of
evidence and are not calibrated against one another.
"""

from cakradana.lanes.graph import GraphLane

__all__ = ["GraphLane"]
