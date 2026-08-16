"""The scoring service and its HTTP contract."""

from cakradana.serving.service import ScoringService, ServiceNotReady, to_donation

__all__ = ["ScoringService", "ServiceNotReady", "to_donation"]
