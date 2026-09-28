"""U-LASTDE: traffic accident detection on road networks."""

from .config import Config
from .model import OurAnomalyDetection

ULASTDE = OurAnomalyDetection

__all__ = ["Config", "ULASTDE"]
