import numpy as np
from typing import Dict, Any


class BaseAmplitudeMask:
    """
    Base class for amplitude masks used by QID.

    Two domains:
    - Event-domain:
        event_weight(event_row) -> scalar multiplier for that event's amplitude.
    - Frequency-domain:
        freq_mask(freqs) -> array of multipliers per frequency bin.
    """

    def __init__(self, typology: str = "BASE", params: Dict[str, Any] | None = None):
        self.typology = typology
        self.params = params or {}

    def event_weight(self, event) -> float:
        """
        Default event weight: neutral (1.0).
        Domain-specific subclasses override this to emphasize or de-emphasize events.
        """
        return 1.0

    def freq_mask(self, freqs: np.ndarray) -> np.ndarray:
        """
        Default frequency mask: neutral (all ones).
        Domain-specific subclasses override this to emphasize certain rhythms.
        """
        return np.ones_like(freqs, dtype=float)
