import numpy as np
from typing import Dict, Any


class QIDPruner:
    """
    Simple top-k pruner.

    Accepts a mapping from entity_id -> score and returns the top-k sorted list.
    """

    def score_signature(self, sig: np.ndarray) -> float:
        """
        Default score = L2 norm of signature.
        """
        return float(np.linalg.norm(sig))

    def prune(self, scores: Dict[Any, float], k: int) -> list[tuple[Any, float]]:
        if k <= 0:
            return []
        # sort by descending score
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return items[:k]
