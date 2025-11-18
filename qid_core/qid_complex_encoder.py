# qid_core/qid_complex_encoder.py

import numpy as np


class QIDComplexEncoder:
    """
    Generic complex encoder for QID.

    Given a magnitude array and a phase array, returns a complex-valued sequence:

        z_t = magnitude_t * exp(i * phase_t)

    Domain-specific layers (e.g., qid_security.SecurityInterferenceEncoder) decide
    how to construct magnitudes and phases from raw events.
    """

    def encode(self, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        magnitude = np.asarray(magnitude, dtype=np.float64)
        phase = np.asarray(phase, dtype=np.float64)

        if magnitude.shape != phase.shape:
            raise ValueError(
                f"QIDComplexEncoder: magnitude.shape {magnitude.shape} "
                f"!= phase.shape {phase.shape}"
            )

        return magnitude * np.exp(1j * phase)
