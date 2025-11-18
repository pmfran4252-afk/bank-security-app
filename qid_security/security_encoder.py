from typing import Optional
import numpy as np
import pandas as pd

from qid_core.qid_complex_encoder import QIDComplexEncoder
from qid_core.qid_fft_backend import FFTBackend
from qid_core.qid_interference import InterferenceEngine
from qid_core.qid_signature import SignatureExtractor
from qid_core.qid_masks import BaseAmplitudeMask


class SecurityInterferenceEncoder:
    """
    Bank-Security-specific wrapper around QID Core.

    Pipeline:
    - Take an account's ordered events (transactions).
    - Build magnitude & phase per event.
    - Use QIDComplexEncoder to construct complex sequence z_t.
    - Apply event-domain amplitude mask weights.
    - Use InterferenceEngine (FFT + freq mask) to get spectrum.
    - Extract compact QID signature.

    The resulting signature can be used to:
    - Assign a scalar risk score (L2 norm, etc.).
    - Feed into downstream models as a feature vector.
    """

    def __init__(self, signature_dim: int = 64):
        self.signature_dim = signature_dim
        self.encoder = QIDComplexEncoder()
        self.fft_backend = FFTBackend()
        self.interference_engine = InterferenceEngine(self.fft_backend)
        self.signature_extractor = SignatureExtractor(signature_dim=signature_dim)

        # Channel-dependent phase offsets: separate channels along the complex plane
        self.channel_phase_offsets = {
            "CARD_POS": 0.0,
            "CARD_ECOM": 0.4 * np.pi,
            "ACH": 0.8 * np.pi,
            "ZELLE": 1.2 * np.pi,
            "WIRE_INTL": 1.6 * np.pi,
            "ATM": 2.0 * np.pi,
        }

    def _events_to_magnitude_phase(self, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Map events to magnitude and phase arrays.

        Magnitude encodes:
        - log(amount)
        - device risk
        - account age risk

        Phase encodes:
        - position in sequence
        - channel type offset
        """
        n = len(events)
        if n == 0:
            return (
                np.zeros(1, dtype=float),
                np.zeros(1, dtype=float),
            )

        events_sorted = events.sort_values("timestamp")
        t = np.arange(n)
        base_phase = 2.0 * np.pi * t / max(1, n - 1)

        ch = events_sorted["channel"].values
        ch_offsets = np.array(
            [self.channel_phase_offsets.get(c, 0.0) for c in ch],
            dtype=float,
        )

        amt = events_sorted["amount"].clip(1.0, None).values.astype(float)
        m_amount = np.log1p(amt)

        device_age = events_sorted["device_age_days"].values.astype(float)
        m_device = 1.0 + 0.5 * np.exp(-device_age / 30.0)

        acc_age = events_sorted["account_age_days"].values.astype(float)
        m_acc_age = 1.0 + 0.3 * np.exp(-acc_age / 365.0)

        magnitude = m_amount * m_device * m_acc_age
        phase = base_phase + ch_offsets

        return magnitude, phase

    def encode_sequence(
        self,
        events: pd.DataFrame,
        amp_mask: Optional[BaseAmplitudeMask] = None,
    ) -> np.ndarray:
        """
        Full QID encoding path for a single account's events.
        """
        if len(events) == 0:
            return np.zeros(self.signature_dim + 4, dtype=np.float32)

        mag, phase = self._events_to_magnitude_phase(events)
        z = self.encoder.encode(mag, phase)

        # Apply event-domain amplitude mask
        if amp_mask is not None:
            weights = np.array(
                [amp_mask.event_weight(row) for _, row in events.sort_values("timestamp").iterrows()],
                dtype=float,
            )
            z = z * weights

        # Frequency-domain interference
        S, _ = self.interference_engine.apply_interference(z, amp_mask)
        sig = self.signature_extractor.extract(S)
        return sig
