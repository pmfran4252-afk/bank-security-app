# qid_security/amplitude_masks_security.py

from typing import Dict, Any, Iterable, Optional

import numpy as np
import pandas as pd

from qid_core.qid_masks import BaseAmplitudeMask


class SecurityAmplitudeMask(BaseAmplitudeMask):
    """
    Bank-security-specific amplitude mask.

    - event_weight(row): boosts/suppresses events based on typology & risk params.
    - freq_mask(freqs): emphasizes rhythms (bursts vs structuring).
    - explain_event(row): returns weight + human-readable reasons.

    Typologies:
        - CARD_FRAUD
        - ATO
        - STRUCTURING
        - MULE
        - ALL_FRAUD  (catch-all, broad fraud radar)
    """

    def __init__(self, typology: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(typology=typology, params=params or {})

    # ------------------------------------------------------------------
    # Core weight + reasons
    # ------------------------------------------------------------------
    def _compute_weight_and_reasons(self, event: pd.Series):
        """
        Internal helper: compute weight and a list of textual reasons.
        """
        w = 1.0
        reasons = []

        ch = str(event.get("channel", ""))
        amt = float(event.get("amount", 0.0))
        dst_country = str(event.get("dst_country", ""))
        device_age = float(event.get("device_age_days", 0.0))
        acc_age = float(event.get("account_age_days", 0.0))
        risk_seg = str(event.get("risk_segment", "MEDIUM"))

        # Generic risk boosts (apply for all typologies)
        if risk_seg == "HIGH":
            w *= 1.2
            reasons.append("High-risk segment (×1.2)")

        if acc_age < 90:  # young account
            w *= 1.1
            reasons.append("New account (<90 days) (×1.1)")

        # Typology-specific event weighting
        if self.typology == "CARD_FRAUD":
            # Card-not-present / e-commerce, cross-border, moderate+ amounts
            if ch == "CARD_ECOM":
                w *= 2.0
                reasons.append("Card e-commerce channel (×2.0)")
            if dst_country not in ["US", "GB", "DE"]:
                w *= 1.7
                reasons.append(f"Cross-border to {dst_country} (×1.7)")
            if amt > 200:
                w *= 1.3
                reasons.append(f"High amount > 200 ({amt:.2f}) (×1.3)")

        elif self.typology == "ATO":
            # Account Takeover: new device + high-value WIRE/ZELLE
            if ch in ["WIRE_INTL", "ZELLE"]:
                w *= 2.0
                reasons.append(f"High-risk channel {ch} (×2.0)")
            if device_age < 7:
                w *= 1.7
                reasons.append("Very new device (<7 days) (×1.7)")
            if amt > 3000:
                w *= 1.5
                reasons.append(f"High-value ATO amount > 3000 ({amt:.2f}) (×1.5)")

        elif self.typology == "STRUCTURING":
            # Deposits just under reporting thresholds, mostly ACH/ATM
            if ch in ["ACH", "ATM"]:
                w *= 1.5
                reasons.append(f"Structuring channel {ch} (×1.5)")
            if 8000 <= amt < 10000:
                w *= 2.0
                reasons.append(f"Amount in structuring band [8k,10k): {amt:.2f} (×2.0)")

        elif self.typology == "MULE":
            # Payments mule: ZELLE/ACH flows, cross-border, mid-ticket
            if ch in ["ZELLE", "ACH"]:
                w *= 1.7
                reasons.append(f"Mule rail {ch} (×1.7)")
            if dst_country != "US":
                w *= 1.4
                reasons.append(f"Cross-border to {dst_country} (×1.4)")
            if 200 <= amt <= 2500:
                w *= 1.4
                reasons.append(f"Mid-ticket mule amount [200,2500]: {amt:.2f} (×1.4)")

        elif self.typology == "ALL_FRAUD":
            # Catch-all broad fraud radar:
            # blend card, mule, structuring, and ATO risk signals

            # Card & e-com risk
            if ch == "CARD_ECOM":
                w *= 1.7
                reasons.append("Card e-commerce channel (×1.7)")

            # Mule-ish rails
            if ch in ["ZELLE", "ACH"]:
                w *= 1.5
                reasons.append(f"ZELLE/ACH mule rail {ch} (×1.5)")

            # Wire-ish rails
            if ch == "WIRE_INTL":
                w *= 1.7
                reasons.append("International wire (×1.7)")

            # Corridor risk
            if dst_country not in ["US", "GB", "DE"]:
                w *= 1.5
                reasons.append(f"High-risk corridor to {dst_country} (×1.5)")

            # Amount-based risk
            if 200 <= amt <= 2500:
                w *= 1.2
                reasons.append(f"Suspicious mid-ticket amount [200,2500]: {amt:.2f} (×1.2)")
            if amt > 3000:
                w *= 1.4
                reasons.append(f"High-value payment > 3000 ({amt:.2f}) (×1.4)")
            if 8000 <= amt < 10000:
                w *= 1.2
                reasons.append(f"Near-threshold structuring band [8k,10k): {amt:.2f} (×1.2)")

            # New device elevates everything
            if device_age < 7:
                w *= 1.3
                reasons.append("Very new device (<7 days) (×1.3)")

        # Corridor-based generic boost from configuration
        high_risk_countries: Iterable[str] = self.params.get("high_risk_countries", [])
        if high_risk_countries and dst_country in high_risk_countries:
            w *= 1.5
            reasons.append(f"Configured high-risk country {dst_country} (×1.5)")

        return float(w), reasons

    def event_weight(self, event: pd.Series) -> float:
        """
        Standard interface used by the encoder: just return the scalar weight.
        """
        w, _ = self._compute_weight_and_reasons(event)
        return w

    def explain_event(self, event: pd.Series) -> dict:
        """
        Explain why this event got its weight under the current typology.

        Returns:
            {
                "weight": float,
                "reasons": [str, ...]
            }
        """
        w, reasons = self._compute_weight_and_reasons(event)
        if not reasons:
            reasons = ["No specific risk factors triggered; baseline weight only."]
        return {"weight": w, "reasons": reasons}

    # ------------------------------------------------------------------
    # Frequency-domain mask
    # ------------------------------------------------------------------
    def freq_mask(self, freqs: np.ndarray) -> np.ndarray:
        """
        Frequency-domain mask. `freqs` are FFT frequency bins.

        We emphasize different bands depending on typology:
        - CARD_FRAUD / ATO / MULE → mid/high frequencies (bursts)
        - STRUCTURING → low/mid frequencies (regular spacing)
        - ALL_FRAUD → both low/mid and mid/high (broad radar)
        """
        mask = np.ones_like(freqs, dtype=float)
        abs_f = np.abs(freqs)

        # bursts / bot-like / rapid flows
        if self.typology in ("CARD_FRAUD", "ATO", "MULE"):
            mask[(abs_f > 0.1) & (abs_f < 0.4)] *= 1.8

        # regular spaced deposits (structuring / smurfing)
        if self.typology == "STRUCTURING":
            mask[(abs_f > 0.02) & (abs_f < 0.15)] *= 1.8

        # catch-all radar: emphasize both bands, slightly less aggressively
        if self.typology == "ALL_FRAUD":
            mask[(abs_f > 0.02) & (abs_f < 0.15)] *= 1.5  # low/mid
            mask[(abs_f > 0.1) & (abs_f < 0.4)] *= 1.5   # mid/high

        return mask


def get_security_mask(
    typology: str, params: Optional[Dict[str, Any]] = None
) -> SecurityAmplitudeMask:
    """
    Convenience factory for security masks.
    """
    return SecurityAmplitudeMask(typology=typology, params=params or {})
