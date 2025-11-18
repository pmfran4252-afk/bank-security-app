from typing import Optional
import numpy as np
import pandas as pd

from qid_core.qid_pruner import QIDPruner
from qid_core.qid_masks import BaseAmplitudeMask
from .security_encoder import SecurityInterferenceEncoder


class SecurityPruner:
    """
    Security-specific pruner that:
    - Groups transactions by account.
    - Uses SecurityInterferenceEncoder to compute QID signatures.
    - Uses QIDPruner to rank accounts by score.
    """

    def __init__(self, encoder: SecurityInterferenceEncoder):
        self.encoder = encoder
        self.pruner = QIDPruner()

    def score_account(self, events: pd.DataFrame, amp_mask: Optional[BaseAmplitudeMask]) -> float:
        sig = self.encoder.encode_sequence(events, amp_mask)
        return self.pruner.score_signature(sig)

    def score_all_accounts(
        self,
        df: pd.DataFrame,
        amp_mask: BaseAmplitudeMask,
        max_accounts: Optional[int] = None,
    ) -> pd.DataFrame:
        scores = []
        grouped = df.groupby("account_id", sort=False)

        for i, (acc_id, g) in enumerate(grouped):
            if (max_accounts is not None) and (i >= max_accounts):
                break
            score = self.score_account(g, amp_mask)
            label_fraud = int((g["label_fraud"] == 1).any())
            typology_counts = g["label_typology"].value_counts()
            dominant_typology = typology_counts.idxmax() if not typology_counts.empty else ""
            scores.append(
                {
                    "account_id": acc_id,
                    "qid_score": score,
                    "label_fraud": label_fraud,
                    "label_typology": dominant_typology,
                    "n_txn": len(g),
                }
            )

        score_df = pd.DataFrame(scores).sort_values("qid_score", ascending=False).reset_index(drop=True)
        return score_df
