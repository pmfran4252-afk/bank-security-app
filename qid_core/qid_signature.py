import numpy as np


class SignatureExtractor:
    """
    Extracts a compact, fixed-length signature from a complex spectrum.

    This is deliberately generic:
    - Uses normalized power spectrum.
    - Compresses into a fixed-length vector.
    - Appends a few global statistics.

    Domain-specific code can add more heads / stats on top of this signature.
    """

    def __init__(self, signature_dim: int = 64):
        if signature_dim <= 0:
            raise ValueError("signature_dim must be positive")
        self.signature_dim = signature_dim

    def extract(self, S: np.ndarray) -> np.ndarray:
        """
        Convert complex spectrum S into a (signature_dim + 4)-length real vector.
        """
        power = np.abs(S) ** 2
        total_power = np.sum(power) + 1e-9
        power_norm = power / total_power

        # compress / pad
        if len(power_norm) >= self.signature_dim:
            sig = power_norm[: self.signature_dim]
        else:
            pad = np.zeros(self.signature_dim - len(power_norm), dtype=float)
            sig = np.concatenate([power_norm, pad])

        # global stats
        stats = np.array(
            [
                power_norm.mean(),
                power_norm.std(),
                power_norm.max(),
                np.argmax(power_norm) / max(1, len(power_norm) - 1),
            ],
            dtype=float,
        )
        sig = np.concatenate([sig, stats]).astype(np.float32)
        return sig
