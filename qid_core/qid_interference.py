import numpy as np
from .qid_fft_backend import FFTBackend
from .qid_masks import BaseAmplitudeMask


class InterferenceEngine:
    """
    Core interference engine.

    - Takes a complex sequence z_t.
    - Applies FFT using the configured backend.
    - Optionally applies a frequency-domain amplitude mask.
    - Returns the complex spectrum.

    Event-domain masks are applied before this stage by the caller.
    """

    def __init__(self, fft_backend: FFTBackend | None = None):
        self.fft_backend = fft_backend or FFTBackend()

    def apply_interference(
        self,
        z: np.ndarray,
        amp_mask: BaseAmplitudeMask | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        z: complex-valued time-domain sequence.
        amp_mask: optional amplitude mask (frequency-domain).

        Returns:
            S: complex spectrum
            freqs: frequency bins
        """
        if z.ndim != 1:
            raise ValueError("InterferenceEngine expects 1D sequences")

        S = self.fft_backend.fft(z)
        freqs = self.fft_backend.freqs(len(z))

        if amp_mask is not None:
            freq_weights = amp_mask.freq_mask(freqs)
            if freq_weights.shape != S.shape:
                raise ValueError("freq_mask must have same shape as spectrum")
            S = S * freq_weights

        return S, freqs
