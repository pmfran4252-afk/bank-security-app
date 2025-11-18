import numpy as np


class FFTBackend:
    """
    Thin abstraction over FFT operations.

    In a production setting this can be backed by:
    - NumPy
    - PyTorch
    - cuFFT
    - FFTW
    with a switch based on configuration.
    """

    def __init__(self, backend: str = "numpy"):
        if backend not in {"numpy"}:
            raise ValueError(f"Unsupported FFT backend: {backend}")
        self.backend = backend

    def fft(self, z: np.ndarray) -> np.ndarray:
        """1D complex FFT."""
        return np.fft.fft(z)

    def ifft(self, Z: np.ndarray) -> np.ndarray:
        """1D complex inverse FFT."""
        return np.fft.ifft(Z)

    def freqs(self, n: int) -> np.ndarray:
        """Return FFT frequency bins for sequence length n."""
        return np.fft.fftfreq(n)
