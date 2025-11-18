"""
Bank-security specific integration for QID Core.

This package:
- Defines the synthetic transaction schema.
- Implements security-specific encoders and pruners.
- Provides a synthetic_data_generator to simulate large fictional datasets.

Heavy / optional pieces like amplitude masks are imported explicitly
by callers (e.g., from qid_security.amplitude_masks_security import get_security_mask),
to avoid import-time issues.
"""

from .security_event_schema import CHANNELS, MERCHANT_TYPES, COUNTRIES
from .security_encoder import SecurityInterferenceEncoder
from .security_pruner import SecurityPruner
from .synthetic_data_generator import generate_synthetic_transactions

__all__ = [
    "CHANNELS",
    "MERCHANT_TYPES",
    "COUNTRIES",
    "SecurityInterferenceEncoder",
    "SecurityPruner",
    "generate_synthetic_transactions",
]
