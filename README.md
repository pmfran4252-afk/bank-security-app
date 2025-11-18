# QID Bank Security Demo

This is a **fictional bank security demo** showing how a **Quantum-Inspired Interference (QID) Core**
can sit on top of existing fraud/AML systems to:

- Encode transaction histories as **complex-valued waveforms**
- Apply **typology-specific amplitude masks** (CARD_FRAUD, ATO, STRUCTURING, MULE)
- Perform FFT-based **interference analysis**
- Extract compact **QID signatures**
- **Prune** to the most suspicious accounts for downstream models & analysts

## Structure

```text
qid-bank-security-demo/
├── streamlit_app.py
├── requirements.txt
├── README.md
├── qid_core/
│   ├── __init__.py
│   ├── qid_fft_backend.py
│   ├── qid_complex_encoder.py
│   ├── qid_masks.py
│   ├── qid_interference.py
│   ├── qid_signature.py
│   ├── qid_pruner.py
│   └── utils.py
└── qid_security/
    ├── __init__.py
    ├── security_event_schema.py
    ├── amplitude_masks_security.py
    ├── security_encoder.py
    ├── security_pruner.py
    └── synthetic_data_generator.py
