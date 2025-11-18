"""
Defines the basic schema for synthetic bank transactions used in the demo.

Each row in the synthetic dataset includes at least:

- txn_id: unique integer transaction id
- account_id: account identifier
- timestamp: seconds since a fictional start time
- channel: CARD_POS, CARD_ECOM, ACH, ZELLE, WIRE_INTL, ATM
- merchant_type: coarse-grained MCC-like label
- src_country, dst_country
- amount: transaction amount
- device_age_days: days since the device was first seen
- account_age_days: days since account was opened
- risk_segment: LOW / MEDIUM / HIGH

Additionally, we inject:

- label_fraud: 0/1 indicating synthetic suspicious activity
- label_typology: one of {"MULE", "STRUCTURING", "CARD_FRAUD", "ATO", ""}

This schema is intentionally simple, but realistic enough to demonstrate QID.
"""

CHANNELS = ["CARD_POS", "CARD_ECOM", "ACH", "ZELLE", "WIRE_INTL", "ATM"]

MERCHANT_TYPES = [
    "GROCERY",
    "RESTAURANT",
    "RETAIL",
    "TRAVEL",
    "DIGITAL_SERVICES",
    "GAS",
    "HOTEL",
    "ATM",
    "PAYROLL",
    "BILLPAY",
]

COUNTRIES = ["US", "MX", "BR", "CN", "RU", "NG", "UA", "GB", "DE"]
