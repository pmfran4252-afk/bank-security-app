import numpy as np
import pandas as pd
from .security_event_schema import CHANNELS, MERCHANT_TYPES, COUNTRIES


def generate_synthetic_transactions(
    n_accounts: int = 10_000,
    days: int = 60,
    avg_txn_per_day: float = 3.0,
    fraud_ratio: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a large, fictional transaction dataset with a mix of benign and
    injected fraud/AML typology patterns.

    We inject TWO styles of suspicious behaviour for each typology:
    - "Obvious" patterns that naive rules catch easily.
    - "Subtle" patterns that QID is more likely to catch than naive rules.

    Typologies:
        - MULE
        - STRUCTURING
        - CARD_FRAUD
        - ATO
    """
    rng = np.random.default_rng(seed)
    total_expected_txns = int(n_accounts * days * avg_txn_per_day)

    # Account-level attributes
    account_ids = np.arange(1, n_accounts + 1)
    account_age_days = rng.integers(30, 365 * 10, size=n_accounts)
    risk_segment = rng.choice(
        ["LOW", "MEDIUM", "HIGH"], size=n_accounts, p=[0.6, 0.3, 0.1]
    )

    # Base transactions
    txn_ids = np.arange(1, total_expected_txns + 1)
    account_choices = rng.choice(account_ids, size=total_expected_txns, replace=True)

    day_offsets = rng.integers(0, days, size=total_expected_txns)
    second_of_day = rng.integers(0, 24 * 3600, size=total_expected_txns)
    timestamp = day_offsets * 24 * 3600 + second_of_day

    channel = rng.choice(
        CHANNELS,
        size=total_expected_txns,
        p=[0.3, 0.15, 0.2, 0.15, 0.05, 0.15],
    )
    merchant_type = rng.choice(MERCHANT_TYPES, size=total_expected_txns)
    src_country = np.array(["US"] * total_expected_txns)
    dst_country = rng.choice(
        ["US", "US", "US", "GB", "DE", "MX", "BR", "CN", "RU", "NG", "UA"],
        size=total_expected_txns,
    )

    # Amounts
    base_amount = rng.lognormal(mean=3.5, sigma=1.0, size=total_expected_txns)

    channel_factor = {
        "CARD_ECOM": 0.8,
        "ACH": 2.0,
        "ZELLE": 1.5,
        "WIRE_INTL": 5.0,
        "ATM": 1.2,
    }
    for ch_name, factor in channel_factor.items():
        mask = channel == ch_name
        base_amount[mask] *= factor

    device_age_days = rng.integers(1, 365, size=total_expected_txns)

    df = pd.DataFrame(
        {
            "txn_id": txn_ids,
            "account_id": account_choices,
            "timestamp": timestamp,
            "channel": channel,
            "merchant_type": merchant_type,
            "src_country": src_country,
            "dst_country": dst_country,
            "amount": base_amount,
            "device_age_days": device_age_days,
        }
    )

    # Attach account attributes
    acc_df = pd.DataFrame(
        {
            "account_id": account_ids,
            "account_age_days": account_age_days,
            "risk_segment": risk_segment,
        }
    )
    df = df.merge(acc_df, on="account_id", how="left")

    # Inject synthetic suspicious patterns
    df["label_fraud"] = 0
    df["label_typology"] = ""

    n_fraud_accounts = max(10, int(n_accounts * fraud_ratio))
    fraud_accounts = rng.choice(account_ids, size=n_fraud_accounts, replace=False)

    n_each = max(1, n_fraud_accounts // 4)
    mule_accounts = fraud_accounts[:n_each]
    structuring_accounts = fraud_accounts[n_each : 2 * n_each]
    card_fraud_accounts = fraud_accounts[2 * n_each : 3 * n_each]
    ato_accounts = fraud_accounts[3 * n_each :]

    # --------------------------
    # "Obvious" patterns (rules)
    # --------------------------

    # Mule pattern: rapid in/out ZELLE/ACH, cross-border
    mule_mask = df["account_id"].isin(mule_accounts)
    df.loc[
        mule_mask & df["channel"].isin(["ZELLE", "ACH"]),
        ["label_fraud", "label_typology"],
    ] = [1, "MULE"]

    # Structuring: deposits just under a threshold (8k-10k)
    struct_mask = df["account_id"].isin(structuring_accounts)
    struct_candidates = df[struct_mask & df["channel"].isin(["ACH", "ATM"])].index
    if len(struct_candidates) > 0:
        struct_idx = rng.choice(
            struct_candidates, size=max(1, len(struct_candidates) // 2), replace=False
        )
        df.loc[struct_idx, "amount"] = rng.uniform(9000, 9900, size=len(struct_idx))
        df.loc[struct_idx, ["label_fraud", "label_typology"]] = [1, "STRUCTURING"]

    # Card fraud: e-com bursts, cross-border, moderate-high value
    card_mask = df["account_id"].isin(card_fraud_accounts)
    card_candidates = df[card_mask & (df["channel"] == "CARD_ECOM")].index
    if len(card_candidates) > 0:
        card_idx = rng.choice(
            card_candidates, size=max(1, len(card_candidates) // 2), replace=False
        )
        df.loc[card_idx, "dst_country"] = rng.choice(
            ["BR", "CN", "RU", "NG"], size=len(card_idx)
        )
        df.loc[card_idx, "amount"] = rng.uniform(50, 500, size=len(card_idx))
        df.loc[card_idx, ["label_fraud", "label_typology"]] = [1, "CARD_FRAUD"]

    # ATO: new device, high-value WIRE/ZELLE
    ato_mask = df["account_id"].isin(ato_accounts)
    ato_candidates = df[ato_mask & df["channel"].isin(["WIRE_INTL", "ZELLE"])].index
    if len(ato_candidates) > 0:
        ato_idx = rng.choice(
            ato_candidates, size=max(1, len(ato_candidates) // 2), replace=False
        )
        df.loc[ato_idx, "device_age_days"] = rng.integers(1, 5, size=len(ato_idx))
        df.loc[ato_idx, "amount"] = rng.uniform(5000, 20000, size=len(ato_idx))
        df.loc[ato_idx, ["label_fraud", "label_typology"]] = [1, "ATO"]

    # ------------------------------------------------
    # "Subtle" patterns: QID-salient, rules less so
    # ------------------------------------------------

    # Subtle mule: many mid-sized domestic ZELLE/ACH flows (low amount, frequent)
    subtle_mule_candidates = df[
        mule_mask & df["channel"].isin(["ZELLE", "ACH"])
    ].index
    if len(subtle_mule_candidates) > 0:
        subtle_mule_idx = rng.choice(
            subtle_mule_candidates,
            size=max(1, len(subtle_mule_candidates) // 3),
            replace=False,
        )
        df.loc[subtle_mule_idx, "dst_country"] = "US"
        df.loc[subtle_mule_idx, "amount"] = rng.uniform(150, 600, size=len(subtle_mule_idx))
        df.loc[subtle_mule_idx, ["label_fraud", "label_typology"]] = [1, "MULE_SUBTLE"]

    # Subtle structuring: repeated mid-range deposits (3k-5k) – frequency pattern, not threshold
    subtle_struct_candidates = df[
        struct_mask & df["channel"].isin(["ACH", "ATM"])
    ].index
    if len(subtle_struct_candidates) > 0:
        subtle_struct_idx = rng.choice(
            subtle_struct_candidates,
            size=max(1, len(subtle_struct_candidates) // 3),
            replace=False,
        )
        df.loc[subtle_struct_idx, "amount"] = rng.uniform(
            3000, 5000, size=len(subtle_struct_idx)
        )
        df.loc[subtle_struct_idx, ["label_fraud", "label_typology"]] = [
            1,
            "STRUCTURING_SUBTLE",
        ]

    # Subtle card fraud: domestic CNP bursts, small-medium tickets – pattern, not size
    subtle_card_candidates = df[
        card_mask & (df["channel"] == "CARD_ECOM")
    ].index
    if len(subtle_card_candidates) > 0:
        subtle_card_idx = rng.choice(
            subtle_card_candidates,
            size=max(1, len(subtle_card_candidates) // 3),
            replace=False,
        )
        df.loc[subtle_card_idx, "dst_country"] = rng.choice(
            ["US", "GB", "DE"], size=len(subtle_card_idx)
        )
        df.loc[subtle_card_idx, "amount"] = rng.uniform(
            20, 180, size=len(subtle_card_idx)
        )
        df.loc[subtle_card_idx, ["label_fraud", "label_typology"]] = [
            1,
            "CARD_FRAUD_SUBTLE",
        ]

    # Subtle ATO: new device, moderate-value WIRE/ZELLE (1k-2.5k), often domestic
    subtle_ato_candidates = df[
        ato_mask & df["channel"].isin(["WIRE_INTL", "ZELLE"])
    ].index
    if len(subtle_ato_candidates) > 0:
        subtle_ato_idx = rng.choice(
            subtle_ato_candidates,
            size=max(1, len(subtle_ato_candidates) // 3),
            replace=False,
        )
        df.loc[subtle_ato_idx, "device_age_days"] = rng.integers(
            1, 5, size=len(subtle_ato_idx)
        )
        df.loc[subtle_ato_idx, "amount"] = rng.uniform(
            1000, 2500, size=len(subtle_ato_idx)
        )
        df.loc[subtle_ato_idx, "dst_country"] = rng.choice(
            ["US", "US", "GB", "DE"], size=len(subtle_ato_idx)
        )
        df.loc[subtle_ato_idx, ["label_fraud", "label_typology"]] = [
            1,
            "ATO_SUBTLE",
        ]

    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    return df
