import pandas as pd

FEATURE_COLS = [
    'incident_severity',
    'insured_hobbies',
    'total_claim_amount',
    'vehicle_claim',
    'property_claim',
    'injury_claim',
    'policy_annual_premium',
    'months_as_customer',
    'age',
    'incident_hour_of_the_day'
]

# Ordinal mapping: 0=trivial → 3=total loss. Natural ordering, safe for trees.
SEVERITY_MAP = {
    'Trivial Damage': 0,
    'Minor Damage': 1,
    'Major Damage': 2,
    'Total Loss': 3
}

# Risk-based mapping instead of arbitrary alphabetical order.
# Higher score = higher-risk / more claim-prone hobby category.
# Extreme sports → 3, active outdoor → 2, casual → 1, sedentary → 0.
HOBBIES_RISK_MAP = {
    'base-jumping':  3,
    'skydiving':     3,
    'bungie-jumping': 3,
    'kayaking':      2,
    'cross-fit':     2,
    'paintball':     2,
    'polo':          2,
    'yachting':      2,
    'hiking':        1,
    'camping':       1,
    'dancing':       1,
    'exercise':      1,
    'golf':          1,
    'basketball':    1,
    'board-games':   0,
    'chess':         0,
    'movies':        0,
    'reading':       0,
    'sleeping':      0,
    'video-games':   0,
}

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw claim data into a cleaned, model-ready feature matrix.
    Pure function — no external state dependencies.
    """
    df_feat = df.copy()

    # Map target label if present (supports 'Y'/'N' strings and 0/1 ints)
    if 'fraud_reported' in df_feat.columns:
        df_feat['fraud_reported'] = (
            df_feat['fraud_reported'].map({'Y': 1, 'N': 0, 1: 1, 0: 0}).fillna(0).astype(int)
        )

    # Ordinal severity (natural order — safe for any model)
    df_feat['incident_severity'] = df_feat['incident_severity'].map(SEVERITY_MAP).fillna(1)

    # Risk-based hobby score (semantically meaningful; default to low-risk=0 for unknowns)
    df_feat['insured_hobbies'] = df_feat['insured_hobbies'].map(HOBBIES_RISK_MAP).fillna(0)

    # Numeric columns — coerce and fill with training-set medians
    df_feat['vehicle_claim']   = pd.to_numeric(df_feat['vehicle_claim'],   errors='coerce').fillna(37900.0)
    df_feat['property_claim']  = pd.to_numeric(df_feat['property_claim'],  errors='coerce').fillna(7400.0)
    df_feat['injury_claim']    = pd.to_numeric(df_feat['injury_claim'],    errors='coerce').fillna(7400.0)

    # Derive total if missing rather than using a fixed fallback
    df_feat['total_claim_amount'] = pd.to_numeric(df_feat['total_claim_amount'], errors='coerce').fillna(
        df_feat['vehicle_claim'] + df_feat['property_claim'] + df_feat['injury_claim']
    )

    df_feat['policy_annual_premium']   = pd.to_numeric(df_feat['policy_annual_premium'],   errors='coerce').fillna(1256.0)
    df_feat['months_as_customer']      = pd.to_numeric(df_feat['months_as_customer'],      errors='coerce').fillna(200.0)
    df_feat['age']                     = pd.to_numeric(df_feat['age'],                     errors='coerce').fillna(39.0)
    df_feat['incident_hour_of_the_day']= pd.to_numeric(df_feat['incident_hour_of_the_day'],errors='coerce').fillna(12.0)

    out_cols = FEATURE_COLS + (['fraud_reported'] if 'fraud_reported' in df_feat.columns else [])
    return df_feat[out_cols]
