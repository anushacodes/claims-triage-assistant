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

SEVERITY_MAP = {
    'Trivial Damage': 0,
    'Minor Damage': 1,
    'Major Damage': 2,
    'Total Loss': 3
}

HOBBIES_LIST = [
    'base-jumping', 'basketball', 'board-games', 'bungie-jumping', 'camping',
    'chess', 'cross-fit', 'dancing', 'exercise', 'golf', 'hiking', 'kayaking',
    'movies', 'paintball', 'polo', 'reading', 'skydiving', 'sleeping',
    'video-games', 'yachting'
]
HOBBIES_MAP = {hobby: i for i, hobby in enumerate(HOBBIES_LIST)}

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw claim data into a cleaned, model-ready feature matrix.
    Pure function with no external state dependencies.
    """
    df_feat = df.copy()
    
    # Map target label if present
    if 'fraud_reported' in df_feat.columns:
        # Support both 'Y'/'N' and binary formats
        df_feat['fraud_reported'] = df_feat['fraud_reported'].map({'Y': 1, 'N': 0, 1: 1, 0: 0}).fillna(0)
        
    # Map categoricals (defaulting to sensible values if unseen/nan)
    df_feat['incident_severity'] = df_feat['incident_severity'].map(SEVERITY_MAP).fillna(1)
    df_feat['insured_hobbies'] = df_feat['insured_hobbies'].map(HOBBIES_MAP).fillna(15) # 15 matches 'reading'
    
    # Parse numericals and fill with median/typical defaults
    df_feat['vehicle_claim'] = pd.to_numeric(df_feat['vehicle_claim'], errors='coerce').fillna(37900.0)
    df_feat['property_claim'] = pd.to_numeric(df_feat['property_claim'], errors='coerce').fillna(7400.0)
    df_feat['injury_claim'] = pd.to_numeric(df_feat['injury_claim'], errors='coerce').fillna(7400.0)
    
    # Ensure total claim matches sum if missing
    df_feat['total_claim_amount'] = pd.to_numeric(df_feat['total_claim_amount'], errors='coerce').fillna(
        df_feat['vehicle_claim'] + df_feat['property_claim'] + df_feat['injury_claim']
    )
    
    df_feat['policy_annual_premium'] = pd.to_numeric(df_feat['policy_annual_premium'], errors='coerce').fillna(1256.0)
    df_feat['months_as_customer'] = pd.to_numeric(df_feat['months_as_customer'], errors='coerce').fillna(200.0)
    df_feat['age'] = pd.to_numeric(df_feat['age'], errors='coerce').fillna(39.0)
    df_feat['incident_hour_of_the_day'] = pd.to_numeric(df_feat['incident_hour_of_the_day'], errors='coerce').fillna(12.0)
    
    # Return output dataframe with exact selected columns
    out_cols = FEATURE_COLS
    if 'fraud_reported' in df_feat.columns:
        out_cols = FEATURE_COLS + ['fraud_reported']
        
    return df_feat[out_cols]
