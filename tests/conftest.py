import pandas as pd

schedule_dtypes = {
    "home": "string",
    "away": "string",
    "home_goals": pd.Float64Dtype(),  # Nullable float
    "away_goals": pd.Float64Dtype(),
    "played": "string",
    "neutral": "string",
    "round": "string",
    "date": "string",
    "notes": "string",
    "country": "string",
    "updated_at": "string",
}
