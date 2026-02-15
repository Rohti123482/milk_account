import pandas as pd

def df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make df safe for Streamlit (Arrow) display.
    """
    if df is None:
        return pd.DataFrame()

    out = df.copy()
    out = out.replace({None: "–"}).fillna("–")

    for col in out.columns:
        if out[col].dtype == "object":
            def _norm(v):
                if v is None:
                    return "–"
                if isinstance(v, (bytes, bytearray)):
                    try:
                        return v.decode("utf-8", errors="ignore")
                    except Exception:
                        return str(v)
                return str(v)

            out[col] = out[col].map(_norm)

    return out
