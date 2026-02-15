import pandas as pd

def _fmt_money(x) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return f"₹{v:,.2f}"

def display_or_dash(v, dash="–") -> str:
    if v is None:
        return dash
    s = str(v).strip()
    return dash if s == "" or s.lower() in ("nan", "none", "-", "–") else s

def df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    # keep YOUR existing implementation here (the one you already wrote)
    return df
