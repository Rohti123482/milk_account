import pandas as pd
import pyarrow as pa

from app import df_for_display

def test_df_for_display_arrow_safe():
    df = pd.DataFrame({
        "BUFFALO": [b"10", 12.5, None, "-"],
        "Sales (₹)": [100.0, 0.0, 50.0, 25.0],
    })
    out = df_for_display(df)
    pa.Table.from_pandas(out)  # will crash if types are mixed
