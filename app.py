import streamlit as st
import pandas as pd
from datetime import date,  timedelta
import time
import zipfile
import io
import plotly.express as px
from supabase import create_client


# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Milk Accounting Pro", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{
  --bg: #F6F7FB;
  --card: rgba(255,255,255,0.92);
  --text: #0F172A;
  --muted: #64748B;
  --border: rgba(15, 23, 42, 0.10);
  --shadow: 0 10px 30px rgba(2, 6, 23, 0.08);
  --shadow2: 0 6px 18px rgba(2, 6, 23, 0.08);
  --radius: 18px;
  --primary: #22C55E;
  --primary2: #16A34A;
}

.stApp {
  background: radial-gradient(1000px 600px at 10% 0%, rgba(34,197,94,0.10), transparent 55%),
              radial-gradient(900px 500px at 90% 10%, rgba(59,130,246,0.08), transparent 55%),
              linear-gradient(180deg, var(--bg) 0%, #EEF2FF 120%);
}

.block-container { padding-top: 1.2rem !important; padding-bottom: 2.5rem !important; }

section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.75) !important;
  backdrop-filter: blur(10px);
  border-right: 1px solid var(--border);
}

h1,h2,h3,h4 { letter-spacing: -0.02em !important; }
h1 { font-weight: 900 !important; }
h2 { font-weight: 800 !important; }
h3 { color: var(--muted) !important; font-weight: 700 !important; }

.stButton>button{
  background: linear-gradient(180deg, var(--primary) 0%, var(--primary2) 100%) !important;
  color:#fff !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 12px !important;
  padding: 0.65rem 1.05rem !important;
  font-weight: 800 !important;
  box-shadow: 0 10px 18px rgba(34,197,94,0.18) !important;
  transition: transform 0.08s ease, filter 0.12s ease !important;
}
.stButton>button:hover{ filter: brightness(1.03); transform: translateY(-1px); }
.stButton>button:active{ transform: translateY(0px); }

div[data-baseweb="input"]>div,
div[data-baseweb="select"]>div,
div[data-baseweb="textarea"]>div{
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.85) !important;
  box-shadow: none !important;
}
label { color: var(--muted) !important; font-weight: 700 !important; }

.stTabs [data-baseweb="tab-list"]{ gap:10px; }
.stTabs [data-baseweb="tab"]{
  background: rgba(255,255,255,0.55) !important;
  border: 1px solid var(--border) !important;
  border-radius: 999px !important;
  padding: 10px 14px !important;
  font-weight: 800 !important;
  color: var(--muted) !important;
}
.stTabs [aria-selected="true"]{
  background: rgba(34,197,94,0.12) !important;
  border: 1px solid rgba(34,197,94,0.35) !important;
  color: var(--text) !important;
}

div[data-testid="stMetric"]{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px !important;
  box-shadow: var(--shadow2) !important;
}
div[data-testid="stMetric"] label{ color: var(--muted) !important; font-weight: 800 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"]{
  font-size: 1.65rem !important;
  font-weight: 950 !important;
}

div[data-testid="stDataFrame"]{
  background: var(--card) !important;
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  box-shadow: var(--shadow) !important;
  overflow: hidden !important;
}

details[data-testid="stExpander"]{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow2) !important;
  overflow: hidden !important;
}

hr { border-color: rgba(15,23,42,0.08) !important; }

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
/* DON'T hide the header; it contains the sidebar toggle */
/* header {visibility:hidden;} */

</style>
""", unsafe_allow_html=True)


# Keep same "file names" so your app code doesn't change
RETAILERS_FILE = "data/retailers.csv"
CATEGORIES_FILE = "data/categories.csv"
PRICES_FILE = "data/prices.csv"
ENTRIES_FILE = "data/entries.csv"
PAYMENTS_FILE = "data/payments.csv"
DISTRIBUTORS_FILE = "data/distributors.csv"
DISTRIBUTOR_PURCHASES_FILE = "data/distributor_purchases.csv"
DISTRIBUTOR_PAYMENTS_FILE = "data/distributor_payments.csv"
DISTRIBUTOR_CATEGORY_MAP_FILE = "data/distributor_category_map.csv"
WASTAGE_FILE = "data/wastage.csv"
EXPENSES_FILE = "data/expenses.csv"

GLOBAL_RETAILER_ID = -1

FILE_TO_TABLE = {
    RETAILERS_FILE: ("retailers", "retailer_id"),
    CATEGORIES_FILE: ("categories", "category_id"),
    PRICES_FILE: ("prices", "price_id"),
    ENTRIES_FILE: ("entries", "entry_id"),
    PAYMENTS_FILE: ("payments", "payment_id"),
    DISTRIBUTORS_FILE: ("distributors", "distributor_id"),
    DISTRIBUTOR_PURCHASES_FILE: ("distributor_purchases", "purchase_id"),
    DISTRIBUTOR_PAYMENTS_FILE: ("distributor_payments", "payment_id"),
    DISTRIBUTOR_CATEGORY_MAP_FILE: ("distributor_category_map", "map_id"),
    WASTAGE_FILE: ("wastage", "wastage_id"),
    EXPENSES_FILE: ("expenses", "expense_id"),
}

@st.cache_resource
def get_sb():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["anon_key"]
    )

sb = None
try:
    sb = get_sb()
except Exception as e:
    st.error("Supabase client init failed. Check .streamlit/secrets.toml and supabase dependency.")
    st.error(str(e))
    st.stop()

if st.sidebar.button("🔌 Test DB Connection"):
    try:
        sb.table("retailers").select("retailer_id").limit(1).execute()
        st.sidebar.success("Supabase connected ✅")
    except Exception as e:
        st.sidebar.error(f"Connection failed ❌\n{e}")



# ================== CACHE INVALIDATION (PERFORMANCE) ==================
def invalidate_data_cache() -> None:
    """Force the UI to reload fresh data after any write/delete."""
    # 1) bump version (used by editor keys + cached wrappers)
    st.session_state["data_version"] = int(st.session_state.get("data_version", 0)) + 1

    # 2) clear Streamlit caches (important when DB changes)
    try:
        load_and_migrate_data_cached.clear()  # type: ignore[name-defined]
    except Exception:
        pass
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass




@st.cache_data(show_spinner=False)
def make_full_backup_zip(data_version: int) -> bytes:
    """Create a ZIP containing CSV exports of all core tables. Cached by data_version."""
    _ = data_version  # cache key only

    tables = {
        "retailers": retailers,
        "categories": categories,
        "prices": prices,
        "entries": entries,
        "payments": payments,
        "distributors": distributors,
        "distributor_purchases": dist_purchases,
        "distributor_payments": dist_payments,
        "distributor_category_map": dist_cat_map,
        "wastage": wastage,
        "expenses": expenses,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            if df is None:
                continue
            if not isinstance(df, pd.DataFrame):
                continue
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            zf.writestr(f"{name}.csv", csv_bytes)
    return buf.getvalue()

@st.cache_data(show_spinner=False)
def load_and_migrate_data_cached(_version: int):
    """Cached wrapper: prevents full-table fetch on every widget interaction."""
    _ = int(_version)  # IMPORTANT: makes cache depend on data_version
    return load_and_migrate_data()
def _secret_bool(key: str, default: bool = False) -> bool:
    try:
        raw = st.secrets["supabase"].get(key, default)
    except Exception:
        return default
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(raw)

# Toggle DB-assigned IDs and server-side filtering via secrets if desired.
USE_DB_IDS = _secret_bool("use_db_ids", False)
USE_SERVER_FILTERS = _secret_bool("use_server_filters", True)

if not USE_SERVER_FILTERS:
    st.warning('⚠️ App is running in LOCAL CSV mode. If you delete data in Supabase, it will not reflect here. Turn on Supabase mode in secrets (use_server_filters=True).')
# ================== SCHEMAS ==================
CSV_SCHEMAS = {
    RETAILERS_FILE: ["retailer_id", "name", "contact", "address", "zone", "is_active"],
    CATEGORIES_FILE: ["category_id", "name", "description", "default_price", "is_active"],
    PRICES_FILE: ["price_id", "retailer_id", "category_id", "price", "effective_date"],
    ENTRIES_FILE: ["entry_id", "date", "retailer_id", "category_id", "qty", "rate", "amount"],
    PAYMENTS_FILE: ["payment_id", "date", "retailer_id", "amount", "payment_mode", "note"],
    DISTRIBUTORS_FILE: ["distributor_id", "name", "contact", "address", "is_active"],
    DISTRIBUTOR_PURCHASES_FILE: ["purchase_id", "date", "distributor_id", "category_id", "qty", "rate", "amount"],
    DISTRIBUTOR_PAYMENTS_FILE: ["payment_id", "date", "distributor_id", "amount", "payment_mode", "note"],
    DISTRIBUTOR_CATEGORY_MAP_FILE: ["map_id", "distributor_id", "category_id", "is_active"],
    WASTAGE_FILE: ["wastage_id", "date", "category_id", "qty", "reason", "estimated_loss"],
    EXPENSES_FILE: ["expense_id", "date", "category", "description", "amount", "payment_mode", "paid"],
}

# legacy columns allowed to be missing in old CSVs (in-memory only, no write on startup)
ALLOWED_LEGACY_MISSING = {
    RETAILERS_FILE: {"zone", "is_active", "contact", "address"},
    CATEGORIES_FILE: {"description", "default_price", "is_active"},
    DISTRIBUTORS_FILE: {"contact", "address", "is_active"},
}

# ================== HELPERS ==================
def _is_missingish(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    s = str(v).strip()
    return s == "" or s.lower() in ("nan", "none", "-", "–")

def _disp_2dec_or_dash(v, dash="–") -> str:
    """Show number with 2 decimals, else dash. 0 => dash (matches your UI logic)."""
    if _is_missingish(v):
        return dash
    try:
        fv = float(v)
        return dash if fv == 0 else f"{fv:.2f}"
    except Exception:
        return dash

def _disp_rate_or_dash(v, dash="–") -> str:
    """Rate display: 2 decimals, but keep dash for missing. 0 => dash."""
    return _disp_2dec_or_dash(v, dash=dash)
# ================== DB WRITE HELPERS (SAFE, TARGETED) ==================
def sb_table_for_path(path: str) -> tuple[str, str]:
    if path not in FILE_TO_TABLE:
        raise ValueError(f"Unknown mapping for: {path}")
    return FILE_TO_TABLE[path]


def sb_next_id(table: str, pk: str) -> int:
    """
    Get next integer ID safely from DB (max(pk)+1).
    Prevents collisions across multiple sessions/users.
    """
    sb = get_sb()
    resp = sb.table(table).select(pk).order(pk, desc=True).limit(1).execute()
    data = resp.data or []
    if not data:
        return 1
    try:
        return int(data[0][pk]) + 1
    except Exception:
        return 1


def sb_new_id(table: str, pk: str):
    return None if USE_DB_IDS else sb_next_id(table, pk)


def _safe_dt(s):
    return pd.to_datetime(s, errors="coerce")
def build_entries_view(df: pd.DataFrame, want_milk_type_col: bool = False) -> pd.DataFrame:
    """
    Builds a clean, UI-ready view of entries with Retailer/Category/Zone names.
    Keeps your existing calls working (want_milk_type_col kept for compatibility).
    """
    if df is None or df.empty:
        cols = ["entry_id", "date", "zone", "Retailer", "Category", "qty", "rate", "amount"]
        return pd.DataFrame(columns=cols)

    out = df.copy()

    # Normalize date
    out["date"] = _safe_dt(out["date"]).dt.strftime("%Y-%m-%d")

    # Make sure IDs are numeric for joins
    out["retailer_id"] = pd.to_numeric(out.get("retailer_id", 0), errors="coerce").fillna(0).astype(int)
    out["category_id"] = pd.to_numeric(out.get("category_id", 0), errors="coerce").fillna(0).astype(int)

    # Merge retailer name + zone
    if "retailers" in globals() and isinstance(retailers, pd.DataFrame) and not retailers.empty:
        rmap = retailers[["retailer_id", "name", "zone"]].copy()
        rmap["retailer_id"] = pd.to_numeric(rmap["retailer_id"], errors="coerce").fillna(0).astype(int)
        rmap["zone"] = rmap["zone"].apply(_norm_zone)
        out = out.merge(rmap, on="retailer_id", how="left")
        out = out.rename(columns={"name": "Retailer"})
    else:
        out["Retailer"] = "-"

    # Merge category name
    if "categories" in globals() and isinstance(categories, pd.DataFrame) and not categories.empty:
        cmap = categories[["category_id", "name"]].copy()
        cmap["category_id"] = pd.to_numeric(cmap["category_id"], errors="coerce").fillna(0).astype(int)
        out = out.merge(cmap, on="category_id", how="left")
        out = out.rename(columns={"name": "Category"})
    else:
        out["Category"] = "-"

    # Clean fallbacks
    out["Retailer"] = out.get("Retailer", "-").fillna("-").astype(str)
    out["Category"] = out.get("Category", "-").fillna("-").astype(str)
    out["zone"] = out.get("zone", "Default").fillna("Default").astype(str).apply(_norm_zone)

    # Ensure numeric columns
    for c in ["qty", "rate", "amount"]:
        out[c] = pd.to_numeric(out.get(c, 0.0), errors="coerce").fillna(0.0).astype(float)

    cols = ["entry_id", "date", "zone", "Retailer", "Category", "qty", "rate", "amount"]
    # keep only what UI uses
    for c in cols:
        if c not in out.columns:
            out[c] = "-" if c in ["zone", "Retailer", "Category", "date"] else 0

    return out[cols]


@st.cache_data(show_spinner=False)
def build_entries_view_cached(df: pd.DataFrame, data_version: int, want_milk_type_col: bool = False) -> pd.DataFrame:
    """Cached wrapper around build_entries_view to avoid repeated merges on every rerun."""
    _ = data_version  # cache key only
    return build_entries_view(df, want_milk_type_col=want_milk_type_col)


def _norm_zone(z: str) -> str:
    z = "" if z is None else str(z).strip()
    return "Default" if not z else " ".join(z.split()).title()

def parse_boolish_active(v) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in ("false", "0", "no", "n", "inactive", "off"):
        return False
    if s in ("true", "1", "yes", "y", "active", "on"):
        return True
    try:
        return bool(int(float(s)))
    except Exception:
        return True  # default active

def fmt_zero_dash(x) -> str:
    """Format numeric values; show en-dash for zero/blank/invalid."""
    if x in (None, "", "–", "-", "—"):
        v = 0.0
    else:
        try:
            v = float(x)
        except Exception:
            v = 0.0
    return "–" if v == 0.0 else f"{v:.2f}"


def parse_boolish_paid(v) -> bool:
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "paid", "on"):
        return True
    if s in ("false", "0", "no", "n", "unpaid", "off"):
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return False

def next_id_from_df(df: pd.DataFrame, id_col: str) -> int:
    if df is None or df.empty or id_col not in df.columns:
        return 1
    s = pd.to_numeric(df[id_col], errors="coerce").dropna()
    return int(s.max()) + 1 if not s.empty else 1

def _fmt_money(x) -> str:
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return "₹0.00"

def display_or_dash(v, dash="–") -> str:
    s = "" if v is None else str(v).strip()
    return dash if not s else s

def df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make df safe for Streamlit (Arrow) display:
    - convert bytes -> str
    - ensure object columns are consistent (string)
    - replace None/NaN with "–"
    """
    if df is None:
        return pd.DataFrame()

    out = df.copy()

    # Replace None/NaN first
    out = out.replace({None: "–"}).fillna("–")

    # Fix Arrow crash: object columns containing mixed bytes/float/etc.
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

def sb_fetch_all(table: str, cols="*", page_size: int = 1000, max_retries: int = 5):
    sb = get_sb()
    out = []
    offset = 0

    while True:
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = sb.table(table).select(cols).range(offset, offset + page_size - 1).execute()
                batch = resp.data or []
                out.extend(batch)

                if len(batch) < page_size:
                    return out

                offset += page_size
                last_exc = None
                break  # success, exit retry loop

            except Exception as e:
                last_exc = e
                time.sleep(0.3 * attempt)  # small backoff

        if last_exc is not None:
            raise last_exc



def _normalize_df_from_rows(path: str, columns: list[str], rows: list[dict]) -> pd.DataFrame:
    if not rows:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(rows)

    # Ensure schema columns exist (create missing as blank/None)
    for c in columns:
        if c not in df.columns:
            df[c] = None

    # Allow legacy missing columns without crashing
    legacy_ok = ALLOWED_LEGACY_MISSING.get(path, set())
    for c in legacy_ok:
        if c not in df.columns:
            df[c] = None

    # Keep only schema columns (and legacy allowed ones if they are in schema)
    keep = [c for c in columns if c in df.columns]
    return df[keep].copy()


def sb_fetch_where(table: str, cols: str = "*", filters: list[tuple] | None = None, page_size: int = 1000, in_chunk: int = 500) -> list[dict]:
    """
    Fetch rows from Supabase table with server-side filters.
    Supported ops: eq, lt, lte, gt, gte, in
    """
    sb = get_sb()
    filters = filters or []
    base = sb.table(table).select(cols)

    in_filters = []
    for col, op, val in filters:
        if op == "in":
            vals = list(val) if isinstance(val, (list, tuple, set)) else [val]
            in_filters.append((col, vals))
        elif op == "eq":
            base = base.eq(col, val)
        elif op == "lt":
            base = base.lt(col, val)
        elif op == "lte":
            base = base.lte(col, val)
        elif op == "gt":
            base = base.gt(col, val)
        elif op == "gte":
            base = base.gte(col, val)
        else:
            raise ValueError(f"Unsupported op: {op}")

    def _fetch(q):
        out = []
        offset = 0
        while True:
            resp = q.range(offset, offset + page_size - 1).execute()
            batch = resp.data or []
            out.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return out

    if not in_filters:
        return _fetch(base)

    first_col, first_vals = in_filters[0]
    rest_in = in_filters[1:]

    out = []
    for i in range(0, len(first_vals), in_chunk):
        q = base.in_(first_col, first_vals[i:i + in_chunk])
        for col, vals in rest_in:
            q = q.in_(col, vals)
        out.extend(_fetch(q))
    return out


def sb_fetch_df(path: str, columns: list[str], filters: list[tuple] | None = None) -> pd.DataFrame:
    if not filters:
        return safe_read_csv(path, columns)
    if path not in FILE_TO_TABLE:
        return pd.DataFrame(columns=columns)
    table, _ = FILE_TO_TABLE[path]
    try:
        rows = sb_fetch_where(table, cols="*", filters=filters)
        return _normalize_df_from_rows(path, columns, rows)
    except Exception:
        return safe_read_csv(path, columns)


def _prepare_df_for_write(df: pd.DataFrame, path: str) -> pd.DataFrame:
    df = df.copy() if df is not None else pd.DataFrame()
    expected_cols = CSV_SCHEMAS.get(path, [])
    if not expected_cols:
        table, _ = FILE_TO_TABLE[path]
        raise RuntimeError(f"{table}: missing schema definition for {path}")

    # Drop any unexpected columns to avoid Supabase "column does not exist" errors.
    extra_cols = [c for c in df.columns if c not in expected_cols]
    if extra_cols:
        df = df.drop(columns=extra_cols, errors="ignore")

    # Ensure all expected columns exist (missing columns become NULLs).
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    # Keep stable column order to match schema.
    return df[expected_cols]

def sb_insert_df(df: object, path: str) -> None:
    # Normalize inputs: allow dict / list[dict] / DataFrame
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    elif isinstance(df, list):
        df = pd.DataFrame(df)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"sb_insert_df expects a DataFrame/dict/list[dict], got {type(df)}")

    """Insert rows into Supabase and then persist a local snapshot CSV.

    Supports tables where the primary key is GENERATED in Postgres (IDENTITY / serial).
    In that case you may pass a dataframe WITHOUT the pk column, or with pk all-NULL.
    We insert without the pk and then use the returned rows (with generated pk) for the snapshot.
    """
    table, pk = FILE_TO_TABLE[path]
    # Primary key handling:
    # - If USE_DB_IDS=True: assign IDs in the app (legacy/CSV-friendly mode)
    # - If USE_DB_IDS=False: let Postgres identity/serial generate IDs (recommended)
    if pk not in df.columns:
        df[pk] = None
    if USE_DB_IDS:
        if df[pk].isna().any():
            next_id = sb_next_id(table, pk)
            mask = df[pk].isna()
            df.loc[mask, pk] = range(next_id, next_id + int(mask.sum()))


    # Accept dict / list[dict] input (some call sites build a single-row dict)
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    elif isinstance(df, list) and df and isinstance(df[0], dict):
        df = pd.DataFrame(df)

    df = _prepare_df_for_write(df, path)

    # If pk is missing OR entirely NULL -> treat as DB-generated PK.
    pk_missing = pk not in df.columns
    pk_all_null = (not pk_missing) and df[pk].isna().all()

    records = df.to_dict(orient="records")

    # Strip pk when DB generates it (recommended for production).
    if (not USE_DB_IDS) and (pk_missing or pk_all_null):
        for r in records:
            r.pop(pk, None)

        resp = sb.table(table).insert(records).execute()
        inserted = resp.data or []
        if not inserted:
            raise RuntimeError(f"{table}: insert returned no rows")
        df = pd.DataFrame(inserted)
        df = _prepare_df_for_write(df, path)
    else:
        # Safety: reject mixed NULL / non-NULL pk values (usually a bug in the UI)
        if df[pk].isna().any():
            raise RuntimeError(f"{table}: mixed NULL/non-NULL primary keys while inserting")

        for i in range(0, len(records), 500):
            sb.table(table).insert(records[i:i+500]).execute()

    # Persist snapshot + refresh caches
    safe_write_csv(df, path, allow_empty=False)
    invalidate_data_cache()


def sb_delete_by_pk(table: str, pk: str, ids: list[int], chunk: int = 500) -> None:
    sb = get_sb()
    ids = [int(x) for x in ids if x is not None]
    if not ids:
        return
    for i in range(0, len(ids), chunk):
        sb.table(table).delete().in_(pk, ids[i:i+chunk]).execute()

    invalidate_data_cache()



def sb_delete_where(table: str, filters: list[tuple], in_chunk: int = 500) -> None:
    """
    FIXED: preserves ALL filters even when using IN().
    filters: list of tuples like:
      ("date","eq","2026-02-05")
      ("retailer_id","in",[1,2,3])
    Supported ops: eq, lt, lte, gt, gte, in
    """
    sb = get_sb()
    base = sb.table(table).delete()

    in_filters = []
    for col, op, val in filters:
        if op == "in":
            vals = list(val) if isinstance(val, (list, tuple, set)) else [val]
            in_filters.append((col, vals))
        elif op == "eq":
            base = base.eq(col, val)
        elif op == "lt":
            base = base.lt(col, val)
        elif op == "lte":
            base = base.lte(col, val)
        elif op == "gt":
            base = base.gt(col, val)
        elif op == "gte":
            base = base.gte(col, val)
        else:
            raise ValueError(f"Unsupported op: {op}")

    if not in_filters:
        base.execute()
        invalidate_data_cache()
        return

    first_col, first_vals = in_filters[0]
    rest_in = in_filters[1:]

    for i in range(0, len(first_vals), in_chunk):
        q = base.in_(first_col, first_vals[i:i + in_chunk])
        for col, vals in rest_in:
            q = q.in_(col, vals)
        q.execute()

    invalidate_data_cache()

def safe_write_csv(df: pd.DataFrame, path: str, allow_empty: bool = False) -> None:
    """
    DB MODE:
    - Upsert rows only (NO implicit deletes).
    - Deletions must be explicit via sb_delete_* helpers.
    """
    if path not in FILE_TO_TABLE:
        raise ValueError(f"Unknown mapping for: {path}")

    table, pk = FILE_TO_TABLE[path]
    df = _prepare_df_for_write(df, path)

    if df.empty and not allow_empty:
        raise RuntimeError(
            f"Refusing to write EMPTY dataframe to {table}. "
            f"Pass allow_empty=True only if you truly want to write nothing."
        )
    if df.empty:
        return

    # If PK is entirely NULL, treat it as "DB-generated IDs": INSERT (not UPSERT).
    # (Your caller should usually use sb_insert_df for this case, but this makes it safe.)
    if pk in df.columns and df[pk].isna().all():
        payload = df.drop(columns=[pk], errors="ignore")
        records = payload.where(pd.notna(payload), None).to_dict(orient="records")
        chunk = 500
        for i in range(0, len(records), chunk):
            sb.table(table).insert(records[i:i+chunk]).execute()
        invalidate_data_cache()
        return

    # Mixed NULL/non-NULL PK is dangerous (can create duplicates).
    if pk in df.columns and df[pk].isna().any():
        raise RuntimeError(f"{table}: mixed NULL/non-NULL primary keys")

    # Defensive: PK must exist for UPSERT
    if pk not in df.columns:
        raise RuntimeError(f"{table}: missing primary key column '{pk}' for upsert")

    if df[pk].duplicated().any():
        dups = df[df[pk].duplicated(keep=False)][pk].tolist()[:10]
        raise RuntimeError(f"{table}: duplicate primary keys in dataframe (sample): {dups}")

    records = df.where(pd.notna(df), None).to_dict(orient="records")
    chunk = 500
    for i in range(0, len(records), chunk):
        sb.table(table).upsert(records[i:i+chunk], on_conflict=pk).execute()

    invalidate_data_cache()


def safe_write_two_csvs(df1: pd.DataFrame, path1: str, df2: pd.DataFrame, path2: str) -> None:
    safe_write_csv(df1, path1, allow_empty=True)
    safe_write_csv(df2, path2, allow_empty=True)

# ================== DISTRIBUTOR LEDGER + BILL HELPERS ==================

def safe_read_csv(path: str, columns: list[str]) -> pd.DataFrame:
    """
    DB MODE:
    - Reads from Supabase tables (NOT local CSV).
    - Returns a DataFrame with exactly the expected schema columns (plus allowed legacy missing columns).
    """
    if path not in FILE_TO_TABLE:
        # fallback: return empty with the requested columns
        return pd.DataFrame(columns=columns)

    table, pk = FILE_TO_TABLE[path]

    rows = sb_fetch_all(table, cols="*")
    if not rows:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(rows)

    # Ensure schema columns exist (create missing as blank/None)
    for c in columns:
        if c not in df.columns:
            df[c] = None

    # Allow legacy missing columns without crashing
    legacy_ok = ALLOWED_LEGACY_MISSING.get(path, set())
    for c in legacy_ok:
        if c not in df.columns:
            df[c] = None

    # Keep only schema columns (and legacy allowed ones if they are in schema)
    keep = [c for c in columns if c in df.columns]
    df = df[keep].copy()

    return df


def distributor_balance_before(distributor_id: int, start_day: date) -> float:
    """
    Due before start_day = (purchases before) - (payments before)
    Positive => you owe distributor.
    """
    if USE_SERVER_FILTERS:
        dp = sb_fetch_df(
            DISTRIBUTOR_PURCHASES_FILE,
            CSV_SCHEMAS[DISTRIBUTOR_PURCHASES_FILE],
            filters=[
                ("distributor_id", "eq", int(distributor_id)),
                ("date", "lt", str(start_day)),
            ],
        )
        pay = sb_fetch_df(
            DISTRIBUTOR_PAYMENTS_FILE,
            CSV_SCHEMAS[DISTRIBUTOR_PAYMENTS_FILE],
            filters=[
                ("distributor_id", "eq", int(distributor_id)),
                ("date", "lt", str(start_day)),
            ],
        )
    else:
        dp = dist_purchases.copy()
        pay = dist_payments.copy()

    purchases_amt = 0.0
    paid_amt = 0.0

    if not dp.empty:
        dp["date"] = _safe_dt(dp["date"]).dt.date
        dp = dp.loc[
            (dp["distributor_id"].astype(int) == int(distributor_id)) &
            (dp["date"] < start_day)
        ].copy()
        if not dp.empty:
            # trust stored amount but normalize if missing
            if "amount" in dp.columns:
                purchases_amt = float(pd.to_numeric(dp["amount"], errors="coerce").fillna(0).sum())
            else:
                purchases_amt = float((pd.to_numeric(dp["qty"], errors="coerce").fillna(0) * pd.to_numeric(dp["rate"], errors="coerce").fillna(0)).sum())

    if not pay.empty:
        pay["date"] = _safe_dt(pay["date"]).dt.date
        pay = pay.loc[
            (pay["distributor_id"].astype(int) == int(distributor_id)) &
            (pay["date"] < start_day)
        ].copy()
        if not pay.empty:
            paid_amt = float(pd.to_numeric(pay["amount"], errors="coerce").fillna(0).sum())

    return float(purchases_amt - paid_amt)


def build_distributor_daily_grid(distributor_id: int, start_day: date, end_day: date, cat_names: list[str]) -> pd.DataFrame:
    """
    One row per date.
    Columns:
      Date,
      for each category: "<cat> Qty", "<cat> Rate",
      Total Milk (L), Purchases (₹), Payment (₹), Running Due (₹)
    Rate shown = weighted avg rate for that day+category (amount/qty) if multiple lines exist.
    """
    days = pd.date_range(start=start_day, end=end_day, freq="D").date

    if USE_SERVER_FILTERS:
        dp = sb_fetch_df(
            DISTRIBUTOR_PURCHASES_FILE,
            CSV_SCHEMAS[DISTRIBUTOR_PURCHASES_FILE],
            filters=[
                ("distributor_id", "eq", int(distributor_id)),
                ("date", "gte", str(start_day)),
                ("date", "lte", str(end_day)),
            ],
        )
    else:
        dp = dist_purchases.copy()
    if not dp.empty:
        dp["date"] = _safe_dt(dp["date"]).dt.date
        dp = dp.loc[
            (dp["distributor_id"].astype(int) == int(distributor_id)) &
            (dp["date"] >= start_day) &
            (dp["date"] <= end_day)
        ].copy()

    if not dp.empty:
        dp["qty"] = pd.to_numeric(dp["qty"], errors="coerce").fillna(0.0).astype(float)
        dp["rate"] = pd.to_numeric(dp["rate"], errors="coerce").fillna(0.0).astype(float)
        dp["amount"] = pd.to_numeric(dp.get("amount", 0.0), errors="coerce").fillna(0.0).astype(float)

        # attach category names
        dp = dp.merge(categories[["category_id", "name"]], on="category_id", how="left").rename(columns={"name": "Category"})
        dp["Category"] = dp["Category"].fillna("").astype(str)

    if USE_SERVER_FILTERS:
        pay = sb_fetch_df(
            DISTRIBUTOR_PAYMENTS_FILE,
            CSV_SCHEMAS[DISTRIBUTOR_PAYMENTS_FILE],
            filters=[
                ("distributor_id", "eq", int(distributor_id)),
                ("date", "gte", str(start_day)),
                ("date", "lte", str(end_day)),
            ],
        )
    else:
        pay = dist_payments.copy()
    if not pay.empty:
        pay["date"] = _safe_dt(pay["date"]).dt.date
        pay = pay.loc[
            (pay["distributor_id"].astype(int) == int(distributor_id)) &
            (pay["date"] >= start_day) &
            (pay["date"] <= end_day)
        ].copy()
        pay["amount"] = pd.to_numeric(pay["amount"], errors="coerce").fillna(0.0).astype(float)

    pay_by_day = pay.groupby("date")["amount"].sum().to_dict() if not pay.empty else {}

    opening_due = distributor_balance_before(distributor_id, start_day)
    running = float(opening_due)

    rows = []
    for d in days:
        row = {"Date": str(d)}
        total_milk = 0.0
        day_amt = 0.0

        if dp.empty:
            dp_day = pd.DataFrame(columns=["Category", "qty", "amount"])
        else:
            dp_day = dp.loc[dp["date"] == d].copy()

        for cat in cat_names:
            qcol = f"{cat} Qty"
            rcol = f"{cat} Rate"

            if dp_day.empty:
                qty = 0.0
                amt = 0.0
            else:
                sub = dp_day.loc[dp_day["Category"] == cat].copy()
                qty = float(sub["qty"].sum()) if not sub.empty else 0.0
                amt = float(sub["amount"].sum()) if not sub.empty else 0.0

            if qty > 0:
                # Weighted avg rate
                rate = (amt / qty) if qty > 0 else 0.0
                row[qcol] = qty
                row[rcol] = rate if rate > 0 else "-"
                total_milk += qty
                day_amt += amt
            else:
                row[qcol] = "-"
                row[rcol] = "-"

        pay_amt = float(pay_by_day.get(d, 0.0))
        running = float(running + day_amt - pay_amt)

        row["Total Milk (L)"] = round(float(total_milk), 2)
        row["Purchases (₹)"] = round(float(day_amt), 2)
        row["Payment (₹)"] = round(float(pay_amt), 2)
        row["Running Due (₹)"] = round(float(running), 2)

        rows.append(row)

    return pd.DataFrame(rows)


def distributor_pay_mode_totals(distributor_id: int, start_day: date, end_day: date) -> pd.DataFrame:
    pay = dist_payments.copy()
    if pay.empty:
        return pd.DataFrame(columns=["Mode", "Total (₹)"])

    pay["date"] = _safe_dt(pay["date"]).dt.date
    pay = pay.loc[
        (pay["distributor_id"].astype(int) == int(distributor_id)) &
        (pay["date"] >= start_day) &
        (pay["date"] <= end_day)
    ].copy()
    if pay.empty:
        return pd.DataFrame(columns=["Mode", "Total (₹)"])

    pay["payment_mode"] = pay["payment_mode"].fillna("Cash").astype(str)
    pay["amount"] = pd.to_numeric(pay["amount"], errors="coerce").fillna(0.0).astype(float)

    out = (
        pay.groupby("payment_mode", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
        .rename(columns={"payment_mode": "Mode", "amount": "Total (₹)"})
    )
    return out


def build_distributor_bill_html(
    distributor_row: dict,
    start_day: date,
    end_day: date,
    grid: pd.DataFrame,
    pay_mode_totals: pd.DataFrame,
    cat_names: list[str],
) -> str:
    shop_name = "JYOTIRLING MILK SUPPLIER"

    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def safe_meta(v):
        v = "" if v is None else str(v)
        v = v.strip()
        return v if v else "-"

    def fmt_money(x) -> str:
        try:
            return f"₹{float(x):,.2f}"
        except Exception:
            return "₹0.00"

    def fmt_num(x) -> str:
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "-"

    vendor = safe_meta(distributor_row.get("name"))
    contact = safe_meta(distributor_row.get("contact"))
    address = safe_meta(distributor_row.get("address"))

    df = grid.copy() if grid is not None else pd.DataFrame()

    total_qty_by_cat = {cat: 0.0 for cat in cat_names}
    total_amt = 0.0
    total_pay = 0.0
    closing_due = 0.0

    if not df.empty:
        for cat in cat_names:
            qcol = f"{cat} Qty"
            if qcol in df.columns:
                s = 0.0
                for v in df[qcol].tolist():
                    try:
                        s += float(v)
                    except Exception:
                        pass
                total_qty_by_cat[cat] = float(s)

        if "Purchases (₹)" in df.columns:
            total_amt = float(pd.to_numeric(df["Purchases (₹)"], errors="coerce").fillna(0).sum())
        if "Payment (₹)" in df.columns:
            total_pay = float(pd.to_numeric(df["Payment (₹)"], errors="coerce").fillna(0).sum())
        if "Running Due (₹)" in df.columns and len(df) > 0:
            closing_due = float(pd.to_numeric(df["Running Due (₹)"], errors="coerce").fillna(0).iloc[-1])

    pay_rows_html = ""
    if pay_mode_totals is not None and not pay_mode_totals.empty:
        pm = pay_mode_totals.copy()
        for _, r in pm.iterrows():
            mode = esc(r.get("Mode", "-"))
            amt = fmt_money(r.get("Total (₹)", 0.0))
            pay_rows_html += f"<tr><td>{mode}</td><td style='text-align:right'>{amt}</td></tr>"
    else:
        pay_rows_html = "<tr><td colspan='2' style='text-align:center;color:#666'>No payments in this period</td></tr>"

    th = "<th>Date</th>"
    for cat in cat_names:
        th += f"<th>{esc(cat)} Qty</th><th>{esc(cat)} Rate</th>"
    th += "<th>Total Milk (L)</th><th>Purchases (₹)</th><th>Payment (₹)</th><th>Running Due (₹)</th>"

    body_rows = ""
    for _, r in df.iterrows():
        tds = f"<td>{esc(r.get('Date','-'))}</td>"
        for cat in cat_names:
            qcol = f"{cat} Qty"
            rcol = f"{cat} Rate"
            qv = r.get(qcol, "-")
            rv = r.get(rcol, "-")

            if qv == "-" or qv is None:
                qdisp = "-"
            else:
                try:
                    qdisp = f"{float(qv):.2f}" if float(qv) != 0 else "-"
                except Exception:
                    qdisp = "-"

            if rv == "-" or rv is None:
                rdisp = "-"
            else:
                try:
                    rdisp = f"{float(rv):.2f}" if float(rv) != 0 else "-"
                except Exception:
                    rdisp = "-"

            tds += f"<td style='text-align:right'>{qdisp}</td><td style='text-align:right'>{rdisp}</td>"

        tds += f"<td style='text-align:right'>{fmt_num(r.get('Total Milk (L)', 0.0))}</td>"
        tds += f"<td style='text-align:right'>{fmt_money(r.get('Purchases (₹)', 0.0))}</td>"
        tds += f"<td style='text-align:right'>{fmt_money(r.get('Payment (₹)', 0.0))}</td>"
        tds += f"<td style='text-align:right'>{fmt_money(r.get('Running Due (₹)', 0.0))}</td>"
        body_rows += f"<tr>{tds}</tr>"

    total_row = "<td><b>TOTAL</b></td>"
    for cat in cat_names:
        total_row += f"<td style='text-align:right'><b>{total_qty_by_cat[cat]:.2f}</b></td><td style='text-align:right'><b>-</b></td>"
    total_milk_all = float(sum(total_qty_by_cat.values()))
    total_row += f"<td style='text-align:right'><b>{total_milk_all:.2f}</b></td>"
    total_row += f"<td style='text-align:right'><b>{fmt_money(total_amt)}</b></td>"
    total_row += f"<td style='text-align:right'><b>{fmt_money(total_pay)}</b></td>"
    total_row += f"<td style='text-align:right'><b>{fmt_money(closing_due)}</b></td>"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Distributor Statement</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; color: #111; }}
  .topbar {{ display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }}
  h1 {{ margin: 0; font-size: 28px; letter-spacing: 1px; }}
  .meta {{ border:1px solid #333; padding:12px; border-radius:8px; margin-top:10px; }}
  .meta b {{ display:inline-block; min-width: 140px; }}
  .btns {{ margin: 12px 0 18px 0; }}
  button {{ padding: 8px 14px; border: 1px solid #333; background: #f2f2f2; cursor:pointer; border-radius: 6px; }}
  button:hover {{ background:#e8e8e8; }}
  table {{ width:100%; border-collapse: collapse; margin-top: 10px; }}
  th, td {{ border:1px solid #333; padding: 6px 8px; font-size: 12.5px; }}
  th {{ background: #efefef; }}
  .section-title {{ font-size: 18px; margin-top: 14px; font-weight: 700; }}
  .summarybox {{ border:1px solid #333; padding:12px; border-radius:8px; margin-top:10px; }}
  .sign {{ margin-top: 34px; display:flex; justify-content:space-between; gap:20px; }}
  .sign .line {{ border-top:1px solid #333; width: 260px; margin-top: 36px; }}
  .muted {{ color:#444; font-size: 12px; }}
  @media print {{
    .btns {{ display: none; }}
    body {{ margin: 8mm; }}
    th {{ background: #eee !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  }}
</style>
</head>
<body>

<div class="topbar">
  <div>
    <h1>{esc(shop_name)}</h1>
    <div class="muted">Distributor Statement / Incoming Milk Ledger</div>
  </div>
  <div class="btns">
    <button onclick="window.print()">🖨️ Print</button>
  </div>
</div>

<div class="meta">
  <div><b>Distributor:</b> {esc(vendor)}</div>
  <div><b>Contact:</b> {esc(contact)}</div>
  <div><b>Address:</b> {esc(address)}</div>
  <div><b>Period:</b> {esc(str(start_day))} to {esc(str(end_day))}</div>
</div>

<div class="section-title">Incoming Milk Details</div>
<table>
  <thead><tr>{th}</tr></thead>
  <tbody>
    {body_rows}
    <tr>{total_row}</tr>
  </tbody>
</table>

<div class="section-title">Summary</div>
<div class="summarybox">
  <div><b>Total Purchases:</b> {fmt_money(total_amt)}</div>
  <div><b>Total Payments:</b> {fmt_money(total_pay)}</div>
  <div><b>Closing Due:</b> {fmt_money(closing_due)}</div>
</div>

<div class="section-title">Payment Mode Totals (This Period)</div>
<table style="width: 420px; max-width:100%;">
  <thead><tr><th>Mode</th><th style="text-align:right">Total (₹)</th></tr></thead>
  <tbody>{pay_rows_html}</tbody>
</table>

<div class="sign">
  <div>
    <div class="line"></div>
    <div><b>Distributor Signature</b></div>
  </div>
  <div style="text-align:right;">
    <div class="line"></div>
    <div><b>Proprietor (Verified)</b></div>
    <div class="muted">{esc(shop_name)}</div>
  </div>
</div>

</body>
</html>
"""
    return html


# ================== LOAD DATA ==================
def load_and_migrate_data():
    retailers = safe_read_csv(RETAILERS_FILE, CSV_SCHEMAS[RETAILERS_FILE])
    retailers["retailer_id"] = pd.to_numeric(retailers["retailer_id"], errors="coerce").fillna(0).astype(int)
    retailers["name"] = retailers["name"].fillna("").astype(str)
    retailers["contact"] = retailers.get("contact", "").fillna("").astype(str)
    retailers["address"] = retailers.get("address", "").fillna("").astype(str)
    retailers["zone"] = retailers.get("zone", "Default").fillna("Default").astype(str).apply(_norm_zone)
    retailers["is_active"] = retailers.get("is_active", True).apply(parse_boolish_active)
    retailers = retailers.loc[retailers["retailer_id"] > 0].copy()

    categories = safe_read_csv(CATEGORIES_FILE, CSV_SCHEMAS[CATEGORIES_FILE])
    categories["category_id"] = pd.to_numeric(categories["category_id"], errors="coerce").fillna(0).astype(int)
    categories["name"] = categories["name"].fillna("").astype(str)
    categories["description"] = categories.get("description", "").fillna("").astype(str)
    categories["default_price"] = pd.to_numeric(categories.get("default_price", 0.0), errors="coerce").fillna(0.0).astype(float)
    categories["is_active"] = categories.get("is_active", True).apply(parse_boolish_active)
    categories = categories.loc[categories["category_id"] > 0].copy()

    prices = safe_read_csv(PRICES_FILE, CSV_SCHEMAS[PRICES_FILE])
    prices["price_id"] = pd.to_numeric(prices["price_id"], errors="coerce").fillna(0).astype(int)
    prices["retailer_id"] = pd.to_numeric(prices["retailer_id"], errors="coerce").fillna(GLOBAL_RETAILER_ID).astype(int)
    prices["category_id"] = pd.to_numeric(prices["category_id"], errors="coerce").fillna(0).astype(int)
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce").fillna(0.0).astype(float)
    prices.loc[prices["retailer_id"] == 0, "retailer_id"] = GLOBAL_RETAILER_ID
    eff = pd.to_datetime(prices["effective_date"], errors="coerce").dt.date
    prices["effective_date"] = eff.fillna(date.today()).astype(str)
    prices = prices.loc[prices["price_id"] > 0].copy()

    entries = safe_read_csv(ENTRIES_FILE, CSV_SCHEMAS[ENTRIES_FILE])
    entries["entry_id"] = pd.to_numeric(entries["entry_id"], errors="coerce").fillna(0).astype(int)
    entries["retailer_id"] = pd.to_numeric(entries["retailer_id"], errors="coerce").fillna(0).astype(int)
    entries["category_id"] = pd.to_numeric(entries["category_id"], errors="coerce").fillna(0).astype(int)
    entries["qty"] = pd.to_numeric(entries["qty"], errors="coerce").fillna(0.0).astype(float)
    entries["rate"] = pd.to_numeric(entries["rate"], errors="coerce").fillna(0.0).astype(float)
    entries["amount"] = pd.to_numeric(entries["amount"], errors="coerce").fillna(0.0).astype(float)
    entries = entries.loc[entries["entry_id"] > 0].copy()

    payments = safe_read_csv(PAYMENTS_FILE, CSV_SCHEMAS[PAYMENTS_FILE])
    payments["payment_id"] = pd.to_numeric(payments["payment_id"], errors="coerce").fillna(0).astype(int)
    payments["retailer_id"] = pd.to_numeric(payments["retailer_id"], errors="coerce").fillna(0).astype(int)
    payments["amount"] = pd.to_numeric(payments["amount"], errors="coerce").fillna(0.0).astype(float)
    payments["payment_mode"] = payments["payment_mode"].fillna("Cash").astype(str)
    payments["note"] = payments["note"].fillna("").astype(str)
    payments = payments.loc[payments["payment_id"] > 0].copy()

    distributors = safe_read_csv(DISTRIBUTORS_FILE, CSV_SCHEMAS[DISTRIBUTORS_FILE])
    distributors["distributor_id"] = pd.to_numeric(distributors["distributor_id"], errors="coerce").fillna(0).astype(int)
    distributors["name"] = distributors["name"].fillna("").astype(str)
    distributors["contact"] = distributors.get("contact", "").fillna("").astype(str)
    distributors["address"] = distributors.get("address", "").fillna("").astype(str)
    distributors["is_active"] = distributors.get("is_active", True).apply(parse_boolish_active)
    distributors = distributors.loc[distributors["distributor_id"] > 0].copy()

    dist_purchases = safe_read_csv(DISTRIBUTOR_PURCHASES_FILE, CSV_SCHEMAS[DISTRIBUTOR_PURCHASES_FILE])
    dist_purchases["purchase_id"] = pd.to_numeric(dist_purchases["purchase_id"], errors="coerce").fillna(0).astype(int)
    dist_purchases["distributor_id"] = pd.to_numeric(dist_purchases["distributor_id"], errors="coerce").fillna(0).astype(int)
    dist_purchases["category_id"] = pd.to_numeric(dist_purchases["category_id"], errors="coerce").fillna(0).astype(int)
    dist_purchases["qty"] = pd.to_numeric(dist_purchases["qty"], errors="coerce").fillna(0.0).astype(float)
    dist_purchases["rate"] = pd.to_numeric(dist_purchases["rate"], errors="coerce").fillna(0.0).astype(float)
    dist_purchases["amount"] = pd.to_numeric(dist_purchases["amount"], errors="coerce").fillna(0.0).astype(float)
    dist_purchases = dist_purchases.loc[dist_purchases["purchase_id"] > 0].copy()

    dist_payments = safe_read_csv(DISTRIBUTOR_PAYMENTS_FILE, CSV_SCHEMAS[DISTRIBUTOR_PAYMENTS_FILE])
    dist_payments["payment_id"] = pd.to_numeric(dist_payments["payment_id"], errors="coerce").fillna(0).astype(int)
    dist_payments["distributor_id"] = pd.to_numeric(dist_payments["distributor_id"], errors="coerce").fillna(0).astype(int)
    dist_payments["amount"] = pd.to_numeric(dist_payments["amount"], errors="coerce").fillna(0.0).astype(float)
    dist_payments["payment_mode"] = dist_payments["payment_mode"].fillna("Cash").astype(str)
    dist_payments["note"] = dist_payments["note"].fillna("").astype(str)
    dist_payments = dist_payments.loc[dist_payments["payment_id"] > 0].copy()

    # NEW: Distributor ↔ Category mapping table
    dist_cat_map = safe_read_csv(DISTRIBUTOR_CATEGORY_MAP_FILE, CSV_SCHEMAS[DISTRIBUTOR_CATEGORY_MAP_FILE])
    dist_cat_map["map_id"] = pd.to_numeric(dist_cat_map["map_id"], errors="coerce").fillna(0).astype(int)
    dist_cat_map["distributor_id"] = pd.to_numeric(dist_cat_map["distributor_id"], errors="coerce").fillna(0).astype(int)
    dist_cat_map["category_id"] = pd.to_numeric(dist_cat_map["category_id"], errors="coerce").fillna(0).astype(int)
    dist_cat_map["is_active"] = dist_cat_map.get("is_active", True).apply(parse_boolish_active)
    dist_cat_map = dist_cat_map.loc[dist_cat_map["map_id"] > 0].copy()

    wastage = safe_read_csv(WASTAGE_FILE, CSV_SCHEMAS[WASTAGE_FILE])
    wastage["wastage_id"] = pd.to_numeric(wastage["wastage_id"], errors="coerce").fillna(0).astype(int)
    wastage["category_id"] = pd.to_numeric(wastage["category_id"], errors="coerce").fillna(0).astype(int)
    wastage["qty"] = pd.to_numeric(wastage["qty"], errors="coerce").fillna(0.0).astype(float)
    wastage["estimated_loss"] = pd.to_numeric(wastage["estimated_loss"], errors="coerce").fillna(0.0).astype(float)
    wastage["reason"] = wastage["reason"].fillna("").astype(str)
    wastage = wastage.loc[wastage["wastage_id"] > 0].copy()

    expenses = safe_read_csv(EXPENSES_FILE, CSV_SCHEMAS[EXPENSES_FILE])
    expenses["expense_id"] = pd.to_numeric(expenses["expense_id"], errors="coerce").fillna(0).astype(int)
    expenses["amount"] = pd.to_numeric(expenses["amount"], errors="coerce").fillna(0.0).astype(float)
    expenses["paid"] = expenses.get("paid", False).apply(parse_boolish_paid)
    expenses = expenses.loc[expenses["expense_id"] > 0].copy()

    return retailers, categories, prices, entries, payments, distributors, dist_purchases, dist_payments, dist_cat_map, wastage, expenses

st.session_state.setdefault("data_version", 0)
retailers, categories, prices, entries, payments, distributors, dist_purchases, dist_payments, dist_cat_map, wastage, expenses = load_and_migrate_data_cached(st.session_state["data_version"])
# ================== ZONE HELPERS ==================
def get_all_zones() -> list[str]:
    if retailers.empty:
        return ["Default"]
    rz = retailers.copy()
    rz["zone"] = rz["zone"].apply(_norm_zone)
    zones = sorted(set(rz["zone"].tolist()))
    return zones if zones else ["Default"]

def get_zone_retailer_ids(selected_zone: str) -> list[int]:
    if retailers.empty:
        return []
    if selected_zone == "All Zones":
        return retailers["retailer_id"].astype(int).tolist()
    z = _norm_zone(selected_zone)
    rz = retailers.copy()
    rz["zone"] = rz["zone"].apply(_norm_zone)
    return rz.loc[rz["zone"] == z, "retailer_id"].astype(int).tolist()

def filter_by_zone(df: pd.DataFrame, retailer_id_col: str, selected_zone: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if selected_zone == "All Zones":
        return df.copy()
    ids = set(get_zone_retailer_ids(selected_zone))
    if not ids:
        return df.iloc[0:0].copy()
    return df.loc[df[retailer_id_col].astype(int).isin(ids)].copy()

# ================== PRICING HELPERS ==================
def _get_effective_price(price_df: pd.DataFrame, entry_dt: pd.Timestamp) -> float | None:
    if price_df.empty:
        return None
    df = price_df.copy()
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce").dt.normalize()
    entry_dt = pd.to_datetime(entry_dt, errors="coerce").normalize()
    df = df.loc[df["effective_date"].notna()]
    df = df.loc[df["effective_date"] <= entry_dt]
    if df.empty:
        return None
    df = df.sort_values("effective_date", ascending=False)
    return float(df.iloc[0]["price"])

def get_price_for_date(retailer_id: int, category_id: int, entry_date) -> float | None:
    entry_dt = pd.to_datetime(entry_date)

    if not prices.empty:
        rs = prices.loc[(prices["retailer_id"] == int(retailer_id)) & (prices["category_id"] == int(category_id))].copy()
        p = _get_effective_price(rs, entry_dt)
        if p is not None and p > 0:
            return float(p)

        gs = prices.loc[(prices["retailer_id"] == GLOBAL_RETAILER_ID) & (prices["category_id"] == int(category_id))].copy()
        p = _get_effective_price(gs, entry_dt)
        if p is not None and p > 0:
            return float(p)

    if not categories.empty and "default_price" in categories.columns:
        cat_row = categories.loc[categories["category_id"] == int(category_id)]
        if not cat_row.empty:
            default_price = cat_row.iloc[0].get("default_price", 0.0)
            if pd.notna(default_price) and float(default_price) > 0:
                return float(default_price)

    return None

# ================== ACCOUNTING HELPERS ==================
def get_retailer_balance(retailer_id: int) -> float:
    total_sales = entries.loc[entries["retailer_id"] == retailer_id, "amount"].sum() if not entries.empty else 0.0
    total_payments = payments.loc[payments["retailer_id"] == retailer_id, "amount"].sum() if not payments.empty else 0.0
    return float(total_sales - total_payments)

def retailer_ledger_as_of(retailer_id: int, as_of_day: date) -> float:
    if entries.empty and payments.empty:
        return 0.0
    e = entries.copy()
    p = payments.copy()
    if not e.empty:
        e["date"] = pd.to_datetime(e["date"], errors="coerce").dt.date
    if not p.empty:
        p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.date

    sales = e.loc[(e["retailer_id"] == int(retailer_id)) & (e["date"] <= as_of_day), "amount"].sum() if not e.empty else 0.0
    paid = p.loc[(p["retailer_id"] == int(retailer_id)) & (p["date"] <= as_of_day), "amount"].sum() if not p.empty else 0.0
    return float(sales - paid)

def is_retailer_referenced(retailer_id: int) -> bool:
    rid = int(retailer_id)
    if (not entries.empty) and (rid in set(entries["retailer_id"].astype(int))):
        return True
    if (not payments.empty) and (rid in set(payments["retailer_id"].astype(int))):
        return True
    if (not prices.empty) and (rid in set(prices["retailer_id"].astype(int))):
        return True
    return False

def is_category_referenced(category_id: int) -> bool:
    cid = int(category_id)
    if (not entries.empty) and (cid in set(entries["category_id"].astype(int))):
        return True
    if (not prices.empty) and (cid in set(prices["category_id"].astype(int))):
        return True
    if (not dist_purchases.empty) and (cid in set(dist_purchases["category_id"].astype(int))):
        return True
    if (not wastage.empty) and (cid in set(wastage["category_id"].astype(int))):
        return True
    return False

def is_distributor_referenced(distributor_id: int) -> bool:
    did = int(distributor_id)
    if (not dist_purchases.empty) and (did in set(dist_purchases["distributor_id"].astype(int))):
        return True
    if (not dist_payments.empty) and (did in set(dist_payments["distributor_id"].astype(int))):
        return True
    return False

# ================== DAILY SHEET HELPERS ==================

def _retailer_ids_for_zone(zone: str) -> list[int]:
    """Returns retailer_ids for the given zone (or all if 'All Zones')."""
    try:
        rz = retailers_active.copy() if 'retailers_active' in globals() else retailers.copy()
        if rz is None or rz.empty:
            return []
        rz = rz.copy()
        rz["zone"] = rz.get("zone", "").astype(str).apply(_norm_zone)
        if zone == "All Zones":
            return rz["retailer_id"].astype(int).tolist()
        return rz.loc[rz["zone"] == _norm_zone(zone), "retailer_id"].astype(int).tolist()
    except Exception:
        return []

def _day_entries_for_zone(day: date, zone: str) -> pd.DataFrame:
    """Always fetches fresh transactional data for the selected day (+ zone filter)."""
    if USE_SERVER_FILTERS:
        rids = _retailer_ids_for_zone(zone)
        if not rids:
            return pd.DataFrame(columns=CSV_SCHEMAS[ENTRIES_FILE])
        df = sb_fetch_df(
            ENTRIES_FILE,
            CSV_SCHEMAS[ENTRIES_FILE],
            filters=[("date", "eq", str(day)), ("retailer_id", "in", rids)],
        )
        return df
    # CSV / local fallback
    df = entries.copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.loc[df["date"] == day].copy()
    df = filter_by_zone(df, "retailer_id", zone)
    return df



def _day_payments_for_zone(day: date, zone: str) -> pd.DataFrame:
    """Always fetches fresh transactional data for the selected day (+ zone filter)."""
    if USE_SERVER_FILTERS:
        rids = _retailer_ids_for_zone(zone)
        if not rids:
            return pd.DataFrame(columns=CSV_SCHEMAS[PAYMENTS_FILE])
        df = sb_fetch_df(
            PAYMENTS_FILE,
            CSV_SCHEMAS[PAYMENTS_FILE],
            filters=[("date", "eq", str(day)), ("retailer_id", "in", rids)],
        )
        return df
    df = payments.copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.loc[df["date"] == day].copy()
    df = filter_by_zone(df, "retailer_id", zone)
    return df

PAYMENT_MODES = ["Cash", "UPI", "Bank", "Cheque", "Other"]


def build_daily_posting_grid(day: date, zone: str, retailers_active: pd.DataFrame, categories_active: pd.DataFrame):
    """Builds the Excel-like daily grid using stable keys (retailer_id), with fresh day data."""
    rz = retailers_active.copy()
    rz["zone"] = rz.get("zone", "").astype(str).apply(_norm_zone)
    if zone != "All Zones":
        rz = rz.loc[rz["zone"] == _norm_zone(zone)].copy()

    cats = categories_active.copy()
    cat_list = cats["name"].tolist()

    # Fetch day data (fresh in Supabase mode)
    day_p = _day_payments_for_zone(day, zone)
    day_e = _day_entries_for_zone(day, zone)

    # Payments pivot by retailer_id
    pay_by_mode = pd.DataFrame()
    if not day_p.empty:
        tmp = day_p.copy()
        tmp["retailer_id"] = pd.to_numeric(tmp["retailer_id"], errors="coerce").fillna(0).astype(int)
        tmp["payment_mode"] = tmp["payment_mode"].fillna("Cash").astype(str).str.strip().str.title()
        tmp["amount"] = pd.to_numeric(tmp["amount"], errors="coerce").fillna(0.0)
        pay_by_mode = (
            tmp.pivot_table(
                index="retailer_id",
                columns="payment_mode",
                values="amount",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(columns=PAYMENT_MODES, fill_value=0.0)
        )

    # Entries pivot by retailer_id x category_name
    pivot_qty = pd.DataFrame()
    if not day_e.empty:
        e = day_e.copy()
        e["retailer_id"] = pd.to_numeric(e["retailer_id"], errors="coerce").fillna(0).astype(int)
        e["category_id"] = pd.to_numeric(e["category_id"], errors="coerce").fillna(0).astype(int)
        e["qty"] = pd.to_numeric(e["qty"], errors="coerce").fillna(0.0)

        cat_map = categories_active[["category_id", "name"]].copy()
        cat_map["category_id"] = pd.to_numeric(cat_map["category_id"], errors="coerce").fillna(0).astype(int)
        e = e.merge(cat_map, on="category_id", how="left").rename(columns={"name": "CategoryName"})
        e["CategoryName"] = e["CategoryName"].fillna("").astype(str)

        pivot_qty = pd.pivot_table(
            e,
            index="retailer_id",
            columns="CategoryName",
            values="qty",
            aggfunc="sum",
            fill_value=0.0,
        )

    # Build grid rows
    rows = []
    for _, r in rz.iterrows():
        rid = int(r["retailer_id"])
        retailer_name = str(r.get("name", ""))

        row = {"ID": rid, "Retailer": retailer_name}

        # payment columns
        for m in PAYMENT_MODES:
            col = f"{m} ₹"
            val = 0.0
            if (not pay_by_mode.empty) and (rid in pay_by_mode.index) and (m in pay_by_mode.columns):
                val = float(pay_by_mode.loc[rid, m])
            row[col] = float(val)

        # qty columns
        for c in cat_list:
            qty = 0.0
            if (not pivot_qty.empty) and (rid in pivot_qty.index) and (c in pivot_qty.columns):
                qty = float(pivot_qty.loc[rid, c])
            row[c] = float(qty)

        rows.append(row)

    grid = pd.DataFrame(rows)
    return grid, cat_list

def compute_today_sales_amount_for_row(rid: int, day: date, row: pd.Series, cat_name_list: list[str], categories_active: pd.DataFrame) -> float:
    amt = 0.0
    for cat_name in cat_name_list:
        qty = float(row.get(cat_name, 0.0) or 0.0)
        if qty <= 0:
            continue
        cid = int(categories_active.loc[categories_active["name"] == cat_name, "category_id"].iloc[0])
        rate = get_price_for_date(rid, cid, day)
        if rate is None or rate <= 0:
            raise ValueError(f"Price missing for Retailer ID {rid} / {cat_name} on {day}")
        amt += qty * float(rate)
    return float(amt)

def zone_category_pivot_for_day(day: date) -> pd.DataFrame:
    df = entries.copy()
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.loc[df["date"] == day].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.merge(retailers[["retailer_id", "zone"]], on="retailer_id", how="left")
    df["zone"] = df["zone"].apply(_norm_zone)

    df = df.merge(categories[["category_id", "name"]], on="category_id", how="left").rename(columns={"name": "Category"})
    df["Category"] = df["Category"].fillna("").astype(str)

    pivot = pd.pivot_table(
        df,
        index="zone",
        columns="Category",
        values="qty",
        aggfunc="sum",
        fill_value=0.0,
    )

    pivot = pivot.sort_index()
    pivot["TOTAL (L)"] = pivot.sum(axis=1)
    pivot = pivot.reset_index().rename(columns={"zone": "Zone"})
    return pivot

# ================== BILL / STATEMENT HELPERS ==================
def retailer_balance_before(retailer_id: int, start_day: date) -> float:
    if USE_SERVER_FILTERS:
        e = sb_fetch_df(
            ENTRIES_FILE,
            CSV_SCHEMAS[ENTRIES_FILE],
            filters=[
                ("retailer_id", "eq", int(retailer_id)),
                ("date", "lt", str(start_day)),
            ],
        )
        p = sb_fetch_df(
            PAYMENTS_FILE,
            CSV_SCHEMAS[PAYMENTS_FILE],
            filters=[
                ("retailer_id", "eq", int(retailer_id)),
                ("date", "lt", str(start_day)),
            ],
        )
    else:
        e = entries.copy()
        p = payments.copy()
    sales = 0.0
    paid = 0.0

    if not e.empty:
        e["date"] = _safe_dt(e["date"]).dt.date
        e = e.loc[(e["retailer_id"].astype(int) == int(retailer_id)) & (e["date"] < start_day)]
        sales = float(e["amount"].sum()) if not e.empty else 0.0

    if not p.empty:
        p["date"] = _safe_dt(p["date"]).dt.date
        p = p.loc[(p["retailer_id"].astype(int) == int(retailer_id)) & (p["date"] < start_day)]
        paid = float(p["amount"].sum()) if not p.empty else 0.0

    return float(sales - paid)

def _rate_from_entries_or_price(retailer_id: int, cid: int, d: date, e_day_cat: pd.DataFrame) -> float | None:
    # Prefer stored entry rates (history-safe). If missing/zero, fallback to price table.
    if e_day_cat is not None and not e_day_cat.empty:
        rates = pd.to_numeric(e_day_cat["rate"], errors="coerce").fillna(0.0)
        rates = rates[rates > 0]
        if not rates.empty:
            # if multiple rates, show average for display but sales uses amount anyway
            return float(rates.mean())
    return get_price_for_date(int(retailer_id), int(cid), d)

def build_bill_daily_grid(retailer_id: int, start_day: date, end_day: date, cat_names: list[str]) -> pd.DataFrame:
    """
    SINGLE authoritative bill grid. No duplicates.
    Logic:
      opening_due = balance before start_day
      daily_sales = sum(qty * rate) using stored entries (amount) primarily
      payments
      running_due
    Columns:
      Date,
      per category: "<cat> Qty", "<cat> Rate",
      Total Milk (L), Sales (₹), Payment (₹), Running Due (₹)
    """
    days = pd.date_range(start=start_day, end=end_day, freq="D").date

    if USE_SERVER_FILTERS:
        e = sb_fetch_df(
            ENTRIES_FILE,
            CSV_SCHEMAS[ENTRIES_FILE],
            filters=[
                ("retailer_id", "eq", int(retailer_id)),
                ("date", "gte", str(start_day)),
                ("date", "lte", str(end_day)),
            ],
        )
    else:
        e = entries.copy()
    if not e.empty:
        e["date"] = _safe_dt(e["date"]).dt.date
        e = e.loc[
            (e["retailer_id"].astype(int) == int(retailer_id)) &
            (e["date"] >= start_day) &
            (e["date"] <= end_day)
        ].copy()

    if not e.empty:
        e = e.merge(categories[["category_id", "name"]], on="category_id", how="left").rename(columns={"name": "Category"})
        e["Category"] = e["Category"].fillna("").astype(str)

    if USE_SERVER_FILTERS:
        p = sb_fetch_df(
            PAYMENTS_FILE,
            CSV_SCHEMAS[PAYMENTS_FILE],
            filters=[
                ("retailer_id", "eq", int(retailer_id)),
                ("date", "gte", str(start_day)),
                ("date", "lte", str(end_day)),
            ],
        )
    else:
        p = payments.copy()
    if not p.empty:
        p["date"] = _safe_dt(p["date"]).dt.date
        p = p.loc[
            (p["retailer_id"].astype(int) == int(retailer_id)) &
            (p["date"] >= start_day) &
            (p["date"] <= end_day)
        ].copy()

    pay_by_day = p.groupby("date")["amount"].sum().to_dict() if not p.empty else {}

    opening_due = retailer_balance_before(int(retailer_id), start_day)
    running = float(opening_due)

    rows = []
    for d in days:
        row = {"Date": str(d)}

        e_day = e.loc[e["date"] == d].copy() if (e is not None and not e.empty) else pd.DataFrame(columns=["Category", "qty", "rate", "amount"])

        total_milk = 0.0
        day_sales = 0.0

        for cat in cat_names:
            qcol = f"{cat} Qty"
            rcol = f"{cat} Rate"

            e_day_cat = e_day.loc[e_day["Category"] == str(cat)].copy() if not e_day.empty else pd.DataFrame()
            qty = float(pd.to_numeric(e_day_cat["qty"], errors="coerce").fillna(0.0).sum()) if not e_day_cat.empty else 0.0

            if qty > 0:
                # Prefer stored amounts (history-safe)
                amt = float(pd.to_numeric(e_day_cat["amount"], errors="coerce").fillna(0.0).sum()) if not e_day_cat.empty else 0.0
                # If amount not reliable, compute qty*rate
                if amt <= 0:
                    cat_row = categories.loc[categories["name"].astype(str) == str(cat)]
                    cid = int(cat_row.iloc[0]["category_id"]) if not cat_row.empty else None
                    rate = _rate_from_entries_or_price(int(retailer_id), int(cid), d, e_day_cat) if cid is not None else None
                    if rate is not None and float(rate) > 0:
                        amt = qty * float(rate)
                day_sales += float(amt)

                # Rate display
                cat_row = categories.loc[categories["name"].astype(str) == str(cat)]
                cid = int(cat_row.iloc[0]["category_id"]) if not cat_row.empty else None
                rate_disp = _rate_from_entries_or_price(int(retailer_id), int(cid), d, e_day_cat) if cid is not None else None

                row[qcol] = qty
                row[rcol] = float(rate_disp) if (rate_disp is not None and float(rate_disp) > 0) else "-"
            else:
                row[qcol] = "-"
                row[rcol] = "-"

            total_milk += qty

        pay = float(pay_by_day.get(d, 0.0))
        running = float(running + day_sales - pay)

        row["Total Milk (L)"] = round(float(total_milk), 2)
        row["Sales (₹)"] = round(float(day_sales), 2)
        row["Payment (₹)"] = round(float(pay), 2)
        row["Running Due (₹)"] = round(float(running), 2)

        rows.append(row)

    return pd.DataFrame(rows)

def build_bill_html(
    retailer_row: dict,
    start_day: date,
    end_day: date,
    grid: pd.DataFrame,
    pay_mode_totals: pd.DataFrame,
    cat_names: list[str],
    opening_due: float,
) -> str:
    shop_name = "JYOTIRLING MILK SUPPLIER"
    cust = display_or_dash(retailer_row.get("name"))
    zone = display_or_dash(retailer_row.get("zone"))
    contact = display_or_dash(retailer_row.get("contact"))
    address = display_or_dash(retailer_row.get("address"))

    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def fmt_money(x) -> str:
        return _fmt_money(x)

    def fmt_num(x) -> str:
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "–"

    df = grid.copy() if grid is not None else pd.DataFrame()

    total_qty_by_cat = {cat: 0.0 for cat in cat_names}
    total_sales = 0.0
    total_pay = 0.0
    closing_due = float(opening_due)

    if df is not None and not df.empty:
        # Totals by category qty
        for cat in cat_names:
            qcol = f"{cat} Qty"
            if qcol in df.columns:
                s = 0.0
                for v in df[qcol].tolist():
                    try:
                        s += float(v)
                    except Exception:
                        pass
                total_qty_by_cat[cat] = float(s)

        # Period totals
        if "Sales (₹)" in df.columns:
            total_sales = float(pd.to_numeric(df["Sales (₹)"], errors="coerce").fillna(0).sum())
        if "Payment (₹)" in df.columns:
            total_pay = float(pd.to_numeric(df["Payment (₹)"], errors="coerce").fillna(0).sum())

        # Closing due = last running due (if present)
        if "Running Due (₹)" in df.columns and not df.empty:
            closing_due = float(
                pd.to_numeric(df["Running Due (₹)"], errors="coerce")
                .fillna(opening_due)
                .iloc[-1]
            )

    # Payment mode totals rows
    pay_rows_html = ""
    if pay_mode_totals is not None and not pay_mode_totals.empty:
        pm = pay_mode_totals.copy()
        for _, r in pm.iterrows():
            mode = esc(r.get("Mode", "-"))
            amt = fmt_money(r.get("Total (₹)", 0.0))
            pay_rows_html += f"<tr><td>{mode}</td><td style='text-align:right'>{amt}</td></tr>"
    else:
        pay_rows_html = "<tr><td colspan='2' style='text-align:center;color:#666'>No payments in this period</td></tr>"

    # ========= TABLE HEADER (Rate removed) =========
    th = "<th>Date</th>"
    for cat in cat_names:
        th += f"<th>{esc(cat)} Qty</th>"
    th += "<th>Total Milk (L)</th><th>Sales (₹)</th><th>Payment (₹)</th><th>Running Due (₹)</th>"

    # ========= TABLE BODY (Rate removed) =========
    body_rows = ""
    for _, r in df.iterrows() if df is not None else []:
        tds = f"<td>{esc(r.get('Date','-'))}</td>"

        for cat in cat_names:
            qcol = f"{cat} Qty"
            qv = r.get(qcol, "-")

            if qv == "-" or qv is None:
                qdisp = "-"
            else:
                try:
                    fq = float(qv)
                    qdisp = "-" if fq == 0 else f"{fq:.2f}"
                except Exception:
                    qdisp = "-"

            tds += f"<td style='text-align:right'>{qdisp}</td>"

        tds += f"<td style='text-align:right'>{fmt_num(r.get('Total Milk (L)', 0.0))}</td>"
        tds += f"<td style='text-align:right'>{fmt_money(r.get('Sales (₹)', 0.0))}</td>"
        tds += f"<td style='text-align:right'>{fmt_money(r.get('Payment (₹)', 0.0))}</td>"
        tds += f"<td style='text-align:right'>{fmt_money(r.get('Running Due (₹)', 0.0))}</td>"
        body_rows += f"<tr>{tds}</tr>"

    # ========= TOTAL ROW (Rate removed) =========
    total_row = "<td><b>TOTAL</b></td>"
    for cat in cat_names:
        total_row += f"<td style='text-align:right'><b>{total_qty_by_cat[cat]:.2f}</b></td>"

    total_milk_all = float(sum(total_qty_by_cat.values()))
    total_row += f"<td style='text-align:right'><b>{total_milk_all:.2f}</b></td>"
    total_row += f"<td style='text-align:right'><b>{fmt_money(total_sales)}</b></td>"
    total_row += f"<td style='text-align:right'><b>{fmt_money(total_pay)}</b></td>"
    total_row += f"<td style='text-align:right'><b>{fmt_money(closing_due)}</b></td>"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Milk Bill</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; color: #111; }}
  .topbar {{ display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }}
  h1 {{ margin: 0; font-size: 28px; letter-spacing: 1px; }}
  .meta {{ border:1px solid #333; padding:12px; border-radius:8px; margin-top:10px; }}
  .meta b {{ display:inline-block; min-width: 120px; }}
  .btns {{ margin: 12px 0 18px 0; }}
  button {{ padding: 8px 14px; border: 1px solid #333; background: #f2f2f2; cursor:pointer; border-radius: 6px; }}
  button:hover {{ background:#e8e8e8; }}
  table {{ width:100%; border-collapse: collapse; margin-top: 10px; }}
  th, td {{ border:1px solid #333; padding: 6px 8px; font-size: 12.5px; }}
  th {{ background: #efefef; }}
  .section-title {{ font-size: 18px; margin-top: 14px; font-weight: 700; }}
  .summarybox {{ border:1px solid #333; padding:12px; border-radius:8px; margin-top:10px; }}
  .sign {{ margin-top: 34px; display:flex; justify-content:space-between; gap:20px; }}
  .sign .line {{ border-top:1px solid #333; width: 260px; margin-top: 36px; }}
  .muted {{ color:#444; font-size: 12px; }}
  @media print {{
    .btns {{ display: none; }}
    body {{ margin: 8mm; }}
    th {{ background: #eee !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  }}
</style>
</head>
<body>

<div class="topbar">
  <div>
    <h1>{esc(shop_name)}</h1>
    <div class="muted">Professional Statement / Bill</div>
  </div>
  <div class="btns">
    <button onclick="window.print()">🖨️ Print</button>
  </div>
</div>

<div class="meta">
  <div><b>Customer:</b> {esc(cust)}</div>
  <div><b>Zone:</b> {esc(zone)}</div>
  <div><b>Contact:</b> {esc(contact)}</div>
  <div><b>Address:</b> {esc(address)}</div>
  <div><b>Period:</b> {esc(str(start_day))} to {esc(str(end_day))}</div>
</div>

<div class="section-title">Summary</div>
<div class="summarybox">
  <div><b>Opening Due:</b> {fmt_money(opening_due)}</div>
  <div><b>Total Sales:</b> {fmt_money(total_sales)}</div>
  <div><b>Total Payments:</b> {fmt_money(total_pay)}</div>
  <div><b>Closing Due:</b> {fmt_money(closing_due)}</div>
</div>

<div class="section-title">Bill Details</div>
<table>
  <thead><tr>{th}</tr></thead>
  <tbody>
    {body_rows}
    <tr>{total_row}</tr>
  </tbody>
</table>

<div class="section-title">Payment Mode Totals (This Period)</div>
<table style="width: 420px; max-width:100%;">
  <thead><tr><th>Mode</th><th style="text-align:right">Total (₹)</th></tr></thead>
  <tbody>{pay_rows_html}</tbody>
</table>

<div class="sign">
  <div>
    <div class="line"></div>
    <div><b>Customer Signature</b></div>
  </div>
  <div style="text-align:right;">
    <div class="line"></div>
    <div><b>Proprietor (Verified)</b></div>
    <div class="muted">{esc(shop_name)}</div>
  </div>
</div>

</body>
</html>
"""
    return html
# PDF generation intentionally disabled.
# HTML is the single source of truth for printing.

def bill_pdf_bytes_from_html(html: str):
    return None

# ================== SIDEBAR: ZONE CONTEXT ==================
zones = get_all_zones()
selected_zone = st.sidebar.selectbox("Zone Context", ["All Zones"] + zones, index=0, key="sidebar_zone")

retailers_active = retailers.loc[retailers.get("is_active", True).apply(parse_boolish_active)].copy() if not retailers.empty else retailers.copy()
categories_active = categories.loc[categories.get("is_active", True).apply(parse_boolish_active)].copy() if not categories.empty else categories.copy()

entries_z = filter_by_zone(entries.copy(), "retailer_id", selected_zone) if not entries.empty else pd.DataFrame(columns=entries.columns)
payments_z = filter_by_zone(payments.copy(), "retailer_id", selected_zone) if not payments.empty else pd.DataFrame(columns=payments.columns)

# ================== UI ==================
st.title("🥛 JYOTIRLING MILK SUPPLIER")
st.markdown(f"""
<div style="
  display:flex; align-items:center; justify-content:space-between;
  padding:18px 20px; margin-bottom:14px;
  background: rgba(255,255,255,0.72);
  border:1px solid rgba(15,23,42,0.10);
  border-radius:18px;
  box-shadow: 0 10px 30px rgba(2,6,23,0.08);
  backdrop-filter: blur(10px);
">
  <div>
    <div style="font-size:28px; font-weight:950; letter-spacing:-0.02em;">🥛 JYOTIRLING MILK SUPPLIER</div>
    <div style="color:#64748B; font-weight:750; margin-top:2px;">Milk Accounting Pro • Clean ledgers • Fast billing</div>
  </div>
  <div style="text-align:right;">
    <div style="color:#64748B; font-weight:800; font-size:12px;">Today</div>
    <div style="font-weight:950; font-size:16px;">{date.today().strftime("%d %b %Y")}</div>
  </div>
</div>
""", unsafe_allow_html=True)



# ---- Production safety: always allow manual refresh ----
if st.sidebar.button("🔄 Refresh (force latest data)", use_container_width=True):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
    st.rerun()

menu = st.sidebar.radio(
    "📋 Navigation",
    [
        "📊 Dashboard",
        "📝 Daily Posting Sheet (Excel)",
        "📅 Date + Zone View",
        "📍 Zone-wise Summary",
        "✏️ Edit (Single Entry)",
        "🥛 Milk Categories",
        "🏪 Retailers",
        "💰 Price Management",
        "📒 Ledger",
        "🔍 Filters & Reports",
        "🚚 Distributors",
        "📦 Milk Purchases",
        "💸 Distributor Payments",
        "📒 Distributor Ledger",
        "🧾 Distributor Bill",
        "🗑️ Milk Wastage",
        "💼 Expenses",
        "🧾 Generate Bill",
        "🛡️ Data Health & Backup",
    ],
)
# ================== DASHBOARD ==================
if menu == "📊 Dashboard":
    st.header(f"📊 Business Overview — {selected_zone}")

    col1, col2, col3, col4 = st.columns(4)
    total_milk = entries_z["qty"].sum() if not entries_z.empty else 0.0
    total_sales = entries_z["amount"].sum() if not entries_z.empty else 0.0
    total_payments = payments_z["amount"].sum() if not payments_z.empty else 0.0
    outstanding = total_sales - total_payments

    with col1:
        st.metric("Total Milk Sold", f"{float(total_milk):.2f} L", delta="Lifetime")
    with col2:
        st.metric("Total Sales", f"₹{float(total_sales):.2f}")
    with col3:
        st.metric("Total Payments", f"₹{float(total_payments):.2f}")
    with col4:
        st.metric("Outstanding", f"₹{float(outstanding):.2f}", delta=f"₹{float(outstanding):.2f}" if outstanding > 0 else "Settled")

    st.divider()

    st.subheader("🧭 Zones Overview (All Zones Together)")
    if retailers.empty:
        st.info("Add retailers and set zones to see zone overview.")
    else:
        zone_rows = []
        for z in zones:
            rz_ids = get_zone_retailer_ids(z)
            ez = entries.loc[entries["retailer_id"].isin(rz_ids)].copy() if not entries.empty else pd.DataFrame()
            pz = payments.loc[payments["retailer_id"].isin(rz_ids)].copy() if not payments.empty else pd.DataFrame()
            z_sales = ez["amount"].sum() if not ez.empty else 0.0
            z_paid = pz["amount"].sum() if not pz.empty else 0.0
            z_qty = ez["qty"].sum() if not ez.empty else 0.0
            zone_rows.append(
                {"Zone": z, "Milk Sold (L)": float(z_qty), "Sales (₹)": float(z_sales), "Paid (₹)": float(z_paid), "Outstanding (₹)": float(z_sales - z_paid)}
            )
        zone_df = pd.DataFrame(zone_rows).sort_values("Sales (₹)", ascending=False)
        st.dataframe(
            zone_df.style.format(
                {"Milk Sold (L)": "{:.2f}", "Sales (₹)": "₹{:.2f}", "Paid (₹)": "₹{:.2f}", "Outstanding (₹)": "₹{:.2f}"}
            ),
            width="stretch",
        )
        if not zone_df.empty:
            fig = px.bar(zone_df, x="Zone", y="Sales (₹)")
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("📅 Daily Business Overview")

    day = st.date_input("Select Day", value=date.today(), key="daily_overview_day")

    dp = dist_purchases.copy()
    if not dp.empty:
        dp["date"] = pd.to_datetime(dp["date"], errors="coerce")
        dp_day = dp.loc[dp["date"].dt.date == day].copy()
        purchased_qty = float(dp_day["qty"].sum()) if not dp_day.empty else 0.0
        purchased_amt = float(dp_day["amount"].sum()) if not dp_day.empty else 0.0
    else:
        purchased_qty, purchased_amt = 0.0, 0.0

    ez = entries_z.copy()
    if not ez.empty:
        ez["date"] = pd.to_datetime(ez["date"], errors="coerce")
        ez_day = ez.loc[ez["date"].dt.date == day].copy()
        sold_qty = float(ez_day["qty"].sum()) if not ez_day.empty else 0.0
        sold_amt = float(ez_day["amount"].sum()) if not ez_day.empty else 0.0
    else:
        sold_qty, sold_amt = 0.0, 0.0

    wz = wastage.copy()
    if not wz.empty:
        wz["date"] = pd.to_datetime(wz["date"], errors="coerce")
        wz_day = wz.loc[wz["date"].dt.date == day].copy()
        waste_qty = float(wz_day["qty"].sum()) if not wz_day.empty else 0.0
        waste_loss = float(wz_day["estimated_loss"].sum()) if not wz_day.empty else 0.0
    else:
        waste_qty, waste_loss = 0.0, 0.0

    net_movement = purchased_qty - sold_qty - waste_qty

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Purchased (L)", f"{purchased_qty:.2f}")
    c2.metric("Purchase (₹)", f"₹{purchased_amt:.2f}")
    c3.metric("Sold (L)", f"{sold_qty:.2f}")
    c4.metric("Sales (₹)", f"₹{sold_amt:.2f}")
    c5.metric("Wastage (L)", f"{waste_qty:.2f}")
    c6.metric("Net Movement (L)", f"{net_movement:.2f}")

    if net_movement < 0:
        st.warning("Net Movement is negative. Purchases might be missing for the day, or you sold from opening stock (not tracked).")

    st.caption("Sales are zone-filtered. Purchases/wastage are not zone-filtered in current model.")

elif menu == "📝 Daily Posting Sheet (Excel)":
    st.header("📝 Daily Posting Sheet — Retailers + Distributors + Wastage (Single Save)")



    if retailers_active.empty or categories_active.empty:
        st.warning("⚠️ Please add active retailers and categories first")
        st.stop()

    posting_date = st.date_input("Posting Date", value=date.today(), key="posting_date")
    zone_choices = ["All Zones"] + get_all_zones()
    default_idx = zone_choices.index(selected_zone) if selected_zone in zone_choices else 0
    posting_zone = st.selectbox("Posting Zone", zone_choices, index=default_idx, key="posting_zone")

    # ---------------- RETAILER GRID ----------------
    grid_df, cat_list = build_daily_posting_grid(posting_date, posting_zone, retailers_active, categories_active)
    if grid_df.empty:
        st.warning("No retailers found for selected zone.")
        st.stop()
    
    # ---- Column order for Daily Entry: ID, Retailer, Categories, Payments ----
    cat_cols = [c for c in cat_list if c in grid_df.columns]
    pay_cols = [f"{m} ₹" for m in PAYMENT_MODES if f"{m} ₹" in grid_df.columns]

    base_cols = []
    for c in ["ID", "Retailer"]:
        if c in grid_df.columns:
            base_cols.append(c)

    ordered_cols = base_cols + cat_cols + pay_cols
    extras = [c for c in grid_df.columns if c not in ordered_cols]

    grid_df = grid_df[ordered_cols + extras]
    edited = st.data_editor(grid_df, width="stretch", num_rows="fixed", key=f"daily_sheet_editor_{posting_date}_{posting_zone}_{st.session_state.get('data_version',0)}")

    # Validate retailer prices exist
    for _, row in edited.iterrows():
        rid = int(row["ID"])
        for cat_name in cat_list:
            qty = float(row.get(cat_name, 0.0) or 0.0)
            if qty <= 0:
                continue
            cid = int(categories_active.loc[categories_active["name"] == cat_name, "category_id"].iloc[0])
            rate = get_price_for_date(rid, cid, posting_date)
            if rate is None or rate <= 0:
                st.error(f"❌ Missing price for Retailer ID {rid} / {cat_name} on {posting_date}")
                st.stop()

    prev_day = posting_date - timedelta(days=1)
    today_sales_list, prev_ledger_list, total_ledger_list = [], [], []

    for _, row in edited.iterrows():
        rid = int(row["ID"])
        prev_ledger = retailer_ledger_as_of(rid, prev_day)
        today_sales = compute_today_sales_amount_for_row(rid, posting_date, row, cat_list, categories_active)

        today_pay = 0.0
        for m in PAYMENT_MODES:
            today_pay += float(row.get(f"{m} ₹", 0.0) or 0.0)

        total_ledger = float(prev_ledger + today_sales - today_pay)
        prev_ledger_list.append(prev_ledger)
        today_sales_list.append(today_sales)
        total_ledger_list.append(total_ledger)

    preview = edited.copy()
    preview["Previous Ledger ₹"] = prev_ledger_list
    preview["Today Sales ₹"] = today_sales_list
    preview["Total Ledger ₹"] = total_ledger_list

    # ---- Column order for Preview: ID, Retailer, Categories, Today Sales, Ledgers, Payments ----
    cat_cols = [c for c in cat_list if c in preview.columns]
    pay_cols = [f"{m} ₹" for m in PAYMENT_MODES if f"{m} ₹" in preview.columns]

    base_cols = []
    for c in ["ID", "Retailer"]:
        if c in preview.columns:
            base_cols.append(c)

    ledger_cols = [c for c in ["Today Sales ₹", "Previous Ledger ₹", "Total Ledger ₹"] if c in preview.columns]

    ordered_cols = base_cols + cat_cols + ledger_cols + pay_cols
    extras = [c for c in preview.columns if c not in ordered_cols]

    preview = preview[ordered_cols + extras]

    st.subheader("📌 Retailer Preview")
    st.dataframe(df_for_display(preview), width="stretch")

    # ---- Totals row (Liters) under Retailer Preview ----
    totals = {"Category Totals": "TOTAL (L)"}
    grand_total = 0.0
    for cat in cat_list:
        s = float(pd.to_numeric(edited.get(cat, 0.0), errors="coerce").fillna(0).sum())
        totals[cat] = s
        grand_total += s
    totals["GRAND TOTAL (L)"] = grand_total

    totals_df = pd.DataFrame([totals])

    # Show totals with dash for zeros (like old bottom row)
    show_totals = totals_df.copy()
    for c in cat_list:
        if c in show_totals.columns:
            show_totals[c] = show_totals[c].apply(fmt_zero_dash)
    if "GRAND TOTAL (L)" in show_totals.columns:
        show_totals["GRAND TOTAL (L)"] = show_totals["GRAND TOTAL (L)"].apply(lambda x: f"{float(x):.2f}")

    st.dataframe(df_for_display(show_totals), width="stretch")



    # ---------------- DISTRIBUTOR ENTRY ----------------
    st.divider()
    st.subheader("📦 Distributor Daily Entry (STRICT mapping + 2-rate + payment)")

    if "dist_daily_payload" not in st.session_state:
        st.session_state["dist_daily_payload"] = {}

    distributors_active = distributors.copy()
    if not distributors_active.empty and "is_active" in distributors_active.columns:
        distributors_active["is_active"] = distributors_active["is_active"].apply(parse_boolish_active)
        distributors_active = distributors_active.loc[distributors_active["is_active"] == True].copy()

    # strict mapping lookup
    def _mapped_category_ids(did: int) -> list[int]:
        if dist_cat_map is None or dist_cat_map.empty:
            return []
        m = dist_cat_map.copy()
        m["distributor_id"] = pd.to_numeric(m.get("distributor_id", 0), errors="coerce").fillna(0).astype(int)
        m["category_id"] = pd.to_numeric(m.get("category_id", 0), errors="coerce").fillna(0).astype(int)
        m["is_active"] = m.get("is_active", True).apply(parse_boolish_active)
        m = m.loc[(m["distributor_id"] == int(did)) & (m["is_active"] == True)].copy()
        return sorted(m["category_id"].astype(int).tolist())

    # default rate = last used for that distributor/category
    def _last_rate_for(did: int, cid: int) -> float:
        if dist_purchases is None or dist_purchases.empty:
            return 0.0
        dp = dist_purchases.copy()
        dp["date"] = _safe_dt(dp.get("date")).dt.date
        dp["distributor_id"] = pd.to_numeric(dp.get("distributor_id", 0), errors="coerce").fillna(0).astype(int)
        dp["category_id"] = pd.to_numeric(dp.get("category_id", 0), errors="coerce").fillna(0).astype(int)
        dp["rate"] = pd.to_numeric(dp.get("rate", 0.0), errors="coerce").fillna(0.0).astype(float)
        dp = dp.loc[(dp["distributor_id"] == int(did)) & (dp["category_id"] == int(cid))].copy()
        if dp.empty:
            return 0.0
        dp = dp.sort_values(["date"], ascending=[False])
        return float(dp.iloc[0]["rate"] or 0.0)

    if distributors_active.empty:
        st.info("No active distributors.")
    else:
        for _, drow in distributors_active.sort_values("name").iterrows():
            did = int(drow["distributor_id"])
            dname = str(drow["name"])

            mapped_ids = _mapped_category_ids(did)
            if not mapped_ids:
                st.warning(f"⚠️ No mapping set for {dname}. Go to '🧩 Distributor Category Mapping' and set categories.")
                show_cats = pd.DataFrame(columns=["category_id", "name"])
            else:
                show_cats = categories_active.loc[categories_active["category_id"].astype(int).isin(mapped_ids), ["category_id", "name"]].copy()
                show_cats = show_cats.sort_values("name").reset_index(drop=True)

            with st.expander(f"🚚 {dname}", expanded=False):
                if show_cats.empty:
                    st.info("No categories mapped → nothing to enter here.")
                    st.session_state["dist_daily_payload"][did] = {
                        "name": dname,
                        "editor": pd.DataFrame(columns=["Category", "Qty A", "Rate A", "Qty B", "Rate B"]),
                        "category_ids": [],
                        "payment_amt": 0.0,
                        "payment_mode": "Cash",
                        "payment_note": "",
                    }
                else:
                    base = pd.DataFrame({
                        "category_id": show_cats["category_id"].astype(int),
                        "Category": show_cats["name"].astype(str),
                        "Qty A": 0.0,
                        "Rate A": 0.0,
                        "Qty B": 0.0,
                        "Rate B": 0.0,
                    })

                    # Prefill from today's saved purchases into A/B slots
                    if dist_purchases is not None and not dist_purchases.empty:
                        dp = dist_purchases.copy()
                        dp["date"] = _safe_dt(dp.get("date")).dt.date
                        dp["distributor_id"] = pd.to_numeric(dp.get("distributor_id", 0), errors="coerce").fillna(0).astype(int)
                        dp["category_id"] = pd.to_numeric(dp.get("category_id", 0), errors="coerce").fillna(0).astype(int)
                        dp["qty"] = pd.to_numeric(dp.get("qty", 0.0), errors="coerce").fillna(0.0).astype(float)
                        dp["rate"] = pd.to_numeric(dp.get("rate", 0.0), errors="coerce").fillna(0.0).astype(float)

                        dp_today = dp.loc[(dp["date"] == posting_date) & (dp["distributor_id"] == did)].copy()
                        if not dp_today.empty:
                            for cid, grp in dp_today.groupby("category_id"):
                                if cid not in set(base["category_id"].tolist()):
                                    continue
                                idx = int(base.index[base["category_id"] == int(cid)][0])
                                rows = grp.sort_values("rate").to_dict(orient="records")
                                if len(rows) >= 1:
                                    base.loc[idx, "Qty A"] = float(rows[0]["qty"])
                                    base.loc[idx, "Rate A"] = float(rows[0]["rate"])
                                if len(rows) >= 2:
                                    base.loc[idx, "Qty B"] = float(rows[1]["qty"])
                                    base.loc[idx, "Rate B"] = float(rows[1]["rate"])

                    # Default Rate A if blank: last rate
                    for i in range(len(base)):
                        cid = int(base.loc[i, "category_id"])
                        if float(base.loc[i, "Rate A"] or 0.0) <= 0:
                            base.loc[i, "Rate A"] = _last_rate_for(did, cid)

                    dist_edit = st.data_editor(
                        base[["Category", "Qty A", "Rate A", "Qty B", "Rate B"]],
                        num_rows="fixed",
                        width="stretch",
                        key=f"dist_editor_{did}_{posting_date}",
                    )

                    # Payment
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1:
                        pay_amt = st.number_input("Payment Made (₹)", min_value=0.0, step=50.0, format="%g",
                                                  key=f"dist_pay_amt_{did}_{posting_date}")
                    with c2:
                        pay_mode = st.selectbox("Mode", ["Cash", "UPI", "Bank", "Cheque", "Other"],
                                                key=f"dist_pay_mode_{did}_{posting_date}")
                    with c3:
                        pay_note = st.text_input("Note", value="", key=f"dist_pay_note_{did}_{posting_date}")

                    st.session_state["dist_daily_payload"][did] = {
                        "name": dname,
                        "editor": dist_edit.copy(),
                        "category_ids": show_cats["category_id"].astype(int).tolist(),
                        "payment_amt": float(pay_amt or 0.0),
                        "payment_mode": str(pay_mode),
                        "payment_note": str(pay_note),
                    }

    # ---------------- DISTRIBUTOR PREVIEW + TOTALS (same-page) ----------------
    st.divider()
    st.subheader("📌 Distributor Preview")

    dist_rows = []
    cat_acc = {}  # category_id -> total qty (L)
    for did, payload in st.session_state.get("dist_daily_payload", {}).items():
        try:
            did_int = int(did)
        except Exception:
            continue

        dname = str(payload.get("name", f"Distributor {did_int}"))
        prev_due = float(distributor_balance_before(did_int, posting_date))

        df_in = payload.get("editor")
        cat_ids = payload.get("category_ids", []) or []

        today_purchase_amt = 0.0
        today_qty_total = 0.0

        if isinstance(df_in, pd.DataFrame) and not df_in.empty:
            for i, r in df_in.reset_index(drop=True).iterrows():
                qA = float(pd.to_numeric(r.get("Qty A", 0.0), errors="coerce") or 0.0)
                rtA = float(pd.to_numeric(r.get("Rate A", 0.0), errors="coerce") or 0.0)
                qB = float(pd.to_numeric(r.get("Qty B", 0.0), errors="coerce") or 0.0)
                rtB = float(pd.to_numeric(r.get("Rate B", 0.0), errors="coerce") or 0.0)

                if qA > 0 and rtA > 0:
                    today_purchase_amt += qA * rtA
                if qB > 0 and rtB > 0:
                    today_purchase_amt += qB * rtB

                qty_sum = max(0.0, qA) + max(0.0, qB)
                today_qty_total += qty_sum

                # map qty to category_id (aligned by row index)
                if i < len(cat_ids):
                    cid = int(cat_ids[i])
                    cat_acc[cid] = cat_acc.get(cid, 0.0) + qty_sum

        pay_amt = float(payload.get("payment_amt", 0.0) or 0.0)
        closing_due = prev_due + today_purchase_amt - pay_amt

        dist_rows.append({
            "Distributor": dname,
            "Milk (L)": today_qty_total,
            "Previous Due ₹": prev_due,
            "Today Purchases ₹": today_purchase_amt,
            "Today Payment ₹": pay_amt,
            "Closing Due ₹": closing_due,
        })

    dist_prev = pd.DataFrame(dist_rows)
    if dist_prev.empty:
        st.info("No distributor entries for this date.")
    else:
        st.dataframe(df_for_display(dist_prev), width="stretch")


        # Category totals bar (Liters) across all distributors
        cid_to_name = {int(c["category_id"]): str(c["name"]) for _, c in categories_active.iterrows()}
        cat_totals = {"Category Totals": "TOTAL (L)"}
        grand = 0.0
        for cid, qty in sorted(cat_acc.items(), key=lambda x: cid_to_name.get(x[0], str(x[0]))):
            cname = cid_to_name.get(cid, f"Category {cid}")
            cat_totals[cname] = float(qty)
            grand += float(qty)
        cat_totals["GRAND TOTAL (L)"] = float(grand)
        tot2 = pd.DataFrame([cat_totals])
        for c in tot2.columns:
            if c not in ("Category Totals", "GRAND TOTAL (L)"):
                tot2[c] = tot2[c].apply(fmt_zero_dash)
        if "GRAND TOTAL (L)" in tot2.columns:
            tot2["GRAND TOTAL (L)"] = tot2["GRAND TOTAL (L)"].apply(lambda x: f"{float(x):.2f}")
        st.dataframe(df_for_display(tot2), width="stretch")


    # ---------------- WASTAGE ENTRY ----------------
    st.divider()
    st.subheader("🗑️ Wastage Entry (same page)")

    waste_base = pd.DataFrame({
        "category_id": categories_active["category_id"].astype(int),
        "Category": categories_active["name"].astype(str),
        "Qty Wasted (L)": 0.0,
        "Estimated Loss (₹)": 0.0,
        "Reason": "",
    }).sort_values("Category").reset_index(drop=True)

    if wastage is not None and not wastage.empty:
        wz = wastage.copy()
        wz["date"] = _safe_dt(wz.get("date")).dt.date
        wz["category_id"] = pd.to_numeric(wz.get("category_id", 0), errors="coerce").fillna(0).astype(int)
        wz["qty"] = pd.to_numeric(wz.get("qty", 0.0), errors="coerce").fillna(0.0).astype(float)
        wz["estimated_loss"] = pd.to_numeric(wz.get("estimated_loss", 0.0), errors="coerce").fillna(0.0).astype(float)
        wz["reason"] = wz.get("reason", "").fillna("").astype(str)

        wz_today = wz.loc[wz["date"] == posting_date].copy()
        if not wz_today.empty:
            for cid, grp in wz_today.groupby("category_id"):
                if cid in set(waste_base["category_id"].tolist()):
                    idx = int(waste_base.index[waste_base["category_id"] == int(cid)][0])
                    r0 = grp.iloc[0]
                    waste_base.loc[idx, "Qty Wasted (L)"] = float(r0["qty"])
                    waste_base.loc[idx, "Estimated Loss (₹)"] = float(r0["estimated_loss"])
                    waste_base.loc[idx, "Reason"] = str(r0["reason"])

    waste_edit = st.data_editor(
        waste_base[["Category", "Qty Wasted (L)", "Estimated Loss (₹)", "Reason"]],
        num_rows="fixed",
        width="stretch",
        key=f"waste_editor_{posting_date}",
    )

    # ---------------- SINGLE SAVE ----------------
    st.divider()
    if st.button("💾 Save (Retailers + Distributors + Wastage)", type="primary", key="single_save_all"):
        # SAFETY GUARD:
        # If DB already has data for this date+zone, but the screen is all zeros,
        # do NOT overwrite (this prevents accidental data wipe when UI state is stale).
        try:
            db_e_chk = _day_entries_for_zone(posting_date, posting_zone)
            db_p_chk = _day_payments_for_zone(posting_date, posting_zone)
            has_db_data = (db_e_chk is not None and not db_e_chk.empty) or (db_p_chk is not None and not db_p_chk.empty)

            screen_total_qty = 0.0
            for c in cat_list:
                if c in preview.columns:
                    screen_total_qty += float(pd.to_numeric(preview[c], errors="coerce").fillna(0.0).sum())

            screen_total_pay = 0.0
            for m in PAYMENT_MODES:
                col = f"{m} ₹"
                if col in preview.columns:
                    screen_total_pay += float(pd.to_numeric(preview[col], errors="coerce").fillna(0.0).sum())

            if has_db_data and (screen_total_qty == 0.0 and screen_total_pay == 0.0):
                st.error("❌ Save blocked: DB has data for this date/zone, but your screen is all zeros. This usually means your screen is stale. Reload the page or change the date away and back, then try again.")
                st.stop()
        except Exception:
            # If anything goes wrong, do not block saving (better than crashing),
            # but still allow manual refresh to fix stale state.
            pass

        # Retailers overwrite (date + zone retailers)
        rz = retailers_active.copy()
        rz["zone"] = rz["zone"].apply(_norm_zone)
        if posting_zone != "All Zones":
            rz = rz.loc[rz["zone"] == _norm_zone(posting_zone)].copy()
        affected_rids = set(rz["retailer_id"].astype(int).tolist())

        if not affected_rids:
            st.error("No retailers found for selected zone.")
            st.stop()

        sb_delete_where("entries", [("date", "eq", str(posting_date)), ("retailer_id", "in", list(affected_rids))])
        sb_delete_where("payments", [("date", "eq", str(posting_date)), ("retailer_id", "in", list(affected_rids))])

        next_entry_id = None if USE_DB_IDS else sb_next_id("entries", "entry_id")
        next_pay_id = None if USE_DB_IDS else sb_next_id("payments", "payment_id")

        new_entries, new_payments = [], []

        for _, row in preview.iterrows():
            rid = int(row["ID"])
            if rid not in affected_rids:
                continue

            for cat_name in cat_list:
                qty = float(row.get(cat_name, 0.0) or 0.0)
                if qty <= 0:
                    continue
                cid = int(categories_active.loc[categories_active["name"] == cat_name, "category_id"].iloc[0])
                rate = float(get_price_for_date(rid, cid, posting_date))
                amt = float(qty * rate)

                eid = None if USE_DB_IDS else next_entry_id
                if not USE_DB_IDS:
                    next_entry_id += 1
                new_entries.append([eid, str(posting_date), int(rid), int(cid), float(qty), float(rate), float(amt)])

            for m in PAYMENT_MODES:
                pamt = float(row.get(f"{m} ₹", 0.0) or 0.0)
                if pamt <= 0:
                    continue
                pid = None if USE_DB_IDS else next_pay_id
                if not USE_DB_IDS:
                    next_pay_id += 1
                new_payments.append([pid, str(posting_date), int(rid), float(pamt), str(m), ""])

        if new_entries:
            sb_insert_df(pd.DataFrame(new_entries, columns=CSV_SCHEMAS[ENTRIES_FILE]), ENTRIES_FILE)
        if new_payments:
            sb_insert_df(pd.DataFrame(new_payments, columns=CSV_SCHEMAS[PAYMENTS_FILE]), PAYMENTS_FILE)

        # Distributors overwrite (date for those distributors)
        payload = st.session_state.get("dist_daily_payload", {}) or {}
        dist_ids = sorted([int(x) for x in payload.keys()])

        if dist_ids:
            sb_delete_where("distributor_purchases", [("date", "eq", str(posting_date)), ("distributor_id", "in", dist_ids)])
            sb_delete_where("distributor_payments", [("date", "eq", str(posting_date)), ("distributor_id", "in", dist_ids)])

            next_pur_id = None if USE_DB_IDS else sb_next_id("distributor_purchases", "purchase_id")
            next_dpay_id = None if USE_DB_IDS else sb_next_id("distributor_payments", "payment_id")

            new_dp, new_dpay = [], []

            for did in dist_ids:
                d = payload[did]
                dist_edit = d["editor"].copy()
                show_cat_ids = d["category_ids"]

                # Purchases with A/B slabs (same category allowed twice with different rates)
                for i in range(len(dist_edit)):
                    if i >= len(show_cat_ids):
                        continue
                    cid = int(show_cat_ids[i])

                    qa = float(dist_edit.loc[i, "Qty A"] or 0.0)
                    ra = float(dist_edit.loc[i, "Rate A"] or 0.0)
                    qb = float(dist_edit.loc[i, "Qty B"] or 0.0)
                    rb = float(dist_edit.loc[i, "Rate B"] or 0.0)

                    if qa > 0:
                        pid = None if USE_DB_IDS else next_pur_id
                        if not USE_DB_IDS:
                            next_pur_id += 1
                        new_dp.append([pid, str(posting_date), int(did), int(cid), float(qa), float(ra), float(qa * ra)])

                    if qb > 0:
                        pid = None if USE_DB_IDS else next_pur_id
                        if not USE_DB_IDS:
                            next_pur_id += 1
                        new_dp.append([pid, str(posting_date), int(did), int(cid), float(qb), float(rb), float(qb * rb)])

                # Payment made today
                pay_amt = float(d.get("payment_amt", 0.0) or 0.0)
                if pay_amt > 0:
                    ppid = None if USE_DB_IDS else next_dpay_id
                    if not USE_DB_IDS:
                        next_dpay_id += 1
                    new_dpay.append([ppid, str(posting_date), int(did), float(pay_amt), str(d.get("payment_mode", "Cash")), str(d.get("payment_note", ""))])

            if new_dp:
                sb_insert_df(pd.DataFrame(new_dp, columns=CSV_SCHEMAS[DISTRIBUTOR_PURCHASES_FILE]), DISTRIBUTOR_PURCHASES_FILE)
            if new_dpay:
                sb_insert_df(pd.DataFrame(new_dpay, columns=CSV_SCHEMAS[DISTRIBUTOR_PAYMENTS_FILE]), DISTRIBUTOR_PAYMENTS_FILE)

        # Wastage overwrite for date
        sb_delete_where("wastage", [("date", "eq", str(posting_date))])

        next_wid = None if USE_DB_IDS else sb_next_id("wastage", "wastage_id")
        new_w = []
        waste_map = waste_base[["Category", "category_id"]].copy().reset_index(drop=True)

        for i in range(len(waste_edit)):
            cid = int(waste_map.loc[i, "category_id"])
            q = float(waste_edit.loc[i, "Qty Wasted (L)"] or 0.0)
            loss = float(waste_edit.loc[i, "Estimated Loss (₹)"] or 0.0)
            reason = str(waste_edit.loc[i, "Reason"] or "")
            if q <= 0:
                continue

            wid = None if USE_DB_IDS else next_wid
            if not USE_DB_IDS:
                next_wid += 1
            new_w.append([wid, str(posting_date), int(cid), float(q), str(reason), float(loss)])

        if new_w:
            sb_insert_df(pd.DataFrame(new_w, columns=CSV_SCHEMAS[WASTAGE_FILE]), WASTAGE_FILE)

        st.success("✅ Saved: Retailers + Distributors + Wastage")
        invalidate_data_cache()
        st.rerun()
# ================== DATE + ZONE VIEW ==================
elif menu == "📅 Date + Zone View":
    st.header("📅 View All Data for a Specific Date + Zone")

    view_date = st.date_input("Select Date", value=date.today(), key="view_date")
    view_zone = st.selectbox("Select Zone", ["All Zones"] + get_all_zones(), key="view_zone")

    e_day = _day_entries_for_zone(view_date, view_zone)
    if e_day.empty:
        st.info("No entries for this date/zone.")
    else:
        e_view = build_entries_view_cached(e_day, st.session_state["data_version"], want_milk_type_col=False)

        pivot = pd.pivot_table(
            e_view,
            index="Retailer",
            columns="Category",
            values="qty",
            aggfunc="sum",
            fill_value=0.0
        )

        display = pivot.copy()
        for c in display.columns:
            vals = pd.to_numeric(display[c], errors="coerce").fillna(0.0)
            display[c] = vals.apply(lambda v: "–" if float(v) == 0.0 else f"{float(v):.2f}")
        display["TOTAL (L)"] = pivot.sum(axis=1).apply(lambda x: f"{float(x):.2f}")

        totals = {c: float(pivot[c].sum()) for c in pivot.columns}
        totals["TOTAL (L)"] = float(pivot.values.sum())
        totals_row = {k: (f"{float(v):.2f}" if k != "TOTAL (L)" else f"{float(v):.2f}") for k, v in totals.items()}
        display = pd.concat([display, pd.DataFrame([totals_row], index=["GRAND TOTAL"])])

        st.subheader("🥛 Retailer × Category (Liters)")
        st.dataframe(df_for_display(display), width="stretch")

    p_day = _day_payments_for_zone(view_date, view_zone)
    st.subheader("💳 Payments (This Date + Zone)")
    if p_day.empty:
        st.info("No payments for this date/zone.")
    else:
        pv = p_day.merge(
            retailers[["retailer_id", "name", "zone"]],
            on="retailer_id",
            how="left"
        ).rename(columns={"name": "Retailer"})
        pv["zone"] = pv["zone"].apply(_norm_zone)

        st.dataframe(
            pv[["date", "zone", "Retailer", "amount", "payment_mode", "note"]],
            width="stretch"
        )

        mode_totals = (
            pv.groupby("payment_mode", as_index=False)["amount"]
              .sum()
              .sort_values("amount", ascending=False)
              .rename(columns={"payment_mode": "Mode", "amount": "Total (₹)"})
        )

        st.subheader("💳 Payment Totals by Mode")
        st.dataframe(
            mode_totals.style.format({"Total (₹)": "₹{:.2f}"}),
            width="stretch"
        )

# ================== ZONE-WISE SUMMARY ==================
elif menu == "📍 Zone-wise Summary":
    st.header("📍 Zone-wise Summary (Category Columns + Payment Mode Totals)")

    s_date = st.date_input("Select Date", value=date.today(), key="zone_sum_date")

    pivot = zone_category_pivot_for_day(s_date)

    st.subheader("🥛 Milk Sent — Zone × Category (Liters)")
    if pivot.empty:
        st.info("No entries on this date.")
    else:
        display = pivot.copy()

        for c in display.columns:
            if c in ("Zone", "TOTAL (L)"):
                continue
            display[c] = display[c].apply(fmt_zero_dash)

        numeric_cols = [c for c in pivot.columns if c != "Zone"]
        grand = {"Zone": "GRAND TOTAL"}
        for c in numeric_cols:
            grand[c] = float(pivot[c].sum())
        display = pd.concat([display, pd.DataFrame([grand])], ignore_index=True)

        st.dataframe(df_for_display(display), width="stretch")

    st.subheader("💳 Payments Collected — Totals by Mode (Zone-aware)")
    p_day = _day_payments_for_zone(s_date, "All Zones")
    if p_day.empty:
        st.info("No payments recorded on this date.")
    else:
        pv = p_day.merge(retailers[["retailer_id", "zone"]], on="retailer_id", how="left")
        pv["zone"] = pv["zone"].apply(_norm_zone)

        mode_totals = (
            pv.groupby("payment_mode", as_index=False)["amount"]
            .sum()
            .sort_values("amount", ascending=False)
            .rename(columns={"payment_mode": "Mode", "amount": "Total (₹)"})
        )

        mode_zone = (
            pv.groupby(["zone", "payment_mode"], as_index=False)["amount"]
            .sum()
            .rename(columns={"zone": "Zone", "payment_mode": "Mode", "amount": "Total (₹)"})
            .sort_values(["Zone", "Total (₹)"], ascending=[True, False])
        )

        st.caption("Overall totals (all zones combined):")
        st.dataframe(mode_totals.style.format({"Total (₹)": "₹{:.2f}"}), width="stretch")

        st.caption("Zone-wise totals by mode:")
        st.dataframe(mode_zone.style.format({"Total (₹)": "₹{:.2f}"}), width="stretch")

# ================== EDIT SINGLE ENTRY ==================
elif menu == "✏️ Edit (Single Entry)":
    st.header("✏️ Edit / Delete Single Entry (Rate is preserved)")

    if entries.empty:
        st.info("No entries yet.")
        st.stop()

    f_date = st.date_input("Filter Date", value=date.today(), key="single_edit_date")
    f_zone = st.selectbox("Filter Zone", ["All Zones"] + get_all_zones(), key="single_edit_zone")

    df = entries.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.loc[df["date"] == f_date].copy()
    df = filter_by_zone(df, "retailer_id", f_zone)

    if df.empty:
        st.info("No entries for this date/zone.")
        st.stop()

    view = build_entries_view_cached(df, st.session_state["data_version"], want_milk_type_col=False)
    st.dataframe(
        view[["entry_id", "date", "zone", "Retailer", "Category", "qty", "rate", "amount"]],
        width="stretch"
    )

    entry_id = st.number_input("Entry ID", min_value=1, step=1, key="single_entry_id")
    if int(entry_id) not in set(entries["entry_id"].astype(int).tolist()):
        st.warning("Entry ID not found.")
        st.stop()

    row = entries.loc[entries["entry_id"] == int(entry_id)].iloc[0]
    new_qty = st.number_input("New Quantity (L)", min_value=0.0, step=0.5, format="%g", value=float(row["qty"]), key="single_new_qty")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Update Qty", key="single_update_btn"):
            rate = float(row["rate"])
            new_amt = float(new_qty) * rate
            entries.loc[entries["entry_id"] == int(entry_id), ["qty", "amount"]] = [float(new_qty), float(new_amt)]
            # DB write only the changed row
            updated_row = entries.loc[entries["entry_id"] == int(entry_id)].copy()
            safe_write_csv(updated_row, ENTRIES_FILE, allow_empty=False)
            
            st.success("Updated quantity (stored rate preserved).")
            st.rerun()

    with col2:
        confirm = st.text_input("Type DELETE to delete this entry", key="single_delete_confirm")
        if st.button("Delete Entry", key="single_delete_btn"):
            if confirm != "DELETE":
                st.warning("Type DELETE to confirm.")
            else:
                sb_delete_by_pk("entries", "entry_id", [int(entry_id)])
                entries = entries.loc[entries["entry_id"].astype(int) != int(entry_id)].copy()
                st.success("Deleted.")
                st.rerun()



# ================== MILK CATEGORIES ==================
elif menu == "🥛 Milk Categories":
    st.header("🥛 Milk Categories Management")
    tab1, tab2 = st.tabs(["➕ Add Category", "✏️ Edit Categories"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Category Name", key="cat_add_name")
        with col2:
            description = st.text_input("Description (optional)", key="cat_add_desc")
        with col3:
            default_price = st.number_input("Default Price per Liter (₹)", min_value=0.0, step=0.5, format="%g", key="cat_add_price")

        if st.button("Add Category", type="primary", key="cat_add_btn"):
            if not name.strip():
                st.error("Category name is required.")
                st.stop()
                
            if default_price <= 0:
                st.error("Default price must be greater than 0.")
                st.stop()
                
            # OPTION B: DB auto-generates category_id
            table, pk = FILE_TO_TABLE[CATEGORIES_FILE]
            
            new_row = pd.DataFrame([{
                "name": name.strip(),
                "description": description.strip() if description else "",
                "default_price": float(default_price),
                "is_active": True,
            }])
            
            sb_insert_df(new_row, CATEGORIES_FILE)

            st.success("Category added.")
            st.rerun()




    with tab2:
        if categories.empty:
            st.info("No categories yet.")
        else:
            st.dataframe(categories, width="stretch")

            edit_cat = st.selectbox("Select category to edit", categories["name"].tolist(), key="cat_edit_sel")
            cat_data = categories.loc[categories["name"] == edit_cat].iloc[0]
            cid = int(cat_data["category_id"])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                new_name = st.text_input("New name", value=str(cat_data["name"]), key="cat_new_name")
            with col2:
                new_desc = st.text_input("New description", value=str(cat_data.get("description", "")), key="cat_new_desc")
            with col3:
                new_default_price = st.number_input(
                    "Default Price (₹/L)",
                    value=float(cat_data.get("default_price", 0.0)),
                    min_value=0.0,
                    step=0.5,
                    format="%g",
                    key="cat_new_price",
                )
            with col4:
                new_active = st.checkbox("Active", value=bool(cat_data.get("is_active", True)), key="cat_new_active")

            colA, colB, colC = st.columns(3)
            
            
            with colA:
                if st.button("Update Category", key="cat_update_btn"):
                    mask = categories["category_id"].astype(int) == int(cid)
                    
                    categories.loc[mask, ["name", "description", "default_price", "is_active"]] = [
                        new_name.strip(),
                        new_desc,
                        float(new_default_price),
                        bool(new_active),
                    ]
                    updated_row = categories.loc[mask].copy()
                    safe_write_csv(updated_row, CATEGORIES_FILE, allow_empty=False)
                    
                    st.success("Updated!")
                    st.rerun()


            with colB:
                if st.button("Deactivate Category (Safe)", key="cat_deactivate_btn"):
                    mask = categories["category_id"].astype(int) == int(cid)
                    categories.loc[mask, "is_active"] = False
                    
                    updated_row = categories.loc[mask].copy()
                    safe_write_csv(updated_row, CATEGORIES_FILE, allow_empty=False)
                    
                    st.success("Category deactivated (history preserved).")
                    st.rerun()


            with colC:
                st.caption("Hard delete is blocked if referenced.")
                confirm = st.text_input("Type DELETE to hard delete category", key="cat_delete_confirm")
                if st.button("🗑️ Hard Delete Category", type="secondary", key="cat_delete_btn"):
                    if confirm != "DELETE":
                        st.warning("Type DELETE to confirm.")
                    elif is_category_referenced(cid):
                        st.error("Blocked: Category is referenced in history. Deactivate instead.")
                    else:
                        sb_delete_by_pk("categories", "category_id", [cid])
                        st.success("Hard deleted.")
                        st.rerun()

# ================== RETAILERS ==================
elif menu == "🏪 Retailers":
    st.header("🏪 Retailer Management (with Zones)")
    tab1, tab2 = st.tabs(["➕ Add Retailer", "✏️ Edit Retailers"])

    all_zones = get_all_zones()

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Retailer Name", key="ret_add_name")
            contact = st.text_input("Contact Number", key="ret_add_contact")
        with col2:
            address = st.text_area("Address", key="ret_add_address")
        with col3:
            zone_mode = st.radio("Zone", ["Select Existing", "Create New"], horizontal=True, key="ret_add_zone_mode")
            if zone_mode == "Select Existing":
                zone = st.selectbox("Select Zone", ["Default"] + all_zones, key="ret_add_zone_sel")
            else:
                zone = st.text_input("New Zone Name", value="", key="ret_add_zone_new")

        if st.button("Add Retailer", type="primary", key="ret_add_btn"):
            if not name.strip():
                st.warning("Retailer name required")
            else:
                z = _norm_zone(zone)
                rid = sb_new_id("retailers", "retailer_id")
                new_row = pd.DataFrame([[rid, name.strip(), contact, address, z, True]], columns=CSV_SCHEMAS[RETAILERS_FILE])
                sb_insert_df(new_row, RETAILERS_FILE)
                st.success(f"✅ Retailer '{name}' added to zone '{z}'!")
                st.rerun()

    with tab2:
        if retailers.empty:
            st.info("No retailers yet.")
        else:
            st.dataframe(retailers, width="stretch")

            edit_ret = st.selectbox("Select retailer to edit", retailers["name"].tolist(), key="ret_edit_sel")
            ret_data = retailers.loc[retailers["name"] == edit_ret].iloc[0]
            rid = int(ret_data["retailer_id"])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                new_name = st.text_input("New name", value=str(ret_data["name"]), key="ret_new_name")
                new_contact = st.text_input("New contact", value=str(ret_data.get("contact", "")), key="ret_new_contact")
            with col2:
                new_address = st.text_area("New address", value=str(ret_data.get("address", "")), key="ret_new_address")
            with col3:
                zone_mode2 = st.radio("Zone Update", ["Select Existing", "Create New"], horizontal=True, key="ret_zone_update_mode")
                if zone_mode2 == "Select Existing":
                    choices = ["Default"] + all_zones
                    current = _norm_zone(ret_data.get("zone", "Default"))
                    idx = choices.index(current) if current in choices else 0
                    new_zone = st.selectbox("Select Zone", choices, index=idx, key="ret_new_zone_sel")
                else:
                    new_zone = st.text_input("New Zone Name", value="", key="ret_new_zone_new")
            with col4:
                new_active = st.checkbox("Active", value=bool(ret_data.get("is_active", True)), key="ret_new_active")

            colA, colB, colC = st.columns(3)
            
            
            with colA:
                if st.button("Update Retailer", key="ret_update_btn"):
                    z = _norm_zone(new_zone)
                    mask = retailers["retailer_id"].astype(int) == int(rid)
                    
                    retailers.loc[mask, ["name", "contact", "address", "zone", "is_active"]] = [
                        new_name.strip(),
                        new_contact,
                        new_address,
                        z,
                        bool(new_active),
                    ]
                    
                    updated_row = retailers.loc[mask].copy()
                    safe_write_csv(updated_row, RETAILERS_FILE, allow_empty=False)
                    
                    st.success("Updated!")
                    st.rerun()


            with colB:
                if st.button("Deactivate Retailer (Safe)", key="ret_deactivate_btn"):
                    mask = retailers["retailer_id"].astype(int) == int(rid)
                    retailers.loc[mask, "is_active"] = False
                    
                    updated_row = retailers.loc[mask].copy()
                    safe_write_csv(updated_row, RETAILERS_FILE, allow_empty=False)
                    
                    st.success("Retailer deactivated (history preserved).")
                    st.rerun()


            with colC:
                st.caption("Hard delete is blocked if referenced.")
                confirm = st.text_input("Type DELETE to hard delete retailer", key="ret_delete_confirm")
                if st.button("🗑️ Hard Delete Retailer", type="secondary", key="ret_delete_btn"):
                    if confirm != "DELETE":
                        st.warning("Type DELETE to confirm.")
                    elif is_retailer_referenced(rid):
                        st.error("Blocked: Retailer is referenced in history. Deactivate instead.")
                    else:
                        sb_delete_by_pk("retailers", "retailer_id", [rid])
                        st.success("Hard deleted.")
                        st.rerun()

# ================== PRICE MANAGEMENT ==================
elif menu == "💰 Price Management":
    st.header("💰 Price Management (Global + Retailer Override)")
    st.info("Reminder: prices affect NEW entries. Old entries keep their stored rate (history is protected).")

    tab1, tab2 = st.tabs(["➕ Set Price (Global or Retailer)", "✏️ View/Edit Prices"])

    with tab1:
        if categories_active.empty:
            st.warning("⚠️ Please add active categories first")
            st.stop()

        scope = st.radio("Price Type", ["🌍 Global Default (All Retailers)", "🏪 Specific Retailer Override"], horizontal=True, key="price_scope")

        col1, col2, col3 = st.columns(3)
        with col1:
            if scope == "🏪 Specific Retailer Override":
                if retailers_active.empty:
                    st.warning("⚠️ Add active retailers first")
                    st.stop()
                retailer_name = st.selectbox("Retailer", retailers_active["name"].tolist(), key="price_retailer")
            else:
                retailer_name = None
                st.info("This applies to ALL retailers (from effective date onward).")

        with col2:
            category_name = st.selectbox("Milk Category", categories_active["name"].tolist(), key="price_category")
            cat_row = categories_active.loc[categories_active["name"] == category_name].iloc[0]
            cid = int(cat_row["category_id"])

        with col3:
            fallback = float(cat_row.get("default_price", 0.0)) if pd.notna(cat_row.get("default_price", 0.0)) else 0.0
            price_val = st.number_input("Set Price per Liter (₹)", min_value=0.0, step=0.5, format="%g", value=float(fallback), key="price_val")

        effective_date = st.date_input("Effective Date", date.today(), key="price_eff_date")

        if st.button("Save Price", type="primary", key="price_save_btn"):
            if float(price_val) <= 0:
                st.warning("Price must be > 0")
            else:
                if scope == "🏪 Specific Retailer Override":
                    rid = int(retailers_active.loc[retailers_active["name"] == retailer_name, "retailer_id"].values[0])
                    target_retailer_id = rid
                else:
                    target_retailer_id = GLOBAL_RETAILER_ID

                pid = sb_new_id("prices", "price_id")
                new_row = pd.DataFrame([[pid, int(target_retailer_id), int(cid), float(price_val), str(effective_date)]], columns=CSV_SCHEMAS[PRICES_FILE])
                sb_insert_df(new_row, PRICES_FILE)
                st.success("✅ Price saved. New entries on/after effective date will use it.")
                st.rerun()

    with tab2:
        if prices.empty:
            st.info("No prices set yet.")
        else:
            view = prices.copy()
            view["retailer_id"] = pd.to_numeric(view["retailer_id"], errors="coerce").fillna(GLOBAL_RETAILER_ID).astype(int)
            view["category_id"] = pd.to_numeric(view["category_id"], errors="coerce").fillna(0).astype(int)

            view = view.merge(categories[["category_id", "name"]], on="category_id", how="left").rename(columns={"name": "Category"})
            retail_map = retailers[["retailer_id", "name"]] if not retailers.empty else pd.DataFrame(columns=["retailer_id", "name"])
            view = view.merge(retail_map, on="retailer_id", how="left").rename(columns={"name": "Retailer"})
            view.loc[view["retailer_id"] == GLOBAL_RETAILER_ID, "Retailer"] = "🌍 GLOBAL (All Retailers)"
            view["effective_date"] = pd.to_datetime(view["effective_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            view = view.sort_values(["retailer_id", "category_id", "effective_date"], ascending=[True, True, False])
            st.dataframe(view[["price_id", "Retailer", "Category", "price", "effective_date"]], width="stretch")

# ================== LEDGER ==================
elif menu == "📒 Ledger":
    st.header(f"📒 Ledger (Grid View) — {selected_zone}")

    if entries_z.empty:
        st.info("No entries in this zone context.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        d_from = st.date_input("From Date", value=date.today() - timedelta(days=30), key="ledger_from")
    with col2:
        d_to = st.date_input("To Date", value=date.today(), key="ledger_to")

    if d_from > d_to:
        st.error("From Date cannot be after To Date.")
        st.stop()

    v = build_entries_view_cached(entries_z, st.session_state["data_version"], want_milk_type_col=False)
    v["date"] = pd.to_datetime(v["date"], errors="coerce").dt.date
    v = v.loc[(v["date"] >= d_from) & (v["date"] <= d_to)].copy()

    if v.empty:
        st.info("No entries in this date range.")
        st.stop()

    pivot = pd.pivot_table(
        v,
        index="Retailer",
        columns="Category",
        values="qty",
        aggfunc="sum",
        fill_value=0.0
    )

    display = pivot.copy()
    for c in display.columns:
        display[c] = display[c].apply(fmt_zero_dash)

    display["TOTAL (L)"] = pivot.sum(axis=1).apply(lambda x: f"{float(x):.2f}")

    grand = {"TOTAL (L)": float(pivot.values.sum())}
    for c in pivot.columns:
        grand[c] = float(pivot[c].sum())
    grand_disp = {k: (f"{float(v):.2f}" if k != "TOTAL (L)" else f"{float(v):.2f}") for k, v in grand.items()}
    display = pd.concat([display, pd.DataFrame([grand_disp], index=["GRAND TOTAL"])])

    st.subheader("🥛 Retailer × Category Grid (Liters)")
    st.dataframe(df_for_display(display), width="stretch")

    st.subheader("📌 Category Totals (Liters)")
    cat_totals = pivot.sum(axis=0).reset_index()
    cat_totals.columns = ["Category", "Total (L)"]
    cat_totals = cat_totals.sort_values("Total (L)", ascending=False)
    st.dataframe(cat_totals, width="stretch")

# ================== FILTERS & REPORTS ==================
elif menu == "🔍 Filters & Reports":
    st.header(f"🔍 Filters & Detailed Reports — {selected_zone}")

    if selected_zone == "All Zones":
        zone_filter = st.multiselect("Zone Filter", ["All"] + zones, default=["All"], key="rep_zone_filter")
    else:
        zone_filter = [selected_zone]

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_retailer = st.multiselect("Select Retailer(s)", ["All"] + retailers["name"].tolist(), default=["All"], key="rep_retailer_filter")
    with col2:
        filter_category = st.multiselect("Select Category(s)", ["All"] + categories["name"].tolist(), default=["All"], key="rep_cat_filter")
    with col3:
        date_range = st.date_input("Date Range", value=[], key="rep_date_range")

    filtered_entries = entries.copy()
    filtered_entries = filter_by_zone(filtered_entries, "retailer_id", selected_zone)

    if selected_zone == "All Zones" and "All" not in zone_filter and not retailers.empty:
        rz = retailers.copy()
        rz["zone"] = rz["zone"].apply(_norm_zone)
        allowed_rids = rz.loc[rz["zone"].isin([_norm_zone(z) for z in zone_filter]), "retailer_id"].astype(int).tolist()
        filtered_entries = filtered_entries.loc[filtered_entries["retailer_id"].astype(int).isin(allowed_rids)].copy()

    if not filtered_entries.empty:
        if "All" not in filter_retailer and not retailers.empty:
            rid_list = retailers.loc[retailers["name"].isin(filter_retailer), "retailer_id"].astype(int).tolist()
            filtered_entries = filtered_entries.loc[filtered_entries["retailer_id"].astype(int).isin(rid_list)].copy()

        if "All" not in filter_category and not categories.empty:
            cid_list = categories.loc[categories["name"].isin(filter_category), "category_id"].astype(int).tolist()
            filtered_entries = filtered_entries.loc[filtered_entries["category_id"].astype(int).isin(cid_list)].copy()

        if len(date_range) == 2:
            filtered_entries["date"] = pd.to_datetime(filtered_entries["date"], errors="coerce")
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered_entries = filtered_entries.loc[(filtered_entries["date"] >= start) & (filtered_entries["date"] <= end)].copy()

    if filtered_entries.empty:
        st.info("No data matching the filters")
    else:
        result_view = build_entries_view_cached(filtered_entries, st.session_state["data_version"], want_milk_type_col=False)
        st.dataframe(
            result_view[["date", "zone", "Retailer", "Category", "qty", "rate", "amount"]],
            width="stretch",
        )

# ================== DISTRIBUTORS ==================
elif menu == "🚚 Distributors":
    st.header("🚚 Distributor Management")
    tab1, tab2 = st.tabs(["➕ Add Distributor", "✏️ Edit Distributors"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Distributor Name", key="dist_add_name")
            contact = st.text_input("Contact Number", key="dist_add_contact")
        with col2:
            address = st.text_area("Address", key="dist_add_address")

        if st.button("Add Distributor", type="primary", key="dist_add_btn"):
            if not name.strip():
                st.warning("Distributor name required.")
            else:
                did = sb_new_id("distributors", "distributor_id")
                new_row = pd.DataFrame(
                    [[did, name.strip(), contact.strip(), address.strip(), True]],
                    columns=CSV_SCHEMAS[DISTRIBUTORS_FILE]
                )
                distributors = pd.concat([distributors, new_row], ignore_index=True)
                sb_insert_df(new_row, DISTRIBUTORS_FILE)
                st.success("✅ Distributor added!")
                st.rerun()

    with tab2:
        if distributors.empty:
            st.info("No distributors yet.")
        else:
            st.dataframe(df_for_display(distributors), width="stretch")

            edit_dis = st.selectbox("Select distributor", distributors["name"].tolist(), key="dist_edit_sel")
            dis_data = distributors.loc[distributors["name"] == edit_dis].iloc[0]
            did = int(dis_data["distributor_id"])

            col1, col2, col3 = st.columns(3)
            with col1:
                new_name = st.text_input("Name", value=str(dis_data["name"]), key="dist_new_name")
            with col2:
                new_contact = st.text_input("Contact", value=str(dis_data.get("contact", "")), key="dist_new_contact")
            with col3:
                new_active = st.checkbox("Active", value=bool(dis_data.get("is_active", True)), key="dist_new_active")

            new_address = st.text_area("Address", value=str(dis_data.get("address", "")), key="dist_new_address")

            colA, colB, colC = st.columns(3)
            
            with colA:
                if st.button("Update Distributor", key="dist_update_btn"):
                    mask = distributors["distributor_id"].astype(int) == int(did)
                    distributors.loc[mask, ["name", "contact", "address", "is_active"]] = [
                        new_name.strip(),
                        new_contact.strip(),
                        new_address.strip(),
                        bool(new_active),
                    ]
                    
                    updated_row = distributors.loc[mask].copy()
                    safe_write_csv(updated_row, DISTRIBUTORS_FILE, allow_empty=False)
                    
                    st.success("Updated!")
                    st.rerun()

            with colB:
                if st.button("Deactivate (Safe)", key="dist_deactivate_btn"):
                    mask = distributors["distributor_id"].astype(int) == int(did)
                    distributors.loc[mask, "is_active"] = False

                    updated_row = distributors.loc[mask].copy()
                    safe_write_csv(updated_row, DISTRIBUTORS_FILE, allow_empty=False)

                    st.success("Distributor deactivated.")
                    st.rerun()



            with colC:
                confirm = st.text_input("Type DELETE to hard delete", key="dist_delete_confirm")
                if st.button("🗑️ Hard Delete", type="secondary", key="dist_delete_btn"):
                    if confirm != "DELETE":
                        st.warning("Type DELETE to confirm.")
                    elif is_distributor_referenced(did):
                        st.error("Blocked: Distributor referenced in purchases/payments. Deactivate instead.")
                    else:
                        sb_delete_by_pk("distributors", "distributor_id", [did])
                        st.success("Hard deleted.")
                        st.rerun()


elif menu == "🧩 Distributor Category Mapping":
    st.header("🧩 Distributor → Category Mapping")
    st.caption("This controls exactly which categories appear for each distributor in the Daily Posting Sheet.")

    if distributors.empty or categories.empty:
        st.warning("Add distributors and categories first.")
        st.stop()

    distributors_active = distributors.copy()
    if "is_active" in distributors_active.columns:
        distributors_active["is_active"] = distributors_active["is_active"].apply(parse_boolish_active)
        distributors_active = distributors_active.loc[distributors_active["is_active"] == True].copy()

    categories_active = categories.copy()
    if "is_active" in categories_active.columns:
        categories_active["is_active"] = categories_active["is_active"].apply(parse_boolish_active)
        categories_active = categories_active.loc[categories_active["is_active"] == True].copy()

    if distributors_active.empty:
        st.info("No active distributors.")
        st.stop()
    if categories_active.empty:
        st.info("No active categories.")
        st.stop()

    # Helper: current mapped category ids
    def _mapped_ids(did: int) -> list[int]:
        if dist_cat_map is None or dist_cat_map.empty:
            return []
        m = dist_cat_map.copy()
        m["distributor_id"] = pd.to_numeric(m["distributor_id"], errors="coerce").fillna(0).astype(int)
        m["category_id"] = pd.to_numeric(m["category_id"], errors="coerce").fillna(0).astype(int)
        m["is_active"] = m.get("is_active", True).apply(parse_boolish_active)
        m = m.loc[(m["distributor_id"] == int(did)) & (m["is_active"] == True)].copy()
        return sorted(m["category_id"].astype(int).tolist())

    # UI
    did_list = distributors_active.sort_values("name")[["distributor_id", "name"]].to_dict(orient="records")
    cat_options = categories_active.sort_values("name")[["category_id", "name"]].to_dict(orient="records")
    cat_id_to_name = {int(c["category_id"]): str(c["name"]) for c in cat_options}
    cat_name_to_id = {str(c["name"]): int(c["category_id"]) for c in cat_options}
    all_cat_names = [str(c["name"]) for c in cat_options]

    st.divider()
    st.subheader("Set categories per distributor")

    for d in did_list:
        did = int(d["distributor_id"])
        dname = str(d["name"])

        current_ids = _mapped_ids(did)
        current_names = [cat_id_to_name[cid] for cid in current_ids if cid in cat_id_to_name]

        with st.expander(f"🚚 {dname}", expanded=False):
            selected_names = st.multiselect(
                "Supplies these categories",
                options=all_cat_names,
                default=current_names,
                key=f"dcm_ms_{did}",
            )

            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("💾 Save Mapping", key=f"dcm_save_{did}"):
                    # Overwrite mapping for this distributor (simple + correct)
                    sb_delete_where("distributor_category_map", [("distributor_id", "eq", int(did))])

                    new_rows = []
                    next_id = None if USE_DB_IDS else sb_next_id("distributor_category_map", "map_id")

                    for nm in selected_names:
                        cid = int(cat_name_to_id[nm])
                        mid = None if USE_DB_IDS else int(next_id)
                        if not USE_DB_IDS:
                            next_id += 1
                        new_rows.append([mid, int(did), int(cid), True])

                    if new_rows:
                        df_new = pd.DataFrame(new_rows, columns=CSV_SCHEMAS[DISTRIBUTOR_CATEGORY_MAP_FILE])
                        sb_insert_df(df_new, DISTRIBUTOR_CATEGORY_MAP_FILE)

                    st.success("Saved ✅")
                    st.rerun()

            with col2:
                st.caption("Tip: Keep this strict. If you don’t map it here, it won’t show in Daily Entry for this distributor.")

# ================== DISTRIBUTOR LEDGER ==================
elif menu == "📒 Distributor Ledger":
    st.header("📒 Distributor Ledger (Incoming Milk + Payments + Running Due)")

    if distributors.empty:
        st.warning("Add at least 1 distributor first.")
        st.stop()
    if categories.empty:
        st.warning("Add at least 1 category first.")
        st.stop()

    dis_name = st.selectbox("Select Distributor", distributors["name"].tolist(), key="dl_dis")
    did = int(distributors.loc[distributors["name"] == dis_name, "distributor_id"].iloc[0])

    colA, colB = st.columns(2)
    with colA:
        start_day = st.date_input("From Date", value=date.today().replace(day=1), key="dl_start")
    with colB:
        end_day = st.date_input("To Date", value=date.today(), key="dl_end")

    if start_day > end_day:
        st.error("From Date cannot be after To Date.")
        st.stop()

    cat_names = categories["name"].dropna().astype(str).tolist()
    cat_names = sorted(list(dict.fromkeys(cat_names)))

    opening_due = distributor_balance_before(did, start_day)
    grid = build_distributor_daily_grid(did, start_day, end_day, cat_names)

    closing_due = float(pd.to_numeric(grid["Running Due (₹)"], errors="coerce").fillna(opening_due).iloc[-1]) if not grid.empty else opening_due
    total_milk = float(pd.to_numeric(grid["Total Milk (L)"], errors="coerce").fillna(0).sum()) if not grid.empty else 0.0
    total_pur = float(pd.to_numeric(grid["Purchases (₹)"], errors="coerce").fillna(0).sum()) if not grid.empty else 0.0
    total_pay = float(pd.to_numeric(grid["Payment (₹)"], errors="coerce").fillna(0).sum()) if not grid.empty else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Milk (L)", f"{total_milk:.2f}")
    c2.metric("Opening Due", _fmt_money(opening_due))
    c3.metric("Purchases (This Period)", _fmt_money(total_pur))
    c4.metric("Payments (This Period)", _fmt_money(total_pay))
    c5.metric("Closing Due", _fmt_money(closing_due))

    st.divider()
    st.subheader("📌 Daily Sheet (Qty + Rate by Category)")

    preview = grid.copy()
    for cat in cat_names:
        qcol = f"{cat} Qty"
        rcol = f"{cat} Rate"
        if qcol in preview.columns:
            preview[qcol] = preview[qcol].apply(_disp_2dec_or_dash)
        if rcol in preview.columns:
            preview[rcol] = preview[rcol].apply(_disp_rate_or_dash)
            
    for c in ["Purchases (₹)", "Payment (₹)", "Running Due (₹)"]:
        if c in preview.columns:
            preview[c] = preview[c].apply(_fmt_money)
            
    if "Total Milk (L)" in preview.columns:
        preview["Total Milk (L)"] = preview["Total Milk (L)"].apply(_disp_2dec_or_dash)


    st.dataframe(df_for_display(preview), width="stretch")

    st.subheader("💳 Payment Mode Totals (Period)")
    pm = distributor_pay_mode_totals(did, start_day, end_day)
    if pm.empty:
        st.info("No payments in this period.")
    else:
        st.dataframe(pm.style.format({"Total (₹)": "₹{:.2f}"}), width="stretch")


# ================== DISTRIBUTOR BILL ==================
elif menu == "🧾 Distributor Bill":
    st.header("🧾 Distributor Statement / Bill (Printable)")

    if distributors.empty:
        st.warning("Add at least 1 distributor first.")
        st.stop()
    if categories.empty:
        st.warning("Add at least 1 category first.")
        st.stop()

    dis_name = st.selectbox("Select Distributor", distributors["name"].tolist(), key="db_dis")
    did = int(distributors.loc[distributors["name"] == dis_name, "distributor_id"].iloc[0])

    colA, colB = st.columns(2)
    with colA:
        start_day = st.date_input("From Date", value=date.today().replace(day=1), key="db_start")
    with colB:
        end_day = st.date_input("To Date", value=date.today(), key="db_end")

    if start_day > end_day:
        st.error("From Date cannot be after To Date.")
        st.stop()

    cat_names = categories["name"].dropna().astype(str).tolist()
    cat_names = sorted(list(dict.fromkeys(cat_names)))

    drow = distributors.loc[distributors["distributor_id"].astype(int) == did].iloc[0].to_dict()

    grid = build_distributor_daily_grid(did, start_day, end_day, cat_names)
    pm = distributor_pay_mode_totals(did, start_day, end_day)

    opening_due = distributor_balance_before(did, start_day)
    closing_due = float(pd.to_numeric(grid["Running Due (₹)"], errors="coerce").fillna(opening_due).iloc[-1]) if not grid.empty else opening_due

    total_pur = float(pd.to_numeric(grid["Purchases (₹)"], errors="coerce").fillna(0).sum()) if not grid.empty else 0.0
    total_pay = float(pd.to_numeric(grid["Payment (₹)"], errors="coerce").fillna(0).sum()) if not grid.empty else 0.0
    total_milk = float(pd.to_numeric(grid["Total Milk (L)"], errors="coerce").fillna(0).sum()) if not grid.empty else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Milk (L)", f"{total_milk:.2f}")
    c2.metric("Opening Due", _fmt_money(opening_due))
    c3.metric("Purchases (This Period)", _fmt_money(total_pur))
    c4.metric("Payments (This Period)", _fmt_money(total_pay))
    c5.metric("Closing Due", _fmt_money(closing_due))

    st.divider()
    st.subheader("📌 Bill Preview (Arrow-safe)")
    preview = grid.copy()

    for cat in cat_names:
        qcol = f"{cat} Qty"
        rcol = f"{cat} Rate"
        
        if qcol in preview.columns:
            preview[qcol] = preview[qcol].apply(_disp_2dec_or_dash)
        if rcol in preview.columns:
            preview[rcol] = preview[rcol].apply(_disp_rate_or_dash)
            
    for c in ["Purchases (₹)", "Payment (₹)", "Running Due (₹)"]:
        
        if c in preview.columns:
            preview[c] = preview[c].apply(_fmt_money)
            
    if "Total Milk (L)" in preview.columns:
        preview["Total Milk (L)"] = preview["Total Milk (L)"].apply(_disp_2dec_or_dash)


    st.dataframe(df_for_display(preview), width="stretch")

    html = build_distributor_bill_html(drow, start_day, end_day, grid, pm, cat_names)

    st.subheader("🖨️ Printable Statement")
    st.components.v1.html(html, height=750, scrolling=True)

    st.download_button(
        "⬇️ Download Distributor Statement (HTML - Print Ready)",
        data=html.encode("utf-8"),
        file_name=f"distributor_statement_{dis_name}_{start_day}_to_{end_day}.html",
        mime="text/html",
        key="db_dl_html",
    )




# ================== EXPENSES ==================
elif menu == "💼 Expenses":
    st.header("💼 Business Expenses Management")

    tab1, tab2 = st.tabs(["➕ Add Expense", "📋 View / Edit Expenses"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ex_date = st.date_input("Date", value=date.today(), key="ex_date")
        with c2:
            ex_cat = st.text_input("Expense Category", value="", key="ex_cat")
        with c3:
            ex_amt = st.number_input("Amount (₹)", min_value=0.0, step=50.0, format="%g", key="ex_amt")
        with c4:
            ex_mode = st.selectbox("Payment Mode", ["Cash", "UPI", "Bank", "Cheque", "Other"], key="ex_mode")

        ex_desc = st.text_input("Description", key="ex_desc")
        ex_paid = st.checkbox("Paid", value=True, key="ex_paid")

        if st.button("Save Expense", type="primary", key="ex_save"):
            if ex_amt <= 0:
                st.error("Amount must be > 0.")
                st.stop()
            eid = sb_new_id("expenses", "expense_id")
            new_row = pd.DataFrame(
                [[eid, str(ex_date), str(ex_cat).strip(), str(ex_desc).strip(), float(ex_amt), str(ex_mode), bool(ex_paid)]],
                columns=CSV_SCHEMAS[EXPENSES_FILE],
            )
            sb_insert_df(new_row, EXPENSES_FILE)
            st.success("✅ Expense saved.")
            st.rerun()

    with tab2:
        if expenses.empty:
            st.info("No expenses yet.")
        else:
            view = expenses.copy()
            view["date"] = _safe_dt(view["date"]).dt.strftime("%Y-%m-%d")
            view = view[["expense_id", "date", "category", "description", "amount", "payment_mode", "paid"]].sort_values(
                ["date", "expense_id"], ascending=[False, False]
            )
            st.dataframe(view.style.format({"amount": "₹{:.2f}"}), width="stretch")

            st.divider()
            st.subheader("✏️ Edit / Delete Expense")

            eid = st.number_input("Expense ID", min_value=1, step=1, key="ex_edit_id")
            if int(eid) not in set(expenses["expense_id"].astype(int).tolist()):
                st.caption("Enter an existing Expense ID to edit.")
            else:
                row = expenses.loc[expenses["expense_id"].astype(int) == int(eid)].iloc[0]
                cur_date = pd.to_datetime(row["date"], errors="coerce").date() if str(row["date"]) else date.today()
                cur_cat = str(row.get("category", "") or "")
                cur_desc = str(row.get("description", "") or "")
                cur_amt = float(row.get("amount", 0.0) or 0.0)
                cur_mode = str(row.get("payment_mode", "Cash") or "Cash")
                cur_paid = bool(row.get("paid", False))

                e1, e2, e3, e4 = st.columns(4)
                with e1:
                    new_date = st.date_input("Date", value=cur_date, key="ex_new_date")
                with e2:
                    new_cat = st.text_input("Category", value=cur_cat, key="ex_new_cat")
                with e3:
                    new_amt = st.number_input("Amount (₹)", min_value=0.0, step=50.0, format="%g", value=cur_amt, key="ex_new_amt")
                with e4:
                    modes = ["Cash", "UPI", "Bank", "Cheque", "Other"]
                    idx = modes.index(cur_mode) if cur_mode in modes else 0
                    new_mode = st.selectbox("Mode", modes, index=idx, key="ex_new_mode")

                new_desc = st.text_input("Description", value=cur_desc, key="ex_new_desc")
                new_paid = st.checkbox("Paid", value=cur_paid, key="ex_new_paid")

                colA, colB = st.columns(2)
                with colA:
                    if st.button("Update Expense", key="ex_update"):
                        if new_amt <= 0:
                            st.error("Amount must be > 0.")
                            st.stop()
                            
                        mask = expenses["expense_id"].astype(int) == int(eid)
                        
                        expenses.loc[mask, ["date", "category", "description", "amount", "payment_mode", "paid"]] = [
                            str(new_date),
                            str(new_cat).strip(),
                            str(new_desc).strip(),
                            float(new_amt),
                            str(new_mode),
                            bool(new_paid),
                        ]
                        
                        updated = expenses.loc[mask].copy()
                        safe_write_csv(updated, EXPENSES_FILE, allow_empty=False)
                        st.success("Updated.")
                        st.rerun()


                with colB:
                    confirm = st.text_input("Type DELETE to delete expense", key="ex_del_confirm")
                    if st.button("Delete Expense", key="ex_delete"):
                        if confirm != "DELETE":
                            st.warning("Type DELETE to confirm.")
                        else:
                            sb_delete_by_pk("expenses", "expense_id", [int(eid)])
                            st.success("Deleted.")
                            st.rerun()

# ================== GENERATE BILL ==================
elif menu == "🧾 Generate Bill":
    st.header("🧾 Generate Professional Customer Bill (Printable + Downloadable)")

    if retailers.empty:
        st.warning("Add at least 1 retailer first.")
        st.stop()

    bill_zone = st.selectbox("Select Zone", ["All Zones"] + get_all_zones(), key="bill_zone")

    rlist = retailers.copy()
    rlist["zone"] = rlist["zone"].apply(_norm_zone)
    if bill_zone != "All Zones":
        rlist = rlist.loc[rlist["zone"] == _norm_zone(bill_zone)].copy()

    if rlist.empty:
        st.warning("No retailers found in this zone.")
        st.stop()

    retailer_name = st.selectbox("Select Customer (Retailer)", rlist["name"].tolist(), key="bill_retailer")
    rid = int(rlist.loc[rlist["name"] == retailer_name, "retailer_id"].iloc[0])

    colA, colB = st.columns(2)
    with colA:
        start_day = st.date_input("From Date", value=date.today().replace(day=1), key="bill_start")
    with colB:
        end_day = st.date_input("To Date", value=date.today(), key="bill_end")

    if start_day > end_day:
        st.error("From Date cannot be after To Date.")
        st.stop()

    cat_df = categories_active.copy() if not categories_active.empty else categories.copy()
    cat_names = cat_df["name"].dropna().astype(str).tolist()
    cat_names = sorted(list(dict.fromkeys(cat_names)))

    if not cat_names:
        st.error("No categories found.")
        st.stop()

    rrow = retailers.loc[retailers["retailer_id"].astype(int) == rid].iloc[0].to_dict()
    rrow["zone"] = _norm_zone(rrow.get("zone", "Default"))

    grid = build_bill_daily_grid(rid, start_day, end_day, cat_names)

    opening_due = retailer_balance_before(rid, start_day)
    closing_due = float(pd.to_numeric(grid["Running Due (₹)"], errors="coerce").fillna(opening_due).iloc[-1]) if not grid.empty else float(opening_due)

    if USE_SERVER_FILTERS:
        p = sb_fetch_df(
            PAYMENTS_FILE,
            CSV_SCHEMAS[PAYMENTS_FILE],
            filters=[
                ("retailer_id", "eq", int(rid)),
                ("date", "gte", str(start_day)),
                ("date", "lte", str(end_day)),
            ],
        )
    else:
        p = payments.copy()
        if not p.empty:
            p["date"] = _safe_dt(p["date"]).dt.date
            p = p.loc[
                (p["retailer_id"].astype(int) == rid)
                & (p["date"] >= start_day)
                & (p["date"] <= end_day)
            ].copy()

    if p.empty:
        pay_mode_totals = pd.DataFrame(columns=["Mode", "Total (₹)"])
    else:
        p["payment_mode"] = p["payment_mode"].fillna("Cash").astype(str)
        pay_mode_totals = (
            p.groupby("payment_mode", as_index=False)["amount"]
             .sum()
             .sort_values("amount", ascending=False)
             .rename(columns={"payment_mode": "Mode", "amount": "Total (₹)"})
        )

    total_sales = float(pd.to_numeric(grid["Sales (₹)"], errors="coerce").fillna(0).sum()) if not grid.empty else 0.0
    total_pay = float(pd.to_numeric(grid["Payment (₹)"], errors="coerce").fillna(0).sum()) if not grid.empty else 0.0
    total_milk = float(pd.to_numeric(grid["Total Milk (L)"], errors="coerce").fillna(0).sum()) if not grid.empty else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Milk (L)", f"{total_milk:.2f}")
    c2.metric("Opening Due", _fmt_money(opening_due))
    c3.metric("Sales (This Period)", _fmt_money(total_sales))
    c4.metric("Payments (This Period)", _fmt_money(total_pay))
    c5.metric("Closing Due", _fmt_money(closing_due))

    st.divider()

    st.subheader("📌 Bill Preview (Arrow-safe)")

    preview = grid.copy()
    
    for cat in cat_names:
        qcol = f"{cat} Qty"
        rcol = f"{cat} Rate"
        
        if qcol in preview.columns:
            preview[qcol] = preview[qcol].apply(_disp_2dec_or_dash)
            
        if rcol in preview.columns:
            preview[rcol] = preview[rcol].apply(_disp_rate_or_dash)
            
    for c in ["Sales (₹)", "Payment (₹)", "Running Due (₹)"]:
        if c in preview.columns:
            preview[c] = preview[c].apply(_fmt_money)
            
    if "Total Milk (L)" in preview.columns:
        preview["Total Milk (L)"] = preview["Total Milk (L)"].apply(_disp_2dec_or_dash)

    st.dataframe(df_for_display(preview), width="stretch")

    st.subheader("💳 Payment Mode Totals (Period)")
    if pay_mode_totals.empty:
        st.info("No payments in this period.")
    else:
        st.dataframe(pay_mode_totals.style.format({"Total (₹)": "₹{:.2f}"}), width="stretch")

    # --- Print ONLY categories that were purchased (qty > 0 anywhere in the period) ---
    purchased_cats = []
    if grid is not None and not grid.empty:
        for cat in cat_names:
            qcol = f"{cat} Qty"
            if qcol not in grid.columns:
                continue
            # grid can contain numbers or "-" strings, so coerce safely
            qty_series = pd.to_numeric(grid[qcol], errors="coerce").fillna(0.0)
            if (qty_series > 0).any():
                purchased_cats.append(cat)
                
    # If nothing detected (edge case), fallback to original cat_names so bill isn't empty/broken
    cats_for_print = purchased_cats if purchased_cats else cat_names
    
    html = build_bill_html(rrow, start_day, end_day, grid, pay_mode_totals, cats_for_print, opening_due)


    st.subheader("🖨️ Printable Bill")
    st.components.v1.html(html, height=750, scrolling=True)

    st.caption("To print: click the Print button in the bill OR download HTML and press Ctrl+P.")
    st.download_button(
        "⬇️ Download Bill (HTML - Print Ready)",
        data=html.encode("utf-8"),
        file_name=f"bill_{retailer_name}_{start_day}_to_{end_day}.html",
        mime="text/html",
        key="bill_dl_html",
    )

    pdf_bytes = bill_pdf_bytes_from_html(html)
    if pdf_bytes is not None:
        st.download_button(
            "⬇️ Download Bill (PDF)",
            data=pdf_bytes,
            file_name=f"bill_{retailer_name}_{start_day}_to_{end_day}.pdf",
            mime="application/pdf",
            key="bill_dl_pdf",
        )

# ================== DATA HEALTH & BACKUP ==================
elif menu == "🛡️ Data Health & Backup":
    st.header("🛡️ Data Health & Backup")

    st.subheader("Backups")
    st.caption("Download a ZIP containing CSV exports of all tables currently in the app.")
    dv = st.session_state.get("data_version", 0)
    # Keep bytes stable across reruns to avoid duplicate downloads in some browsers.
    if st.session_state.get("backup_zip_version") != dv or "backup_zip_bytes" not in st.session_state:
        st.session_state["backup_zip_bytes"] = make_full_backup_zip(dv)
        st.session_state["backup_zip_version"] = dv
        
    zip_bytes = st.session_state["backup_zip_bytes"]
    
    st.download_button(
        "⬇️ Download Full Backup ZIP (CSV)",
        data=zip_bytes,
        file_name=f"milk_accounting_backup_{date.today().isoformat()}.zip",
        mime="application/zip",
        key=f"backup_zip_{dv}",
        width="stretch",
    )


    st.divider()
    st.subheader("Integrity Checks")

    issues = []

    if not entries.empty and not retailers.empty:
        known_r = set(retailers["retailer_id"].astype(int))
        bad = entries.loc[~entries["retailer_id"].astype(int).isin(known_r)][["entry_id", "retailer_id", "date", "amount"]]
        if not bad.empty:
            issues.append(("Orphan retailer_id in entries", bad))

    if not payments.empty and not retailers.empty:
        known_r = set(retailers["retailer_id"].astype(int))
        bad = payments.loc[~payments["retailer_id"].astype(int).isin(known_r)][["payment_id", "retailer_id", "date", "amount"]]
        if not bad.empty:
            issues.append(("Orphan retailer_id in payments", bad))

    if not entries.empty and not categories.empty:
        known_c = set(categories["category_id"].astype(int))
        bad = entries.loc[~entries["category_id"].astype(int).isin(known_c)][["entry_id", "category_id", "date", "amount"]]
        if not bad.empty:
            issues.append(("Orphan category_id in entries", bad))

    if not entries.empty:
        bad = entries.loc[(entries["qty"] < 0) | (entries["rate"] < 0) | (entries["amount"] < 0)]
        if not bad.empty:
            issues.append(("Negative values in entries", bad))

    if not payments.empty:
        bad = payments.loc[payments["amount"] < 0]
        if not bad.empty:
            issues.append(("Negative payments", bad))

    for name, df, idcol in [
        ("entries", entries, "entry_id"),
        ("payments", payments, "payment_id"),
        ("prices", prices, "price_id"),
        ("retailers", retailers, "retailer_id"),
        ("categories", categories, "category_id"),
        ("distributors", distributors, "distributor_id"),
    ]:
        if not df.empty and idcol in df.columns:
            dup = df[df[idcol].duplicated(keep=False)].sort_values(idcol)
            if not dup.empty:
                issues.append((f"Duplicate IDs in {name} ({idcol})", dup))

    if not entries.empty:
        chk = entries.copy()
        chk["calc"] = (chk["qty"].astype(float) * chk["rate"].astype(float)).round(2)
        chk["amount_r"] = chk["amount"].astype(float).round(2)
        bad = chk.loc[(chk["qty"] > 0) & (chk["calc"] != chk["amount_r"])][["entry_id", "date", "qty", "rate", "amount", "calc"]]
        if not bad.empty:
            issues.append(("Entries amount mismatch (qty*rate != amount)", bad))

    if not issues:
        st.success("✅ No integrity problems found.")
    else:
        st.error(f"⚠️ Found {len(issues)} integrity issue group(s). Fix before trusting reports.")
        for title, df in issues:
            st.subheader(title)
            st.dataframe(df, width="stretch")
            st.divider()
  
