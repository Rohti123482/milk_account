import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
import os
import tempfile
import time
import shutil
import zipfile
import io
import plotly.express as px
from pandas.errors import EmptyDataError, ParserError

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Milk Accounting Pro", layout="wide", initial_sidebar_state="expanded")

# ================== CUSTOM CSS ==================
st.markdown(
    """
<style>
    .main {background-color: #f8f9fa;}
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {color: #555 !important; font-weight: 600 !important;}
    .stMetric [data-testid="stMetricValue"] {color: #1f1f1f !important; font-size: 1.5rem !important;}
    .stMetric [data-testid="stMetricDelta"] {color: #666 !important;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {background-color: #45a049;}
    h1 {color: #2c3e50 !important;}
    h2 {color: #34495e !important;}
    h3 {color: #7f8c8d !important;}
    .stDataFrame {background-color: white;}
    div[data-testid="stDataFrame"] * {color: #1f1f1f !important;}
    @media (max-width: 768px) {
        .stMetric {margin-bottom: 10px;}
        .stMetric label {font-size: 0.9rem !important;}
        .stMetric [data-testid="stMetricValue"] {font-size: 1.2rem !important;}
    }
</style>
""",
    unsafe_allow_html=True,
)

# ================== FILE SETUP ==================
os.makedirs("data", exist_ok=True)
BACKUP_DIR = "data/_backups"
EXPORT_DIR = "data/_exports"
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

RETAILERS_FILE = "data/retailers.csv"
CATEGORIES_FILE = "data/categories.csv"
PRICES_FILE = "data/prices.csv"
ENTRIES_FILE = "data/entries.csv"
PAYMENTS_FILE = "data/payments.csv"
DISTRIBUTORS_FILE = "data/distributors.csv"
DISTRIBUTOR_PURCHASES_FILE = "data/distributor_purchases.csv"
DISTRIBUTOR_PAYMENTS_FILE = "data/distributor_payments.csv"
WASTAGE_FILE = "data/wastage.csv"
EXPENSES_FILE = "data/expenses.csv"

GLOBAL_RETAILER_ID = -1

# ================== SAFETY SETTINGS ==================
BACKUPS_PER_FILE = 10
LOCK_RETRIES = 60
LOCK_SLEEP_SEC = 0.15
STALE_LOCK_SEC = 180  # consider lock stale after 3 minutes (single-user machine safety)

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
def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _lock_path(path: str) -> str:
    return os.path.abspath(path) + ".lock"

def _lock_is_stale(lp: str) -> bool:
    try:
        mtime = os.path.getmtime(lp)
        return (time.time() - mtime) > STALE_LOCK_SEC
    except OSError:
        return False

def acquire_lock(path: str) -> None:
    path_abs = os.path.abspath(path)
    lp = _lock_path(path_abs)

    for _ in range(LOCK_RETRIES):
        try:
            fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={datetime.now().isoformat()}".encode("utf-8"))
            os.close(fd)
            return
        except FileExistsError:
            # stale lock cleanup (single-user safety)
            if _lock_is_stale(lp):
                try:
                    os.remove(lp)
                    st.warning(f"⚠️ Removed stale lock: {os.path.basename(lp)}")
                    continue
                except OSError:
                    pass
            time.sleep(LOCK_SLEEP_SEC)

    raise TimeoutError(f"Could not acquire lock for {path_abs}. Close other programs using CSVs and retry.")

def release_lock(path: str) -> None:
    lp = _lock_path(path)
    try:
        if os.path.exists(lp):
            os.remove(lp)
    except OSError:
        pass

def ensure_csv_if_missing(path: str, columns: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")

def quarantine_corrupt_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    qpath = f"{path}.CORRUPT_{_ts()}.bak"
    try:
        os.replace(path, qpath)
    except OSError:
        shutil.copy2(path, qpath)
        try:
            os.remove(path)
        except OSError:
            pass
    return qpath

def rotate_backups_for(base_name: str) -> None:
    backups = []
    for fn in os.listdir(BACKUP_DIR):
        if fn.startswith(base_name + ".") and fn.endswith(".bak"):
            backups.append(os.path.join(BACKUP_DIR, fn))
    backups.sort(reverse=True)
    for p in backups[BACKUPS_PER_FILE:]:
        try:
            os.remove(p)
        except OSError:
            pass

def backup_before_write(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    base = os.path.basename(path)
    bname = f"{base}.{_ts()}.bak"
    bpath = os.path.join(BACKUP_DIR, bname)
    try:
        shutil.copy2(path, bpath)
        rotate_backups_for(base)
    except Exception:
        pass

def safe_write_csv(df: pd.DataFrame, path: str, retries: int = 25, delay_sec: float = 0.20) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    path_abs = os.path.abspath(path)

    acquire_lock(path_abs)
    try:
        last_err = None
        for attempt in range(retries):
            fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".csv", dir=folder)
            os.close(fd)
            try:
                df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
                backup_before_write(path_abs)

                try:
                    os.replace(tmp_path, path_abs)
                    return
                except PermissionError as e:
                    last_err = e
                    time.sleep(delay_sec * (1 + attempt * 0.15))
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
            except Exception as e:
                last_err = e
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                time.sleep(delay_sec * (1 + attempt * 0.15))

        raise PermissionError(
            f"Could not write '{path_abs}'. Close anything using it (Excel/preview/antivirus)."
        ) from last_err
    finally:
        release_lock(path_abs)

def safe_write_two_csvs(df1: pd.DataFrame, path1: str, df2: pd.DataFrame, path2: str) -> None:
    # avoid deadlocks by consistent lock order
    p1 = os.path.abspath(path1)
    p2 = os.path.abspath(path2)
    ordered = sorted([(p1, df1), (p2, df2)], key=lambda x: x[0])
    acquired = []
    try:
        for p, _df in ordered:
            acquire_lock(p)
            acquired.append(p)
        # write sequentially with temp+replace while holding locks (no interleaving)
        for p, _df in ordered:
            folder = os.path.dirname(p)
            os.makedirs(folder, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".csv", dir=folder)
            os.close(fd)
            try:
                _df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
                backup_before_write(p)
                os.replace(tmp_path, p)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
    finally:
        for p in reversed(acquired):
            release_lock(p)

def safe_read_csv(path: str, required_columns: list[str]) -> pd.DataFrame:
    ensure_csv_if_missing(path, required_columns)

    # empty (0 bytes) is treated as corruption: quarantine + recreate
    if os.path.exists(path) and os.path.getsize(path) == 0:
        q = quarantine_corrupt_file(path)
        pd.DataFrame(columns=required_columns).to_csv(path, index=False, encoding="utf-8-sig")
        st.error(f"⚠️ Empty file detected and quarantined: {os.path.basename(q)}")
        return pd.DataFrame(columns=required_columns)

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df is None:
            df = pd.DataFrame(columns=required_columns)
    except (EmptyDataError, ParserError, UnicodeDecodeError, ValueError):
        q = quarantine_corrupt_file(path)
        pd.DataFrame(columns=required_columns).to_csv(path, index=False, encoding="utf-8-sig")
        st.error(f"⚠️ Corrupted file detected and quarantined: {os.path.basename(q)}")
        return pd.DataFrame(columns=required_columns)

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        allowed = ALLOWED_LEGACY_MISSING.get(path, set())
        if set(missing).issubset(allowed):
            for c in missing:
                df[c] = pd.NA
        else:
            q = quarantine_corrupt_file(path)
            pd.DataFrame(columns=required_columns).to_csv(path, index=False, encoding="utf-8-sig")
            st.error(f"⚠️ Bad schema detected and quarantined: {os.path.basename(q)}")
            return pd.DataFrame(columns=required_columns)

    extras = [c for c in df.columns if c not in required_columns]
    df = df[required_columns + extras].copy()
    return df

def parse_boolish_active(x) -> bool:
    # legacy blanks should default to True (so you don't hide all retailers/categories)
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return True
    s = str(x).strip().lower()
    if s == "":
        return True
    return s in ("true", "1", "yes", "y")

def parse_boolish_paid(x) -> bool:
    # blanks default False for "paid"
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    s = str(x).strip().lower()
    if s == "":
        return False
    return s in ("true", "1", "yes", "y")

def _norm_zone(z: str) -> str:
    return str(z or "Default").strip() or "Default"

def build_entries_view(df_entries: pd.DataFrame, want_milk_type_col: bool = False) -> pd.DataFrame:
    """
    Returns a view of entries with guaranteed columns:
    - Retailer
    - Category (and optionally Milk Type)
    - zone
    Never depends on name_x/name_y.
    """
    if df_entries is None or df_entries.empty:
        out = df_entries.copy() if df_entries is not None else pd.DataFrame()
        for c in ["Retailer", "Category", "Milk Type", "zone"]:
            if c not in out.columns:
                out[c] = pd.Series(dtype="object")
        return out

    out = df_entries.copy()

    # retailer join (adds name + zone)
    rmap = retailers[["retailer_id", "name", "zone"]].copy() if not retailers.empty else pd.DataFrame(columns=["retailer_id", "name", "zone"])
    out = out.merge(rmap, on="retailer_id", how="left", suffixes=("", "_ret"))

    # retailer name
    if "Retailer" not in out.columns:
        if "name" in out.columns:
            out = out.rename(columns={"name": "Retailer"})
        elif "name_ret" in out.columns:
            out = out.rename(columns={"name_ret": "Retailer"})
        else:
            out["Retailer"] = ""

    # zone normalize
    if "zone" in out.columns:
        out["zone"] = out["zone"].apply(_norm_zone)
    else:
        out["zone"] = "Default"

    # category join
    cmap = categories[["category_id", "name"]].copy() if not categories.empty else pd.DataFrame(columns=["category_id", "name"])
    out = out.merge(cmap, on="category_id", how="left", suffixes=("", "_cat"))

    # category name
    if "name_cat" in out.columns:
        out = out.rename(columns={"name_cat": "Category"})
    elif "name" in out.columns and "Category" not in out.columns:
        # safe here because retailer name already renamed away from "name"
        out = out.rename(columns={"name": "Category"})
    else:
        if "Category" not in out.columns:
            out["Category"] = ""

    if "Category" not in out.columns:
        out["Category"] = ""

    if want_milk_type_col:
        out["Milk Type"] = out["Category"]

    return out


def next_id_from_df(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 1
    m = pd.to_numeric(df[col], errors="coerce").max()
    if pd.isna(m):
        return 1
    return int(m) + 1

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

    return retailers, categories, prices, entries, payments, distributors, dist_purchases, dist_payments, wastage, expenses


retailers, categories, prices, entries, payments, distributors, dist_purchases, dist_payments, wastage, expenses = load_and_migrate_data()

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
def _day_entries_for_zone(day: date, zone: str) -> pd.DataFrame:
    df = entries.copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.loc[df["date"] == day].copy()
    df = filter_by_zone(df, "retailer_id", zone)
    return df

def _day_payments_for_zone(day: date, zone: str) -> pd.DataFrame:
    df = payments.copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.loc[df["date"] == day].copy()
    df = filter_by_zone(df, "retailer_id", zone)
    return df

def build_daily_posting_grid(day: date, zone: str, retailers_active: pd.DataFrame, categories_active: pd.DataFrame):
    rz = retailers_active.copy()
    rz["zone"] = rz["zone"].apply(_norm_zone)
    if zone != "All Zones":
        rz = rz.loc[rz["zone"] == _norm_zone(zone)].copy()

    cats = categories_active.copy()
    cat_list = cats["name"].tolist()

    day_e = _day_entries_for_zone(day, zone)
    pivot_qty = pd.DataFrame()
    if not day_e.empty:
        e_view = build_entries_view(day_e, want_milk_type_col=False)
        pivot_qty = pd.pivot_table(
            e_view,
            index="Retailer",
            columns="Category",
            values="qty",
            aggfunc="sum",
            fill_value=0.0
        )


    day_p = _day_payments_for_zone(day, zone)
    pay_map, mode_map = {}, {}
    if not day_p.empty:
        tmp = day_p.merge(retailers[["retailer_id", "name"]], on="retailer_id", how="left").rename(columns={"name": "Retailer"})
        pay_map = tmp.groupby("Retailer")["amount"].sum().to_dict()
        for rn, sub in tmp.groupby("Retailer"):
            modes = sub["payment_mode"].dropna().astype(str).unique().tolist()
            if len(modes) == 1:
                mode_map[rn] = modes[0]
            elif len(modes) == 0:
                mode_map[rn] = "Cash"
            else:
                mode_map[rn] = "Mixed"

    rows = []
    for _, r in rz.iterrows():
        retailer_name = str(r["name"])
        rid = int(r["retailer_id"])
        row = {
            "ID": rid,
            "Retailer": retailer_name,
            "Today Payment ₹": float(pay_map.get(retailer_name, 0.0)),
            "Mode": str(mode_map.get(retailer_name, "Cash")),
        }
        for c in cat_list:
            qty = 0.0
            if not pivot_qty.empty and retailer_name in pivot_qty.index and c in pivot_qty.columns:
                qty = float(pivot_qty.loc[retailer_name, c])
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
    """
    Wide table:
    Row = Zone
    Columns = each Category (Liters)
    Adds TOTAL (L)
    """
    df = entries.copy()
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.loc[df["date"] == day].copy()
    if df.empty:
        return pd.DataFrame()

    # zone
    df = df.merge(retailers[["retailer_id", "zone"]], on="retailer_id", how="left")
    df["zone"] = df["zone"].apply(_norm_zone)

    # category name
    df = df.merge(categories[["category_id", "name"]], on="category_id", how="left").rename(columns={"name": "Category"})
    df["Category"] = df["Category"].fillna("").astype(str)

    # pivot
    pivot = pd.pivot_table(
        df,
        index="zone",
        columns="Category",
        values="qty",
        aggfunc="sum",
        fill_value=0.0,
    )

    # ensure stable ordering of zones
    pivot = pivot.sort_index()

    # totals
    pivot["TOTAL (L)"] = pivot.sum(axis=1)

    # display "-" for zeros later in UI; keep numeric here
    pivot = pivot.reset_index().rename(columns={"zone": "Zone"})
    return pivot


# ================== SIDEBAR: ZONE CONTEXT ==================
zones = get_all_zones()
selected_zone = st.sidebar.selectbox("🗺 Zone Context", ["All Zones"] + zones)

entries_z = filter_by_zone(entries, "retailer_id", selected_zone)
payments_z = filter_by_zone(payments, "retailer_id", selected_zone)

retailers_active = retailers.loc[retailers["is_active"] == True].copy() if "is_active" in retailers.columns else retailers.copy()
categories_active = categories.loc[categories["is_active"] == True].copy() if "is_active" in categories.columns else categories.copy()
distributors_active = distributors.loc[distributors["is_active"] == True].copy() if "is_active" in distributors.columns else distributors.copy()

# ================== UI ==================
st.title("🥛 JYOTIRLING MILK SUPPLIER")

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
        st.metric("Total Milk Sold", f"{total_milk:.2f} L", delta="Lifetime")
    with col2:
        st.metric("Total Sales", f"₹{total_sales:.2f}")
    with col3:
        st.metric("Total Payments", f"₹{total_payments:.2f}")
    with col4:
        st.metric("Outstanding", f"₹{outstanding:.2f}", delta=f"₹{outstanding:.2f}" if outstanding > 0 else "Settled")

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
            use_container_width=True,
        )
        if not zone_df.empty:
            fig = px.bar(zone_df, x="Zone", y="Sales (₹)")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

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

# ================== DAILY POSTING SHEET (EXCEL) ==================
elif menu == "📝 Daily Posting Sheet (Excel)":
    st.header("📝 Daily Posting Sheet (Excel-like) — Category wise + Zone wise")

    if retailers_active.empty or categories_active.empty:
        st.warning("⚠️ Please add active retailers and categories first")
        st.stop()

    posting_date = st.date_input("Posting Date", value=date.today(), key="posting_date")

    zone_choices = ["All Zones"] + get_all_zones()
    default_idx = zone_choices.index(selected_zone) if selected_zone in zone_choices else 0
    posting_zone = st.selectbox("Posting Zone", zone_choices, index=default_idx, key="posting_zone")

    grid_df, cat_list = build_daily_posting_grid(posting_date, posting_zone, retailers_active, categories_active)
    if grid_df.empty:
        st.warning("No retailers found for selected zone (check retailers.csv zone values).")
        st.stop()

    st.caption("Edit liters + today's payment. Saving overwrites ONLY this date + zone retailers.")

    edited = st.data_editor(
        grid_df,
        use_container_width=True,
        num_rows="fixed",
        key="daily_sheet_editor",
    )

    # validate prices first (no partial saves)
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
        today_pay = float(row.get("Today Payment ₹", 0.0) or 0.0)
        total_ledger = float(prev_ledger + today_sales - today_pay)
        prev_ledger_list.append(prev_ledger)
        today_sales_list.append(today_sales)
        total_ledger_list.append(total_ledger)

    preview = edited.copy()
    preview["Previous Ledger ₹"] = prev_ledger_list
    preview["Today Sales ₹"] = today_sales_list
    preview["Total Ledger ₹"] = total_ledger_list

    view = preview.copy()
    for c in cat_list:
        view[c] = view[c].apply(lambda x: "-" if float(x or 0.0) == 0.0 else float(x))

    st.subheader("📌 Preview")
    st.dataframe(view, use_container_width=True)

    totals = {}
    grand = 0.0
    for c in cat_list:
        s = preview[c].apply(lambda x: float(x or 0.0)).sum()
        totals[c] = float(s)
        grand += float(s)
    totals_df = pd.DataFrame([{"Category Totals": "TOTAL (L)", **totals, "GRAND TOTAL (L)": grand}])
    st.dataframe(totals_df, use_container_width=True)

    confirm = st.text_input("Type SAVE to confirm overwrite for this date+zone", key="sheet_save_confirm")
    if st.button("💾 Save Posting Sheet", type="primary"):
        if confirm != "SAVE":
            st.warning("Type SAVE to confirm.")
            st.stop()

        rz = retailers_active.copy()
        rz["zone"] = rz["zone"].apply(_norm_zone)
        if posting_zone != "All Zones":
            rz = rz.loc[rz["zone"] == _norm_zone(posting_zone)].copy()
        affected_rids = set(rz["retailer_id"].astype(int).tolist())

        if not affected_rids:
            st.error("No retailers found for selected zone. Fix zones in retailers.csv.")
            st.stop()

        e = entries.copy()
        p = payments.copy()
        if not e.empty:
            e["date"] = pd.to_datetime(e["date"], errors="coerce").dt.date
        if not p.empty:
            p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.date

        if not e.empty:
            e = e.loc[~((e["date"] == posting_date) & (e["retailer_id"].astype(int).isin(affected_rids)))].copy()
        if not p.empty:
            p = p.loc[~((p["date"] == posting_date) & (p["retailer_id"].astype(int).isin(affected_rids)))].copy()

        next_entry_id = next_id_from_df(e, "entry_id") - 1
        next_pay_id = next_id_from_df(p, "payment_id") - 1

        new_entries = []
        new_payments = []

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
                amount = qty * rate
                next_entry_id += 1
                new_entries.append([next_entry_id, str(posting_date), rid, cid, float(qty), float(rate), float(amount)])

            pay_amt = float(row.get("Today Payment ₹", 0.0) or 0.0)
            if pay_amt > 0:
                mode = str(row.get("Mode", "Cash") or "Cash")
                next_pay_id += 1
                new_payments.append([next_pay_id, str(posting_date), rid, float(pay_amt), mode, "Daily Posting Sheet"])

        if new_entries:
            e = pd.concat([e, pd.DataFrame(new_entries, columns=CSV_SCHEMAS[ENTRIES_FILE])], ignore_index=True)
        if new_payments:
            p = pd.concat([p, pd.DataFrame(new_payments, columns=CSV_SCHEMAS[PAYMENTS_FILE])], ignore_index=True)

        safe_write_two_csvs(e, ENTRIES_FILE, p, PAYMENTS_FILE)
        st.success("✅ Saved. Data will reflect in Date+Zone View and Zone-wise Summary.")
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
        e_view = build_entries_view(e_day, want_milk_type_col=False)

        pivot = pd.pivot_table(
            e_view,
            index="Retailer",
            columns="Category",
            values="qty",
            aggfunc="sum",
            fill_value=0.0
        )

        display = pivot.applymap(lambda x: "-" if float(x) == 0.0 else float(x))
        display["TOTAL (L)"] = pivot.sum(axis=1)

        totals = {c: float(pivot[c].sum()) for c in pivot.columns}
        totals["TOTAL (L)"] = float(pivot.values.sum())
        display = pd.concat([display, pd.DataFrame([totals], index=["GRAND TOTAL"])])

        
        st.subheader("🥛 Retailer × Category (Liters)")
        st.dataframe(display, use_container_width=True)

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
            use_container_width=True
        )

        # ---- totals by payment mode ----
        mode_totals = (
            pv.groupby("payment_mode", as_index=False)["amount"]
              .sum()
              .sort_values("amount", ascending=False)
              .rename(columns={"payment_mode": "Mode", "amount": "Total (₹)"})
        )

        st.subheader("💳 Payment Totals by Mode")
        st.dataframe(
            mode_totals.style.format({"Total (₹)": "₹{:.2f}"}),
            use_container_width=True
        )

# ================== ZONE-WISE SUMMARY ==================
elif menu == "📍 Zone-wise Summary":
    st.header("📍 Zone-wise Summary (Category Columns + Payment Mode Totals)")

    s_date = st.date_input("Select Date", value=date.today(), key="zone_sum_date")

    # ---------- Milk Sent: Zone × Category (Wide) ----------
    pivot = zone_category_pivot_for_day(s_date)

    st.subheader("🥛 Milk Sent — Zone × Category (Liters)")
    if pivot.empty:
        st.info("No entries on this date.")
    else:
        display = pivot.copy()

        # convert category columns to "-" if 0 (keep TOTAL numeric)
        for c in display.columns:
            if c in ("Zone", "TOTAL (L)"):
                continue
            display[c] = display[c].apply(lambda x: "-" if float(x or 0.0) == 0.0 else float(x))

        # add grand total row
        numeric_cols = [c for c in pivot.columns if c != "Zone"]
        grand = {"Zone": "GRAND TOTAL"}
        for c in numeric_cols:
            grand[c] = float(pivot[c].sum())
        display = pd.concat([display, pd.DataFrame([grand])], ignore_index=True)

        st.dataframe(display, use_container_width=True)

    # ---------- Payments: Total by Mode (Zone-aware) ----------
    st.subheader("💳 Payments Collected — Totals by Mode (Zone-aware)")
    p_day = _day_payments_for_zone(s_date, "All Zones")  # we summarize all zones together below
    if p_day.empty:
        st.info("No payments recorded on this date.")
    else:
        # attach zone to each payment
        pv = p_day.merge(retailers[["retailer_id", "zone"]], on="retailer_id", how="left")
        pv["zone"] = pv["zone"].apply(_norm_zone)

        # totals by payment mode (overall)
        mode_totals = (
            pv.groupby("payment_mode", as_index=False)["amount"]
            .sum()
            .sort_values("amount", ascending=False)
            .rename(columns={"payment_mode": "Mode", "amount": "Total (₹)"})
        )

        # totals by payment mode per zone (optional but usually what you actually want)
        mode_zone = (
            pv.groupby(["zone", "payment_mode"], as_index=False)["amount"]
            .sum()
            .rename(columns={"zone": "Zone", "payment_mode": "Mode", "amount": "Total (₹)"})
            .sort_values(["Zone", "Total (₹)"], ascending=[True, False])
        )

        st.caption("Overall totals (all zones combined):")
        st.dataframe(mode_totals.style.format({"Total (₹)": "₹{:.2f}"}), use_container_width=True)

        st.caption("Zone-wise totals by mode:")
        st.dataframe(mode_zone.style.format({"Total (₹)": "₹{:.2f}"}), use_container_width=True)

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

    view = build_entries_view(df, want_milk_type_col=False)
    st.dataframe(
        view[["entry_id", "date", "zone", "Retailer", "Category", "qty", "rate", "amount"]],
        use_container_width=True
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
            safe_write_csv(entries, ENTRIES_FILE)
            st.success("Updated quantity (stored rate preserved).")
            st.rerun()

    with col2:
        confirm = st.text_input("Type DELETE to delete this entry", key="single_delete_confirm")
        if st.button("Delete Entry", key="single_delete_btn"):
            if confirm != "DELETE":
                st.warning("Type DELETE to confirm.")
            else:
                entries2 = entries.loc[entries["entry_id"] != int(entry_id)].copy()
                safe_write_csv(entries2, ENTRIES_FILE)
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
            if name.strip():
                cid = next_id_from_df(categories, "category_id")
                new_row = pd.DataFrame([[cid, name.strip(), description, float(default_price), True]], columns=CSV_SCHEMAS[CATEGORIES_FILE])
                categories = pd.concat([categories, new_row], ignore_index=True)
                safe_write_csv(categories, CATEGORIES_FILE)
                st.success(f"✅ Category '{name}' added.")
                st.rerun()

    with tab2:
        if categories.empty:
            st.info("No categories yet.")
        else:
            st.dataframe(categories, use_container_width=True)

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
                    mask = categories["category_id"] == cid
                    categories.loc[mask, ["name", "description", "default_price", "is_active"]] = [
                        new_name.strip(),
                        new_desc,
                        float(new_default_price),
                        bool(new_active),
                    ]
                    safe_write_csv(categories, CATEGORIES_FILE)
                    st.success("Updated!")
                    st.rerun()

            with colB:
                if st.button("Deactivate Category (Safe)", key="cat_deactivate_btn"):
                    mask = categories["category_id"] == cid
                    categories.loc[mask, "is_active"] = False
                    safe_write_csv(categories, CATEGORIES_FILE)
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
                        categories = categories.loc[categories["category_id"] != cid].copy()
                        safe_write_csv(categories, CATEGORIES_FILE)
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
                rid = next_id_from_df(retailers, "retailer_id")
                new_row = pd.DataFrame([[rid, name.strip(), contact, address, z, True]], columns=CSV_SCHEMAS[RETAILERS_FILE])
                retailers = pd.concat([retailers, new_row], ignore_index=True)
                safe_write_csv(retailers, RETAILERS_FILE)
                st.success(f"✅ Retailer '{name}' added to zone '{z}'!")
                st.rerun()

    with tab2:
        if retailers.empty:
            st.info("No retailers yet.")
        else:
            st.dataframe(retailers, use_container_width=True)

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
                    mask = retailers["retailer_id"] == rid
                    retailers.loc[mask, ["name", "contact", "address", "zone", "is_active"]] = [
                        new_name.strip(),
                        new_contact,
                        new_address,
                        z,
                        bool(new_active),
                    ]
                    safe_write_csv(retailers, RETAILERS_FILE)
                    st.success("Updated!")
                    st.rerun()

            with colB:
                if st.button("Deactivate Retailer (Safe)", key="ret_deactivate_btn"):
                    mask = retailers["retailer_id"] == rid
                    retailers.loc[mask, "is_active"] = False
                    safe_write_csv(retailers, RETAILERS_FILE)
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
                        retailers = retailers.loc[retailers["retailer_id"] != rid].copy()
                        safe_write_csv(retailers, RETAILERS_FILE)
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

                pid = next_id_from_df(prices, "price_id")
                new_row = pd.DataFrame([[pid, int(target_retailer_id), int(cid), float(price_val), str(effective_date)]], columns=CSV_SCHEMAS[PRICES_FILE])
                prices = pd.concat([prices, new_row], ignore_index=True)
                safe_write_csv(prices, PRICES_FILE)
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
            st.dataframe(view[["price_id", "Retailer", "Category", "price", "effective_date"]], use_container_width=True)

# ================== LEDGER ==================
elif menu == "📒 Ledger":
    st.header(f"📒 Ledger (Grid View) — {selected_zone}")

    if entries_z.empty:
        st.info("No entries in this zone context.")
        st.stop()

    # filters
    col1, col2 = st.columns(2)
    with col1:
        d_from = st.date_input("From Date", value=date.today() - timedelta(days=30), key="ledger_from")
    with col2:
        d_to = st.date_input("To Date", value=date.today(), key="ledger_to")

    # safety
    if d_from > d_to:
        st.error("From Date cannot be after To Date.")
        st.stop()

    # build view and apply date filter
    v = build_entries_view(entries_z, want_milk_type_col=False)
    v["date"] = pd.to_datetime(v["date"], errors="coerce").dt.date
    v = v.loc[(v["date"] >= d_from) & (v["date"] <= d_to)].copy()

    if v.empty:
        st.info("No entries in this date range.")
        st.stop()

    # Pivot grid: Retailer × Category (Liters)
    pivot = pd.pivot_table(
        v,
        index="Retailer",
        columns="Category",
        values="qty",
        aggfunc="sum",
        fill_value=0.0
    )

    # display formatting "-" for zero
    display = pivot.copy()
    for c in display.columns:
        display[c] = display[c].apply(lambda x: "-" if float(x or 0.0) == 0.0 else float(x))

    # row totals
    display["TOTAL (L)"] = pivot.sum(axis=1)

    # grand total row
    grand = {"TOTAL (L)": float(pivot.values.sum())}
    for c in pivot.columns:
        grand[c] = float(pivot[c].sum())
    display = pd.concat([display, pd.DataFrame([grand], index=["GRAND TOTAL"])])

    st.subheader("🥛 Retailer × Category Grid (Liters)")
    st.dataframe(display, use_container_width=True)

    # optional: totals by category only
    st.subheader("📌 Category Totals (Liters)")
    cat_totals = pivot.sum(axis=0).reset_index()
    cat_totals.columns = ["Category", "Total (L)"]
    cat_totals = cat_totals.sort_values("Total (L)", ascending=False)
    st.dataframe(cat_totals, use_container_width=True)

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
        result_view = build_entries_view(filtered_entries, want_milk_type_col=False)
        st.dataframe(
            result_view[["date", "zone", "Retailer", "Category", "qty", "rate", "amount"]],
            use_container_width=True,
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
            if name.strip():
                did = next_id_from_df(distributors, "distributor_id")
                new_row = pd.DataFrame([[did, name.strip(), contact, address, True]], columns=CSV_SCHEMAS[DISTRIBUTORS_FILE])
                distributors = pd.concat([distributors, new_row], ignore_index=True)
                safe_write_csv(distributors, DISTRIBUTORS_FILE)
                st.success("✅ Distributor added!")
                st.rerun()

    with tab2:
        if distributors.empty:
            st.info("No distributors yet.")
        else:
            st.dataframe(distributors, use_container_width=True)

            edit_dis = st.selectbox("Select distributor to edit", distributors["name"].tolist(), key="dist_edit_sel")
            dis_data = distributors.loc[distributors["name"] == edit_dis].iloc[0]
            did = int(dis_data["distributor_id"])

            col1, col2, col3 = st.columns(3)
            with col1:
                new_name = st.text_input("New name", value=str(dis_data["name"]), key="dist_new_name")
            with col2:
                new_contact = st.text_input("New contact", value=str(dis_data.get("contact", "")), key="dist_new_contact")
            with col3:
                new_active = st.checkbox("Active", value=bool(dis_data.get("is_active", True)), key="dist_new_active")

            new_address = st.text_area("New address", value=str(dis_data.get("address", "")), key="dist_new_address")

            colA, colB, colC = st.columns(3)
            with colA:
                if st.button("Update Distributor", key="dist_update_btn"):
                    mask = distributors["distributor_id"] == did
                    distributors.loc[mask, ["name", "contact", "address", "is_active"]] = [
                        new_name.strip(),
                        new_contact,
                        new_address,
                        bool(new_active),
                    ]
                    safe_write_csv(distributors, DISTRIBUTORS_FILE)
                    st.success("Updated!")
                    st.rerun()

            with colB:
                if st.button("Deactivate Distributor (Safe)", key="dist_deactivate_btn"):
                    mask = distributors["distributor_id"] == did
                    distributors.loc[mask, "is_active"] = False
                    safe_write_csv(distributors, DISTRIBUTORS_FILE)
                    st.success("Distributor deactivated (history preserved).")
                    st.rerun()

            with colC:
                confirm = st.text_input("Type DELETE to hard delete distributor", key="dist_delete_confirm")
                if st.button("🗑️ Hard Delete Distributor", type="secondary", key="dist_delete_btn"):
                    if confirm != "DELETE":
                        st.warning("Type DELETE to confirm.")
                    elif is_distributor_referenced(did):
                        st.error("Blocked: Distributor is referenced in purchases/payments. Deactivate instead.")
                    else:
                        distributors = distributors.loc[distributors["distributor_id"] != did].copy()
                        safe_write_csv(distributors, DISTRIBUTORS_FILE)
                        st.success("Hard deleted.")
                        st.rerun()

# ================== PLACEHOLDERS ==================
elif menu == "📦 Milk Purchases":
    st.header("📦 Milk Purchase from Distributors")
    st.info("Purchases are not zone-based here (zones are for retailer accounting). (UI entry form can be added next.)")

elif menu == "💸 Distributor Payments":
    st.header("💸 Payments to Distributors")
    st.info("Not zone-based (zones are for retailers). (UI entry form can be added next.)")

elif menu == "🗑️ Milk Wastage":
    st.header("🗑️ Daily Milk Wastage Tracking")
    st.info("Not zone-based (zones are for retailer accounting). (UI entry form can be added next.)")

elif menu == "💼 Expenses":
    st.header("💼 Business Expenses Management")
    st.info("Not zone-based (zones are for retailer accounting). (UI entry form can be added next.)")

elif menu == "🧾 Generate Bill":
    st.header(f"🧾 Generate Customer Bill — {selected_zone}")
    st.info("Billing preview/invoice can be re-integrated next. Core accounting is preserved.")

# ================== DATA HEALTH & BACKUP ==================
elif menu == "🛡️ Data Health & Backup":
    st.header("🛡️ Data Health & Backup")

    st.subheader("Backups")
    st.write(f"- Automatic backups are created before each write: `{BACKUP_DIR}`")
    st.write(f"- Backups kept per file: **{BACKUPS_PER_FILE}**")

    st.subheader("Create Full Backup ZIP")
    if st.button("📦 Create Backup ZIP Now", key="zip_now_btn"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk("data"):
                for fn in files:
                    path = os.path.join(root, fn)
                    zf.write(path, arcname=path)
        buf.seek(0)
        st.download_button("⬇️ Download Backup ZIP", data=buf, file_name=f"milk_backup_{_ts()}.zip", key="zip_dl_btn")

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
            st.dataframe(df, use_container_width=True)
