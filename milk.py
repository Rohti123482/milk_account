import streamlit as st
import pandas as pd
from datetime import date
import os
import plotly.express as px
import plotly.graph_objects as go

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Milk Accounting Pro", layout="wide", initial_sidebar_state="expanded")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stMetric {
        background-color: white; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #555 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #1f1f1f !important;
        font-size: 1.5rem !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #666 !important;
    }
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
""", unsafe_allow_html=True)

# ------------------ FILE SETUP ------------------
if not os.path.exists("data"):
    os.mkdir("data")

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


# ------------------ LOAD DATA WITH MIGRATION ------------------
def load_and_migrate_data():
    # Retailers
    if os.path.exists(RETAILERS_FILE):
        retailers = pd.read_csv(RETAILERS_FILE)
        if "contact" not in retailers.columns:
            retailers["contact"] = ""
        if "address" not in retailers.columns:
            retailers["address"] = ""
        if "retailer_id" in retailers.columns:
            retailers["retailer_id"] = retailers["retailer_id"].astype(int)
    else:
        retailers = pd.DataFrame(columns=["retailer_id", "name", "contact", "address"])

    # Categories
    if os.path.exists(CATEGORIES_FILE):
        categories = pd.read_csv(CATEGORIES_FILE)
        if "description" not in categories.columns:
            categories["description"] = ""
        if "default_price" not in categories.columns:
            categories["default_price"] = 0.0
        if "category_id" in categories.columns:
            categories["category_id"] = categories["category_id"].astype(int)
    else:
        categories = pd.DataFrame(columns=["category_id", "name", "description", "default_price"])

    # Prices
    if os.path.exists(PRICES_FILE):
        prices = pd.read_csv(PRICES_FILE)
        if "price_id" not in prices.columns:
            prices.insert(0, "price_id", range(1, len(prices) + 1))
        if "effective_date" not in prices.columns:
            prices["effective_date"] = str(date.today())
        prices["price_id"] = prices["price_id"].astype(int)
        prices["retailer_id"] = prices["retailer_id"].astype(int)
        prices["category_id"] = prices["category_id"].astype(int)
    else:
        prices = pd.DataFrame(columns=["price_id", "retailer_id", "category_id", "price", "effective_date"])

    # Entries (MIGRATION: add rate column)
    if os.path.exists(ENTRIES_FILE):
        entries = pd.read_csv(ENTRIES_FILE)
        if "entry_id" not in entries.columns:
            entries.insert(0, "entry_id", range(1, len(entries) + 1))
        if "rate" not in entries.columns:
            # If amount and qty exist, approximate old rate; otherwise 0
            if "amount" in entries.columns and "qty" in entries.columns:
                entries["rate"] = entries.apply(lambda r: (r["amount"] / r["qty"]) if r.get("qty", 0) else 0, axis=1)
            else:
                entries["rate"] = 0.0
        entries["entry_id"] = entries["entry_id"].astype(int)
        entries["retailer_id"] = entries["retailer_id"].astype(int)
        entries["category_id"] = entries["category_id"].astype(int)
    else:
        entries = pd.DataFrame(columns=["entry_id", "date", "retailer_id", "category_id", "qty", "rate", "amount"])

    # Payments
    if os.path.exists(PAYMENTS_FILE):
        payments = pd.read_csv(PAYMENTS_FILE)
        if "payment_id" not in payments.columns:
            payments.insert(0, "payment_id", range(1, len(payments) + 1))
        if "payment_mode" not in payments.columns:
            payments["payment_mode"] = "Cash"
        if "note" not in payments.columns:
            payments["note"] = ""
        payments["payment_id"] = payments["payment_id"].astype(int)
        payments["retailer_id"] = payments["retailer_id"].astype(int)
    else:
        payments = pd.DataFrame(columns=["payment_id", "date", "retailer_id", "amount", "payment_mode", "note"])

    # Distributors
    if os.path.exists(DISTRIBUTORS_FILE):
        distributors = pd.read_csv(DISTRIBUTORS_FILE)
        if "contact" not in distributors.columns:
            distributors["contact"] = ""
        if "address" not in distributors.columns:
            distributors["address"] = ""
        distributors["distributor_id"] = distributors["distributor_id"].astype(int)
    else:
        distributors = pd.DataFrame(columns=["distributor_id", "name", "contact", "address"])

    # Distributor Purchases
    if os.path.exists(DISTRIBUTOR_PURCHASES_FILE):
        dist_purchases = pd.read_csv(DISTRIBUTOR_PURCHASES_FILE)
        dist_purchases["purchase_id"] = dist_purchases["purchase_id"].astype(int)
        dist_purchases["distributor_id"] = dist_purchases["distributor_id"].astype(int)
        dist_purchases["category_id"] = dist_purchases["category_id"].astype(int)
    else:
        dist_purchases = pd.DataFrame(
            columns=["purchase_id", "date", "distributor_id", "category_id", "qty", "rate", "amount"]
        )

    # Distributor Payments
    if os.path.exists(DISTRIBUTOR_PAYMENTS_FILE):
        dist_payments = pd.read_csv(DISTRIBUTOR_PAYMENTS_FILE)
        if "payment_mode" not in dist_payments.columns:
            dist_payments["payment_mode"] = "Cash"
        if "note" not in dist_payments.columns:
            dist_payments["note"] = ""
        dist_payments["payment_id"] = dist_payments["payment_id"].astype(int)
        dist_payments["distributor_id"] = dist_payments["distributor_id"].astype(int)
    else:
        dist_payments = pd.DataFrame(columns=["payment_id", "date", "distributor_id", "amount", "payment_mode", "note"])

    # Wastage
    if os.path.exists(WASTAGE_FILE):
        wastage = pd.read_csv(WASTAGE_FILE)
        wastage["wastage_id"] = wastage["wastage_id"].astype(int)
        wastage["category_id"] = wastage["category_id"].astype(int)
    else:
        wastage = pd.DataFrame(columns=["wastage_id", "date", "category_id", "qty", "reason", "estimated_loss"])

    # Expenses
    if os.path.exists(EXPENSES_FILE):
        expenses = pd.read_csv(EXPENSES_FILE)
        if "paid" not in expenses.columns:
            expenses["paid"] = True
        expenses["expense_id"] = expenses["expense_id"].astype(int)
    else:
        expenses = pd.DataFrame(columns=["expense_id", "date", "category", "description", "amount", "payment_mode", "paid"])

    # Save back after migration (so future loads are clean)
    retailers.to_csv(RETAILERS_FILE, index=False)
    categories.to_csv(CATEGORIES_FILE, index=False)
    prices.to_csv(PRICES_FILE, index=False)
    entries.to_csv(ENTRIES_FILE, index=False)
    payments.to_csv(PAYMENTS_FILE, index=False)
    distributors.to_csv(DISTRIBUTORS_FILE, index=False)
    dist_purchases.to_csv(DISTRIBUTOR_PURCHASES_FILE, index=False)
    dist_payments.to_csv(DISTRIBUTOR_PAYMENTS_FILE, index=False)
    wastage.to_csv(WASTAGE_FILE, index=False)
    expenses.to_csv(EXPENSES_FILE, index=False)

    return retailers, categories, prices, entries, payments, distributors, dist_purchases, dist_payments, wastage, expenses


retailers, categories, prices, entries, payments, distributors, dist_purchases, dist_payments, wastage, expenses = load_and_migrate_data()


# ------------------ HELPER FUNCTIONS ------------------
def get_retailer_balance(retailer_id: int) -> float:
    total_sales = entries.loc[entries["retailer_id"] == retailer_id, "amount"].sum() if not entries.empty else 0
    total_payments = payments.loc[payments["retailer_id"] == retailer_id, "amount"].sum() if not payments.empty else 0
    return float(total_sales - total_payments)


def get_price_for_date(retailer_id: int, category_id: int, entry_date) -> float | None:
    """
    Return the most recent effective custom price for (retailer, category) at entry_date.
    If none, return category default_price if > 0.
    """
    entry_dt = pd.to_datetime(entry_date)

    if not prices.empty:
        price_rows = prices.loc[
            (prices["retailer_id"] == retailer_id) & (prices["category_id"] == category_id)
        ].copy()

        if not price_rows.empty and "effective_date" in price_rows.columns:
            price_rows["effective_date"] = pd.to_datetime(price_rows["effective_date"], errors="coerce")
            valid_prices = price_rows.loc[price_rows["effective_date"] <= entry_dt]
            if not valid_prices.empty:
                return float(valid_prices.sort_values("effective_date", ascending=False).iloc[0]["price"])

    # fallback: category default
    if not categories.empty and "default_price" in categories.columns:
        cat_row = categories.loc[categories["category_id"] == category_id]
        if not cat_row.empty:
            default_price = cat_row.iloc[0].get("default_price", 0.0)
            if pd.notna(default_price) and float(default_price) > 0:
                return float(default_price)

    return None


# ------------------ UI ------------------
st.title("🥛 SHREE SAI MILK CENTER")

menu = st.sidebar.radio(
    "📋 Navigation",
    [
        "📊 Dashboard",
        "🥛 Milk Categories",
        "🏪 Retailers",
        "💰 Price Management",
        "📝 Daily Entry & Payment",
        "📒 Ledger",
        "🔍 Filters & Reports",
        "🚚 Distributors",
        "📦 Milk Purchases",
        "💸 Distributor Payments",
        "🗑️ Milk Wastage",
        "💼 Expenses",
        "🧾 Generate Bill",
    ],
)

# ------------------ DASHBOARD ------------------
if menu == "📊 Dashboard":
    st.header("📊 Business Overview")

    col1, col2, col3, col4 = st.columns(4)
    total_milk = entries["qty"].sum() if not entries.empty else 0
    total_sales = entries["amount"].sum() if not entries.empty else 0
    total_payments = payments["amount"].sum() if not payments.empty else 0
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

    col1, col2 = st.columns(2)
    with col1:
        if total_sales > 0 or total_payments > 0:
            st.subheader("💰 Sales vs Payments")
            pie_data = pd.DataFrame(
                {"Category": ["Total Sales", "Paid", "Outstanding"], "Amount": [total_sales, total_payments, outstanding]}
            )
            fig = px.pie(pie_data, values="Amount", names="Category", hole=0.4)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=True, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not entries.empty:
            st.subheader("📈 Sales Trend (Last 30 Days)")
            daily_sales = entries.copy()
            daily_sales["date"] = pd.to_datetime(daily_sales["date"], errors="coerce")
            last_30 = daily_sales.loc[daily_sales["date"] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]
            daily_sales_agg = last_30.groupby("date")["amount"].sum().reset_index()
            fig = px.area(daily_sales_agg, x="date", y="amount")
            fig.update_layout(xaxis_title="Date", yaxis_title="Amount (₹)", showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if not retailers.empty and not entries.empty:
            st.subheader("🏪 Top 5 Retailers by Sales")
            retailer_sales = (
                entries.merge(retailers, on="retailer_id")
                .groupby("name")["amount"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )
            fig = px.bar(retailer_sales, x="amount", y="name", orientation="h", text="amount")
            fig.update_traces(texttemplate="₹%{text:.2f}", textposition="outside")
            fig.update_layout(xaxis_title="Sales (₹)", yaxis_title="Retailer", showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not categories.empty and not entries.empty:
            st.subheader("🥛 Milk Category Distribution")
            category_sales = entries.merge(categories, on="category_id").groupby("name")["qty"].sum().reset_index()
            fig = px.pie(category_sales, values="qty", names="name")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=True, height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    if not retailers.empty and not entries.empty:
        st.subheader("💳 Retailer Payment Status")
        balance_data = []
        for _, r in retailers.iterrows():
            rid = int(r["retailer_id"])
            sales = entries.loc[entries["retailer_id"] == rid, "amount"].sum()
            paid = payments.loc[payments["retailer_id"] == rid, "amount"].sum() if not payments.empty else 0
            if sales > 0:
                balance_data.append(
                    {"Retailer": r["name"], "Sales": sales, "Paid": paid, "Outstanding": sales - paid, "Payment %": (paid / sales * 100)}
                )

        if balance_data:
            balance_df = pd.DataFrame(balance_data)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Paid", x=balance_df["Retailer"], y=balance_df["Paid"], text=balance_df["Paid"]))
            fig.add_trace(
                go.Bar(name="Outstanding", x=balance_df["Retailer"], y=balance_df["Outstanding"], text=balance_df["Outstanding"])
            )
            fig.update_layout(barmode="stack", xaxis_title="Retailer", yaxis_title="Amount (₹)", height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                balance_df.style.format({"Sales": "₹{:.2f}", "Paid": "₹{:.2f}", "Outstanding": "₹{:.2f}", "Payment %": "{:.1f}%"}),
                use_container_width=True,
            )

# ------------------ MILK CATEGORIES ------------------
elif menu == "🥛 Milk Categories":
    st.header("🥛 Milk Categories Management")
    tab1, tab2 = st.tabs(["➕ Add Category", "✏️ Edit Categories"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Category Name")
        with col2:
            description = st.text_input("Description (optional)")
        with col3:
            default_price = st.number_input("Default Price per Liter (₹)", min_value=0.0, step=0.5)

        if st.button("Add Category", type="primary"):
            if name.strip():
                cid = int(categories["category_id"].max()) + 1 if not categories.empty else 1
                new_row = pd.DataFrame([[cid, name.strip(), description, float(default_price)]], columns=categories.columns)
                categories = pd.concat([categories, new_row], ignore_index=True)
                categories.to_csv(CATEGORIES_FILE, index=False)
                st.success(f"✅ Category '{name}' added.")
                st.rerun()

    with tab2:
        if categories.empty:
            st.info("No categories yet.")
        else:
            display_cats = categories.copy()
            display_cats["default_price"] = display_cats["default_price"].apply(lambda x: f"₹{float(x):.2f}/L")
            st.dataframe(display_cats, use_container_width=True)

            edit_cat = st.selectbox("Select category to edit", categories["name"].tolist())
            cat_data = categories.loc[categories["name"] == edit_cat].iloc[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                new_name = st.text_input("New name", value=str(cat_data["name"]))
            with col2:
                new_desc = st.text_input("New description", value=str(cat_data.get("description", "")) if pd.notna(cat_data.get("description", "")) else "")
            with col3:
                new_default_price = st.number_input(
                    "Default Price (₹/L)",
                    value=float(cat_data.get("default_price", 0.0)) if pd.notna(cat_data.get("default_price", 0.0)) else 0.0,
                    min_value=0.0,
                    step=0.5,
                )

            colA, colB = st.columns(2)
            with colA:
                if st.button("Update Category"):
                    mask = categories["name"] == edit_cat
                    categories.loc[mask, ["name", "description", "default_price"]] = [new_name.strip(), new_desc, float(new_default_price)]
                    categories.to_csv(CATEGORIES_FILE, index=False)
                    st.success("Updated!")
                    st.rerun()

            with colB:
                if st.button("🗑️ Delete Category", type="secondary"):
                    categories = categories.loc[categories["name"] != edit_cat].copy()
                    categories.to_csv(CATEGORIES_FILE, index=False)
                    st.success("Deleted!")
                    st.rerun()

# ------------------ RETAILERS ------------------
elif menu == "🏪 Retailers":
    st.header("🏪 Retailer Management")
    tab1, tab2 = st.tabs(["➕ Add Retailer", "✏️ Edit Retailers"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Retailer Name")
            contact = st.text_input("Contact Number")
        with col2:
            address = st.text_area("Address")

        if st.button("Add Retailer", type="primary"):
            if name.strip():
                rid = int(retailers["retailer_id"].max()) + 1 if not retailers.empty else 1
                new_row = pd.DataFrame([[rid, name.strip(), contact, address]], columns=retailers.columns)
                retailers = pd.concat([retailers, new_row], ignore_index=True)
                retailers.to_csv(RETAILERS_FILE, index=False)
                st.success(f"✅ Retailer '{name}' added!")
                st.rerun()

    with tab2:
        if retailers.empty:
            st.info("No retailers yet.")
        else:
            st.dataframe(retailers, use_container_width=True)
            edit_ret = st.selectbox("Select retailer to edit", retailers["name"].tolist())
            ret_data = retailers.loc[retailers["name"] == edit_ret].iloc[0]

            new_name = st.text_input("New name", value=str(ret_data["name"]))
            new_contact = st.text_input("New contact", value=str(ret_data.get("contact", "")) if pd.notna(ret_data.get("contact", "")) else "")
            new_address = st.text_area("New address", value=str(ret_data.get("address", "")) if pd.notna(ret_data.get("address", "")) else "")

            colA, colB = st.columns(2)
            with colA:
                if st.button("Update Retailer"):
                    mask = retailers["name"] == edit_ret
                    retailers.loc[mask, ["name", "contact", "address"]] = [new_name.strip(), new_contact, new_address]
                    retailers.to_csv(RETAILERS_FILE, index=False)
                    st.success("Updated!")
                    st.rerun()

            with colB:
                if st.button("🗑️ Delete Retailer", type="secondary"):
                    retailers = retailers.loc[retailers["name"] != edit_ret].copy()
                    retailers.to_csv(RETAILERS_FILE, index=False)
                    st.success("Deleted!")
                    st.rerun()

# ------------------ PRICE MANAGEMENT ------------------
elif menu == "💰 Price Management":
    st.header("💰 Price Management")

    if not categories.empty and "default_price" in categories.columns:
        st.subheader("📋 Default Prices")
        default_view = categories[["name", "default_price"]].rename(columns={"name": "Category", "default_price": "Default Price (₹/L)"})
        st.dataframe(default_view.style.format({"Default Price (₹/L)": "₹{:.2f}"}), use_container_width=True)
        st.info("💡 Default prices apply to all retailers unless overridden.")
        st.divider()

    tab1, tab2 = st.tabs(["➕ Set Custom Price", "✏️ View/Edit Prices"])

    with tab1:
        if retailers.empty or categories.empty:
            st.warning("⚠️ Please add retailers and categories first")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                retailer = st.selectbox("Retailer", retailers["name"])
            with col2:
                category = st.selectbox("Milk Category", categories["name"])
                cat_data = categories.loc[categories["name"] == category].iloc[0]
                default_price_val = float(cat_data.get("default_price", 0.0)) if pd.notna(cat_data.get("default_price", 0.0)) else 0.0
                st.info(f"Default Price: ₹{default_price_val:.2f}/L")
            with col3:
                price = st.number_input("Custom Price per Liter (₹)", min_value=0.0, step=0.5, value=float(default_price_val))

            effective_date = st.date_input("Effective Date", date.today())

            if st.button("Save Custom Price", type="primary"):
                rid = int(retailers.loc[retailers["name"] == retailer, "retailer_id"].values[0])
                cid = int(categories.loc[categories["name"] == category, "category_id"].values[0])
                pid = int(prices["price_id"].max()) + 1 if not prices.empty else 1

                new_row = pd.DataFrame([[pid, rid, cid, float(price), str(effective_date)]], columns=prices.columns)
                prices = pd.concat([prices, new_row], ignore_index=True)
                prices.to_csv(PRICES_FILE, index=False)
                st.success(f"✅ Custom price saved.")
                st.rerun()

    with tab2:
        if prices.empty:
            st.info("No custom prices set.")
        else:
            price_view = prices.merge(retailers, on="retailer_id").merge(categories, on="category_id")
            price_view = price_view.rename(columns={"name_x": "Retailer", "name_y": "Category"})
            st.dataframe(price_view[["price_id", "Retailer", "Category", "price", "effective_date"]], use_container_width=True)

            del_price_id = st.number_input("Price ID to delete", min_value=1, step=1)
            if st.button("🗑️ Delete Price"):
                if del_price_id in prices["price_id"].values:
                    prices = prices.loc[prices["price_id"] != del_price_id].copy()
                    prices.to_csv(PRICES_FILE, index=False)
                    st.success("Deleted!")
                    st.rerun()

# ------------------ DAILY ENTRY & PAYMENT ------------------
elif menu == "📝 Daily Entry & Payment":
    st.header("📝 Daily Milk Entry & Payment")
    tab1, tab2, tab3 = st.tabs(["➕ New Entry", "💳 Record Payment", "✏️ Edit Entries"])

    with tab1:
        if retailers.empty or categories.empty:
            st.warning("⚠️ Please add retailers and categories first")
        else:
            col1, col2 = st.columns(2)
            with col1:
                entry_date = st.date_input("Date", date.today())
                retailer = st.selectbox("Retailer", retailers["name"])
            with col2:
                category = st.selectbox("Milk Category", categories["name"])
                qty = st.number_input("Quantity (Liters)", min_value=0.0, step=0.5)

            if st.button("Save Entry", type="primary"):
                if qty <= 0:
                    st.warning("Enter quantity > 0")
                else:
                    rid = int(retailers.loc[retailers["name"] == retailer, "retailer_id"].values[0])
                    cid = int(categories.loc[categories["name"] == category, "category_id"].values[0])
                    rate = get_price_for_date(rid, cid, entry_date)

                    if rate is None:
                        st.error("❌ Price not set for this retailer & category (and default price missing).")
                    else:
                        amount = float(qty) * float(rate)
                        eid = int(entries["entry_id"].max()) + 1 if not entries.empty else 1
                        new_row = pd.DataFrame([[eid, str(entry_date), rid, cid, float(qty), float(rate), float(amount)]], columns=entries.columns)
                        entries = pd.concat([entries, new_row], ignore_index=True)
                        entries.to_csv(ENTRIES_FILE, index=False)
                        st.success(f"✅ Saved — {qty}L × ₹{rate} = ₹{amount:.2f}")
                        st.rerun()

    with tab2:
        if retailers.empty:
            st.warning("⚠️ Add retailers first")
        else:
            retailer = st.selectbox("Select Retailer", retailers["name"].tolist(), key="payment_retailer")
            rid = int(retailers.loc[retailers["name"] == retailer, "retailer_id"].values[0])

            balance = get_retailer_balance(rid)
            st.metric(f"{retailer}'s Outstanding Balance", f"₹{balance:.2f}")

            col1, col2 = st.columns(2)
            with col1:
                payment_date = st.date_input("Payment Date", date.today())
                amount = st.number_input("Payment Amount (₹)", min_value=0.0, step=100.0)
            with col2:
                payment_mode = st.selectbox("Payment Mode", ["Cash", "UPI", "Bank Transfer", "Cheque", "Card"])
                note = st.text_area("Note (optional)")

            if st.button("Record Payment", type="primary"):
                if amount <= 0:
                    st.warning("Enter amount > 0")
                else:
                    pid = int(payments["payment_id"].max()) + 1 if not payments.empty else 1
                    new_row = pd.DataFrame([[pid, str(payment_date), rid, float(amount), payment_mode, note]], columns=payments.columns)
                    payments = pd.concat([payments, new_row], ignore_index=True)
                    payments.to_csv(PAYMENTS_FILE, index=False)
                    st.success(f"✅ Payment recorded.")
                    st.rerun()

    with tab3:
        if entries.empty:
            st.info("No entries yet.")
        else:
            entry_view = entries.merge(retailers, on="retailer_id").merge(categories, on="category_id")
            entry_view = entry_view.rename(columns={"name_x": "Retailer", "name_y": "Category"})
            st.dataframe(entry_view[["entry_id", "date", "Retailer", "Category", "qty", "rate", "amount"]], use_container_width=True)

            edit_entry_id = st.number_input("Entry ID to edit", min_value=1, step=1)
            if edit_entry_id in entries["entry_id"].values:
                entry_data = entries.loc[entries["entry_id"] == edit_entry_id].iloc[0]
                new_qty = st.number_input("New Quantity", value=float(entry_data["qty"]), min_value=0.0, step=0.5)

                colA, colB = st.columns(2)
                with colA:
                    if st.button("Update Entry"):
                        # keep original rate to preserve accounting truth
                        rate = float(entry_data["rate"])
                        new_amount = float(new_qty) * rate
                        entries.loc[entries["entry_id"] == edit_entry_id, ["qty", "amount"]] = [float(new_qty), float(new_amount)]
                        entries.to_csv(ENTRIES_FILE, index=False)
                        st.success("Updated!")
                        st.rerun()

                with colB:
                    if st.button("🗑️ Delete Entry"):
                        entries = entries.loc[entries["entry_id"] != edit_entry_id].copy()
                        entries.to_csv(ENTRIES_FILE, index=False)
                        st.success("Deleted!")
                        st.rerun()

# ------------------ LEDGER ------------------
elif menu == "📒 Ledger":
    st.header("📒 Complete Ledger")

    if entries.empty:
        st.info("No entries yet")
    else:
        ledger = entries.merge(retailers, on="retailer_id").merge(categories, on="category_id")
        ledger = ledger.rename(columns={"name_x": "Retailer", "name_y": "Milk Type"})
        st.dataframe(ledger[["date", "Retailer", "Milk Type", "qty", "rate", "amount"]], use_container_width=True)

        st.subheader("💰 Balance Summary by Retailer")
        balance_data = []
        for _, r in retailers.iterrows():
            rid = int(r["retailer_id"])
            total_sales = entries.loc[entries["retailer_id"] == rid, "amount"].sum()
            total_payments = payments.loc[payments["retailer_id"] == rid, "amount"].sum() if not payments.empty else 0
            balance_data.append(
                {"Retailer": r["name"], "Total Sales (₹)": float(total_sales), "Total Payments (₹)": float(total_payments), "Outstanding (₹)": float(total_sales - total_payments)}
            )
        st.dataframe(pd.DataFrame(balance_data), use_container_width=True)

# ------------------ FILTERS & REPORTS ------------------
elif menu == "🔍 Filters & Reports":
    st.header("🔍 Filters & Detailed Reports")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_retailer = st.multiselect("Select Retailer(s)", ["All"] + retailers["name"].tolist(), default=["All"])
    with col2:
        filter_category = st.multiselect("Select Category(s)", ["All"] + categories["name"].tolist(), default=["All"])
    with col3:
        date_range = st.date_input("Date Range", value=[], key="filter_date_range")

    filtered_entries = entries.copy()

    if not filtered_entries.empty:
        if "All" not in filter_retailer:
            rid_list = retailers.loc[retailers["name"].isin(filter_retailer), "retailer_id"].tolist()
            filtered_entries = filtered_entries.loc[filtered_entries["retailer_id"].isin(rid_list)].copy()

        if "All" not in filter_category:
            cid_list = categories.loc[categories["name"].isin(filter_category), "category_id"].tolist()
            filtered_entries = filtered_entries.loc[filtered_entries["category_id"].isin(cid_list)].copy()

        if len(date_range) == 2:
            filtered_entries["date"] = pd.to_datetime(filtered_entries["date"], errors="coerce")
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered_entries = filtered_entries.loc[(filtered_entries["date"] >= start) & (filtered_entries["date"] <= end)].copy()

    if filtered_entries.empty:
        st.info("No data matching the filters")
    else:
        st.subheader("📊 Filtered Results")
        result_view = filtered_entries.merge(retailers, on="retailer_id").merge(categories, on="category_id")
        result_view = result_view.rename(columns={"name_x": "Retailer", "name_y": "Category"})
        st.dataframe(result_view[["date", "Retailer", "Category", "qty", "rate", "amount"]], use_container_width=True)

        total_qty = filtered_entries["qty"].sum()
        total_sales = filtered_entries["amount"].sum()

        if "All" not in filter_retailer:
            rid_list = retailers.loc[retailers["name"].isin(filter_retailer), "retailer_id"].tolist()
            filtered_payments = payments.loc[payments["retailer_id"].isin(rid_list)].copy()
        else:
            filtered_payments = payments.copy()

        if len(date_range) == 2 and not filtered_payments.empty:
            filtered_payments["date"] = pd.to_datetime(filtered_payments["date"], errors="coerce")
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered_payments = filtered_payments.loc[(filtered_payments["date"] >= start) & (filtered_payments["date"] <= end)].copy()

        total_paid = filtered_payments["amount"].sum() if not filtered_payments.empty else 0.0
        outstanding = total_sales - total_paid

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Quantity", f"{total_qty:.2f} L")
        c2.metric("Total Sales", f"₹{total_sales:.2f}")
        c3.metric("Total Paid", f"₹{total_paid:.2f}")
        c4.metric("Outstanding", f"₹{outstanding:.2f}", delta=f"₹{outstanding:.2f}" if outstanding > 0 else "Settled")

# ------------------ DISTRIBUTORS ------------------
elif menu == "🚚 Distributors":
    st.header("🚚 Distributor Management")
    tab1, tab2 = st.tabs(["➕ Add Distributor", "✏️ Edit Distributors"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Distributor Name")
            contact = st.text_input("Contact Number")
        with col2:
            address = st.text_area("Address")

        if st.button("Add Distributor", type="primary"):
            if name.strip():
                did = int(distributors["distributor_id"].max()) + 1 if not distributors.empty else 1
                new_row = pd.DataFrame([[did, name.strip(), contact, address]], columns=distributors.columns)
                distributors = pd.concat([distributors, new_row], ignore_index=True)
                distributors.to_csv(DISTRIBUTORS_FILE, index=False)
                st.success("✅ Distributor added!")
                st.rerun()

    with tab2:
        if distributors.empty:
            st.info("No distributors yet.")
        else:
            st.dataframe(distributors, use_container_width=True)
            edit_dist = st.selectbox("Select distributor to edit", distributors["name"].tolist())
            dist_data = distributors.loc[distributors["name"] == edit_dist].iloc[0]

            new_name = st.text_input("New name", value=str(dist_data["name"]))
            new_contact = st.text_input("New contact", value=str(dist_data.get("contact", "")) if pd.notna(dist_data.get("contact", "")) else "")
            new_address = st.text_area("New address", value=str(dist_data.get("address", "")) if pd.notna(dist_data.get("address", "")) else "")

            colA, colB = st.columns(2)
            with colA:
                if st.button("Update Distributor"):
                    mask = distributors["name"] == edit_dist
                    distributors.loc[mask, ["name", "contact", "address"]] = [new_name.strip(), new_contact, new_address]
                    distributors.to_csv(DISTRIBUTORS_FILE, index=False)
                    st.success("Updated!")
                    st.rerun()

            with colB:
                if st.button("🗑️ Delete Distributor", type="secondary"):
                    distributors = distributors.loc[distributors["name"] != edit_dist].copy()
                    distributors.to_csv(DISTRIBUTORS_FILE, index=False)
                    st.success("Deleted!")
                    st.rerun()

# ------------------ MILK PURCHASES ------------------
elif menu == "📦 Milk Purchases":
    st.header("📦 Milk Purchase from Distributors")

    if distributors.empty:
        st.warning("⚠️ Please add distributors first")
    else:
        st.subheader("📊 Distributor Balance Overview")
        dist_balance_data = []
        for _, d in distributors.iterrows():
            did = int(d["distributor_id"])
            total_purchases = dist_purchases.loc[dist_purchases["distributor_id"] == did, "amount"].sum() if not dist_purchases.empty else 0
            total_paid = dist_payments.loc[dist_payments["distributor_id"] == did, "amount"].sum() if not dist_payments.empty else 0
            dist_balance_data.append({"Distributor": d["name"], "Total Purchases": total_purchases, "Total Paid": total_paid, "Outstanding": total_purchases - total_paid})
        if dist_balance_data:
            st.dataframe(pd.DataFrame(dist_balance_data).style.format({"Total Purchases": "₹{:.2f}", "Total Paid": "₹{:.2f}", "Outstanding": "₹{:.2f}"}), use_container_width=True)

        st.divider()
        tab1, tab2, tab3 = st.tabs(["➕ Record Purchase", "✏️ Purchase History", "🔍 Filter Purchases"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                purchase_date = st.date_input("Purchase Date", date.today())
                distributor = st.selectbox("Distributor", distributors["name"].tolist())
            with col2:
                category = st.selectbox("Milk Category", categories["name"].tolist())
                qty = st.number_input("Quantity (Liters)", min_value=0.0, step=0.5)
            with col3:
                rate = st.number_input("Rate per Liter (₹)", min_value=0.0, step=0.5)

            if st.button("Record Purchase", type="primary"):
                if qty > 0 and rate > 0:
                    did = int(distributors.loc[distributors["name"] == distributor, "distributor_id"].values[0])
                    cid = int(categories.loc[categories["name"] == category, "category_id"].values[0])
                    amount = float(qty) * float(rate)
                    pid = int(dist_purchases["purchase_id"].max()) + 1 if not dist_purchases.empty else 1
                    new_row = pd.DataFrame([[pid, str(purchase_date), did, cid, float(qty), float(rate), float(amount)]], columns=dist_purchases.columns)
                    dist_purchases = pd.concat([dist_purchases, new_row], ignore_index=True)
                    dist_purchases.to_csv(DISTRIBUTOR_PURCHASES_FILE, index=False)
                    st.success("✅ Purchase recorded!")
                    st.rerun()

        with tab2:
            if dist_purchases.empty:
                st.info("No purchases recorded yet.")
            else:
                purchase_view = dist_purchases.merge(distributors, on="distributor_id").merge(categories, on="category_id")
                purchase_view = purchase_view.rename(columns={"name_x": "Distributor", "name_y": "Category"})
                st.dataframe(purchase_view[["purchase_id", "date", "Distributor", "Category", "qty", "rate", "amount"]], use_container_width=True)

                del_purchase_id = st.number_input("Purchase ID to delete", min_value=1, step=1)
                if st.button("🗑️ Delete Purchase"):
                    if del_purchase_id in dist_purchases["purchase_id"].values:
                        dist_purchases = dist_purchases.loc[dist_purchases["purchase_id"] != del_purchase_id].copy()
                        dist_purchases.to_csv(DISTRIBUTOR_PURCHASES_FILE, index=False)
                        st.success("Deleted!")
                        st.rerun()

        with tab3:
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_dist = st.multiselect("Distributor(s)", ["All"] + distributors["name"].tolist(), default=["All"])
            with col2:
                filter_cat = st.multiselect("Category(s)", ["All"] + categories["name"].tolist(), default=["All"])
            with col3:
                dr = st.date_input("Date Range", value=[], key="purch_date_range")

            if dist_purchases.empty:
                st.info("No purchase data available")
            else:
                fp = dist_purchases.copy()
                if "All" not in filter_dist:
                    did_list = distributors.loc[distributors["name"].isin(filter_dist), "distributor_id"].tolist()
                    fp = fp.loc[fp["distributor_id"].isin(did_list)].copy()
                if "All" not in filter_cat:
                    cid_list = categories.loc[categories["name"].isin(filter_cat), "category_id"].tolist()
                    fp = fp.loc[fp["category_id"].isin(cid_list)].copy()
                if len(dr) == 2:
                    fp["date"] = pd.to_datetime(fp["date"], errors="coerce")
                    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
                    fp = fp.loc[(fp["date"] >= start) & (fp["date"] <= end)].copy()

                if fp.empty:
                    st.info("No purchases matching filters")
                else:
                    rv = fp.merge(distributors, on="distributor_id").merge(categories, on="category_id")
                    rv = rv.rename(columns={"name_x": "Distributor", "name_y": "Category"})
                    st.dataframe(rv[["date", "Distributor", "Category", "qty", "rate", "amount"]], use_container_width=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Quantity", f"{fp['qty'].sum():.2f} L")
                    c2.metric("Total Amount", f"₹{fp['amount'].sum():.2f}")
                    c3.metric("Avg Rate", f"₹{fp['rate'].mean():.2f}/L")

# ------------------ DISTRIBUTOR PAYMENTS ------------------
elif menu == "💸 Distributor Payments":
    st.header("💸 Payments to Distributors")
    tab1, tab2, tab3 = st.tabs(["➕ Record Payment", "✏️ Payment History", "🔍 Filter Payments"])

    with tab1:
        if distributors.empty:
            st.warning("⚠️ Add distributors first")
        else:
            distributor = st.selectbox("Distributor", distributors["name"].tolist())
            did = int(distributors.loc[distributors["name"] == distributor, "distributor_id"].values[0])

            total_purchases = dist_purchases.loc[dist_purchases["distributor_id"] == did, "amount"].sum() if not dist_purchases.empty else 0
            total_paid = dist_payments.loc[dist_payments["distributor_id"] == did, "amount"].sum() if not dist_payments.empty else 0
            balance = total_purchases - total_paid
            st.metric(f"{distributor}'s Outstanding Balance", f"₹{balance:.2f}")

            col1, col2 = st.columns(2)
            with col1:
                payment_date = st.date_input("Payment Date", date.today(), key="dist_pay_date_input")
                amount = st.number_input("Payment Amount (₹)", min_value=0.0, step=100.0, key="dist_pay_amt_input")
            with col2:
                payment_mode = st.selectbox("Payment Mode", ["Cash", "UPI", "Bank Transfer", "Cheque", "Card"], key="dist_pay_mode_input")
                note = st.text_area("Note (optional)", key="dist_pay_note_input")

            if st.button("Record Payment", type="primary"):
                if amount > 0:
                    pid = int(dist_payments["payment_id"].max()) + 1 if not dist_payments.empty else 1
                    new_row = pd.DataFrame([[pid, str(payment_date), did, float(amount), payment_mode, note]], columns=dist_payments.columns)
                    dist_payments = pd.concat([dist_payments, new_row], ignore_index=True)
                    dist_payments.to_csv(DISTRIBUTOR_PAYMENTS_FILE, index=False)
                    st.success("✅ Payment recorded!")
                    st.rerun()

    with tab2:
        if dist_payments.empty:
            st.info("No payments recorded yet")
        else:
            pv = dist_payments.merge(distributors, on="distributor_id").rename(columns={"name": "Distributor"})
            st.dataframe(pv[["payment_id", "date", "Distributor", "amount", "payment_mode", "note"]], use_container_width=True)

            del_payment_id = st.number_input("Payment ID to delete", min_value=1, step=1, key="del_dist_payment_id")
            if st.button("🗑️ Delete Payment"):
                if del_payment_id in dist_payments["payment_id"].values:
                    dist_payments = dist_payments.loc[dist_payments["payment_id"] != del_payment_id].copy()
                    dist_payments.to_csv(DISTRIBUTOR_PAYMENTS_FILE, index=False)
                    st.success("Deleted!")
                    st.rerun()

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_dist = st.multiselect("Distributor(s)", ["All"] + distributors["name"].tolist(), default=["All"], key="filter_dist_pay")
        with col2:
            filter_mode = st.multiselect("Payment Mode", ["All", "Cash", "UPI", "Bank Transfer", "Cheque", "Card"], default=["All"], key="filter_dist_pay_mode")
        with col3:
            dr = st.date_input("Date Range", value=[], key="filter_dist_pay_date")

        if dist_payments.empty:
            st.info("No payment data available")
        else:
            fp = dist_payments.copy()
            if "All" not in filter_dist:
                did_list = distributors.loc[distributors["name"].isin(filter_dist), "distributor_id"].tolist()
                fp = fp.loc[fp["distributor_id"].isin(did_list)].copy()
            if "All" not in filter_mode:
                fp = fp.loc[fp["payment_mode"].isin(filter_mode)].copy()
            if len(dr) == 2:
                fp["date"] = pd.to_datetime(fp["date"], errors="coerce")
                start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
                fp = fp.loc[(fp["date"] >= start) & (fp["date"] <= end)].copy()

            if fp.empty:
                st.info("No payments matching filters")
            else:
                rv = fp.merge(distributors, on="distributor_id").rename(columns={"name": "Distributor"})
                st.dataframe(rv[["date", "Distributor", "amount", "payment_mode", "note"]], use_container_width=True)
                c1, c2 = st.columns(2)
                c1.metric("Total Payments", f"₹{fp['amount'].sum():.2f}")
                c2.metric("Transactions", len(fp))

# ------------------ MILK WASTAGE ------------------
elif menu == "🗑️ Milk Wastage":
    st.header("🗑️ Daily Milk Wastage Tracking")
    tab1, tab2, tab3 = st.tabs(["➕ Record Wastage", "📊 Wastage History", "🔍 Filter Wastage"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            wastage_date = st.date_input("Date", date.today())
            category = st.selectbox("Milk Category", categories["name"].tolist() if not categories.empty else [])
            qty = st.number_input("Quantity Wasted (Liters)", min_value=0.0, step=0.1)
        with col2:
            reason = st.selectbox("Reason", ["Spoilage", "Spill", "Expired", "Quality Issue", "Other"])
            if reason == "Other":
                reason = st.text_input("Specify reason")

            estimated_loss = 0.0
            if qty > 0 and not dist_purchases.empty and not categories.empty and category:
                cid = int(categories.loc[categories["name"] == category, "category_id"].values[0])
                avg_rate = dist_purchases.loc[dist_purchases["category_id"] == cid, "rate"].mean()
                estimated_loss = float(qty) * float(avg_rate) if pd.notna(avg_rate) else 0.0
            else:
                estimated_loss = st.number_input("Estimated Loss (₹)", min_value=0.0, step=10.0)

        if st.button("Record Wastage", type="primary"):
            if qty > 0 and category:
                cid = int(categories.loc[categories["name"] == category, "category_id"].values[0])
                wid = int(wastage["wastage_id"].max()) + 1 if not wastage.empty else 1
                new_row = pd.DataFrame([[wid, str(wastage_date), cid, float(qty), reason, float(estimated_loss)]], columns=wastage.columns)
                wastage = pd.concat([wastage, new_row], ignore_index=True)
                wastage.to_csv(WASTAGE_FILE, index=False)
                st.success("✅ Wastage recorded!")
                st.rerun()

    with tab2:
        if wastage.empty:
            st.info("No wastage recorded yet")
        else:
            wv = wastage.merge(categories, on="category_id").rename(columns={"name": "Category"})
            st.dataframe(wv[["wastage_id", "date", "Category", "qty", "reason", "estimated_loss"]], use_container_width=True)

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_cat = st.multiselect("Category(s)", ["All"] + (categories["name"].tolist() if not categories.empty else []), default=["All"], key="waste_filter_cat")
        with col2:
            filter_reason = st.multiselect("Reason", ["All", "Spoilage", "Spill", "Expired", "Quality Issue", "Other"], default=["All"], key="waste_filter_reason")
        with col3:
            dr = st.date_input("Date Range", value=[], key="waste_filter_date")

        if wastage.empty:
            st.info("No wastage data available")
        else:
            fw = wastage.copy()
            if "All" not in filter_cat and not categories.empty:
                cid_list = categories.loc[categories["name"].isin(filter_cat), "category_id"].tolist()
                fw = fw.loc[fw["category_id"].isin(cid_list)].copy()
            if "All" not in filter_reason:
                fw = fw.loc[fw["reason"].isin(filter_reason)].copy()
            if len(dr) == 2:
                fw["date"] = pd.to_datetime(fw["date"], errors="coerce")
                start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
                fw = fw.loc[(fw["date"] >= start) & (fw["date"] <= end)].copy()

            if fw.empty:
                st.info("No wastage matching filters")
            else:
                rv = fw.merge(categories, on="category_id").rename(columns={"name": "Category"})
                st.dataframe(rv[["date", "Category", "qty", "reason", "estimated_loss"]], use_container_width=True)

# ------------------ EXPENSES (ONLY ONCE, FIXED) ------------------
elif menu == "💼 Expenses":
    st.header("💼 Business Expenses Management")
    tab1, tab2, tab3 = st.tabs(["➕ Add Expense", "📊 Expense Tracking", "🔍 Filter Expenses"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            expense_date = st.date_input("Date", date.today())
            category = st.selectbox(
                "Expense Category",
                ["Fuel", "Electricity", "Maintenance", "Salary", "Rent", "Transportation", "Packaging", "Marketing", "Other"],
            )
            if category == "Other":
                category = st.text_input("Specify category")
        with col2:
            description = st.text_input("Description")
            amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0)
            payment_mode = st.selectbox("Payment Mode", ["Cash", "UPI", "Bank Transfer", "Cheque", "Card"])
            paid = st.checkbox("Paid", value=True)

        if st.button("Add Expense", type="primary"):
            if amount > 0:
                eid = int(expenses["expense_id"].max()) + 1 if not expenses.empty else 1
                new_row = pd.DataFrame([[eid, str(expense_date), category, description, float(amount), payment_mode, bool(paid)]], columns=expenses.columns)
                expenses = pd.concat([expenses, new_row], ignore_index=True)
                expenses.to_csv(EXPENSES_FILE, index=False)
                st.success("✅ Expense added!")
                st.rerun()

    with tab2:
        if expenses.empty:
            st.info("No expenses recorded yet")
        else:
            ev = expenses.copy()
            ev["paid"] = ev["paid"].apply(lambda x: "✅ Paid" if bool(x) else "⏳ Pending")
            st.dataframe(ev, use_container_width=True)

            paid_amount = expenses.loc[expenses["paid"] == True, "amount"].sum()
            pending_amount = expenses.loc[expenses["paid"] == False, "amount"].sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Expenses", f"₹{expenses['amount'].sum():.2f}")
            c2.metric("Paid", f"₹{paid_amount:.2f}")
            c3.metric("Pending", f"₹{pending_amount:.2f}")
            c4.metric("Total Records", len(expenses))

            del_expense_id = st.number_input("Expense ID to delete", min_value=1, step=1, key="del_expense_id")
            if st.button("🗑️ Delete Expense"):
                if del_expense_id in expenses["expense_id"].values:
                    expenses = expenses.loc[expenses["expense_id"] != del_expense_id].copy()
                    expenses.to_csv(EXPENSES_FILE, index=False)
                    st.success("Deleted!")
                    st.rerun()

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_cat = st.multiselect(
                "Expense Category",
                ["All", "Fuel", "Electricity", "Maintenance", "Salary", "Rent", "Transportation", "Packaging", "Marketing", "Other"],
                default=["All"],
                key="exp_filter_cat",
            )
        with col2:
            filter_mode = st.multiselect("Payment Mode", ["All", "Cash", "UPI", "Bank Transfer", "Cheque", "Card"], default=["All"], key="exp_filter_mode")
            filter_paid = st.selectbox("Payment Status", ["All", "Paid", "Pending"], key="exp_filter_paid")
        with col3:
            dr = st.date_input("Date Range", value=[], key="exp_filter_date")

        if expenses.empty:
            st.info("No expense data available")
        else:
            fe = expenses.copy()
            if "All" not in filter_cat:
                fe = fe.loc[fe["category"].isin(filter_cat)].copy()
            if "All" not in filter_mode:
                fe = fe.loc[fe["payment_mode"].isin(filter_mode)].copy()
            if filter_paid == "Paid":
                fe = fe.loc[fe["paid"] == True].copy()
            elif filter_paid == "Pending":
                fe = fe.loc[fe["paid"] == False].copy()
            if len(dr) == 2:
                fe["date"] = pd.to_datetime(fe["date"], errors="coerce")
                start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
                fe = fe.loc[(fe["date"] >= start) & (fe["date"] <= end)].copy()

            if fe.empty:
                st.info("No expenses matching filters")
            else:
                st.dataframe(fe, use_container_width=True)

# ------------------ GENERATE BILL (FIXED, NO tab3 ERROR) ------------------
elif menu == "🧾 Generate Bill":
    st.header("🧾 Generate Customer Bill")

    if retailers.empty:
        st.warning("⚠️ Please add retailers first")
    else:
        col1, col2 = st.columns(2)
        with col1:
            retailer = st.selectbox("Select Retailer", retailers["name"].tolist())
            retailer_id = int(retailers.loc[retailers["name"] == retailer, "retailer_id"].values[0])
        with col2:
            date_range_bill = st.date_input("Select Date Range for Bill", value=[])

        if st.button("Generate Bill", type="primary"):
            if len(date_range_bill) != 2:
                st.info("👆 Please select a date range (start and end).")
            else:
                start = pd.to_datetime(date_range_bill[0])
                end = pd.to_datetime(date_range_bill[1])

                retailer_data = retailers.loc[retailers["retailer_id"] == retailer_id].iloc[0]

                re = entries.loc[entries["retailer_id"] == retailer_id].copy()
                if re.empty:
                    st.info(f"No sales records found for {retailer}.")
                else:
                    re["date"] = pd.to_datetime(re["date"], errors="coerce")

                    bill_entries = re.loc[(re["date"] >= start) & (re["date"] <= end)].copy()
                    if bill_entries.empty:
                        st.info(f"No entries found for {retailer} in selected date range.")
                    else:
                        prev_entries = re.loc[re["date"] < start]
                        previous_sales = prev_entries["amount"].sum()

                        pay = payments.loc[payments["retailer_id"] == retailer_id].copy()
                        if not pay.empty:
                            pay["date"] = pd.to_datetime(pay["date"], errors="coerce")
                            previous_payments = pay.loc[pay["date"] < start, "amount"].sum()
                            period_payments = pay.loc[(pay["date"] >= start) & (pay["date"] <= end), "amount"].sum()
                        else:
                            previous_payments = 0.0
                            period_payments = 0.0

                        previous_balance = previous_sales - previous_payments
                        current_total = bill_entries["amount"].sum()
                        final_balance = previous_balance + current_total - period_payments

                        bill_details = bill_entries.merge(categories, on="category_id").rename(columns={"name": "Milk Type"})
                        bill_details = bill_details.sort_values("date")

                        # Display bill
                        st.subheader("🧾 Invoice Preview")
                        st.write(f"**Bill To:** {retailer_data['name']}")
                        st.write(f"**Contact:** {retailer_data.get('contact','')}")
                        st.write(f"**Address:** {retailer_data.get('address','')}")
                        st.write(f"**Period:** {date_range_bill[0]} to {date_range_bill[1]}")
                        st.divider()

                        show_df = bill_details[["date", "Milk Type", "qty", "rate", "amount"]].copy()
                        show_df["date"] = show_df["date"].dt.strftime("%d-%m-%Y")
                        st.dataframe(show_df, use_container_width=True)

                        st.divider()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Previous Balance", f"₹{previous_balance:.2f}")
                        c2.metric("Current Period Total", f"₹{current_total:.2f}")
                        c3.metric("Payments Received", f"₹{period_payments:.2f}")
                        c4.metric("Total Outstanding", f"₹{final_balance:.2f}", delta="Settled" if final_balance <= 0 else None)

                        # Download text bill
                        st.divider()
                        st.subheader("📥 Download Bill")

                        text_bill = []
                        text_bill.append("==============================================")
                        text_bill.append("            MILK BUSINESS - INVOICE")
                        text_bill.append("==============================================")
                        text_bill.append(f"Bill To: {retailer_data['name']}")
                        text_bill.append(f"Contact: {retailer_data.get('contact','')}")
                        text_bill.append(f"Address: {retailer_data.get('address','')}")
                        text_bill.append(f"Period: {date_range_bill[0]} to {date_range_bill[1]}")
                        text_bill.append("----------------------------------------------")
                        text_bill.append("Date        Milk Type        Qty    Rate   Amount")
                        text_bill.append("----------------------------------------------")

                        for _, r in bill_details.iterrows():
                            d = pd.to_datetime(r["date"]).strftime("%d-%m-%Y")
                            mt = str(r["Milk Type"])[:14].ljust(14)
                            qty = f"{float(r['qty']):>6.2f}"
                            rate = f"{float(r['rate']):>6.2f}"
                            amt = f"{float(r['amount']):>8.2f}"
                            text_bill.append(f"{d}  {mt}  {qty}  {rate}  {amt}")

                        text_bill.append("----------------------------------------------")
                        text_bill.append(f"Previous Balance:      ₹{previous_balance:>10.2f}")
                        text_bill.append(f"Current Period Total:  ₹{current_total:>10.2f}")
                        text_bill.append(f"Payments Received:   - ₹{period_payments:>10.2f}")
                        text_bill.append("----------------------------------------------")
                        text_bill.append(f"TOTAL OUTSTANDING:     ₹{final_balance:>10.2f}")
                        text_bill.append("==============================================")

                        st.download_button(
                            label="📄 Download Bill as Text",
                            data="\n".join(text_bill),
                            file_name=f"bill_{retailer}_{date.today().strftime('%Y%m%d')}.txt",
                            mime="text/plain",
                        )
