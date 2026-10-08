"""
Data loading and cleaning for the UCI "Online Retail II" dataset.

The raw file contains ~1.07M invoice lines (Dec. 2009 - Dec. 2011) from a
UK-based online retailer selling giftware, mostly to wholesalers.

Cleaning steps (each one is logged in a data quality report):
1. Exact duplicate lines are removed.
2. Cancellations (invoices starting with "C") are removed, and so are the
   original purchase lines they cancel (same customer, product and quantity).
3. Accounting adjustments ("A" invoices) are removed.
4. Non-product lines (postage, fees, discounts, manual entries, gift vouchers,
   tests...) are removed.
5. Lines with a quantity <= 0 or a unit price <= 0 are removed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "C2", "D", "S", "B", "BANK CHARGES", "ADJUST",
    "ADJUST2", "AMAZONFEE", "PADS", "CRUK", "TEST001", "TEST002",
}


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read both yearly sheets of the Excel file (or a cached parquet copy)."""
    path = Path(path) if path else RAW_DIR / "online_retail_II.xlsx"
    cache = path.with_suffix(".parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine="calamine")
    except (ImportError, ValueError):
        sheets = pd.read_excel(path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df.rename(columns={"Customer ID": "CustomerID"})
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype("string")
    df["Country"] = df["Country"].astype(str)
    try:
        df.to_parquet(cache, index=False)
    except Exception:  # pyarrow not installed: no cache, no problem
        pass
    return df


def _is_non_product(code: pd.Series) -> pd.Series:
    upper = code.str.upper()
    return upper.isin(NON_PRODUCT_CODES) | upper.str.startswith("GIFT_")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean product sales lines, data quality report)."""
    report = []

    def log(step: str, before: int, after: int):
        report.append({"step": step, "rows_removed": before - after, "rows_left": after})

    n = len(df)
    report.append({"step": "Raw data", "rows_removed": 0, "rows_left": n})

    df = df.drop_duplicates()
    log("Exact duplicates", n, len(df)); n = len(df)

    is_cancel = df["Invoice"].str.startswith("C")
    cancels = df[is_cancel]
    df = df[~is_cancel]
    log("Cancellation lines (invoice 'C...')", n, len(df)); n = len(df)

    # Remove the purchase lines that were cancelled afterwards
    # (e.g. the famous 80,995-unit order cancelled a few minutes later).
    key = ["CustomerID", "StockCode", "abs_qty"]
    c = cancels.dropna(subset=["CustomerID"]).assign(abs_qty=lambda x: x["Quantity"].abs())
    c = c.groupby(key).size().rename("n_cancel").reset_index()
    p = df.assign(abs_qty=df["Quantity"].abs(), _row=range(len(df)))
    p["_rank"] = p.groupby(key, dropna=False).cumcount() + 1
    p = p.merge(c, on=key, how="left")
    cancelled = p["_rank"] <= p["n_cancel"].fillna(0)
    df = p.loc[~cancelled].drop(columns=["abs_qty", "_row", "_rank", "n_cancel"])
    log("Purchases matched by a later cancellation", n, len(df)); n = len(df)

    df = df[~df["Invoice"].str.startswith("A")]
    log("Bad-debt adjustments (invoice 'A...')", n, len(df)); n = len(df)

    df = df[~_is_non_product(df["StockCode"])]
    log("Non-product lines (postage, fees, discounts, vouchers...)", n, len(df)); n = len(df)

    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    log("Quantity <= 0 or price <= 0", n, len(df)); n = len(df)

    df = df.copy()
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["Date"] = df["InvoiceDate"].dt.normalize()
    return df.reset_index(drop=True), pd.DataFrame(report)


def daily_sales(df: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Daily revenue / orders / customers on a continuous calendar (missing days = 0)."""
    if country:
        df = df[df["Country"] == country]
    daily = df.groupby("Date").agg(
        revenue=("Revenue", "sum"),
        orders=("Invoice", "nunique"),
        customers=("CustomerID", "nunique"),
        units=("Quantity", "sum"),
    )
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full, fill_value=0)
    daily.index.name = "date"
    return daily.reset_index()


APP_COUNTRIES = ["United Kingdom", "EIRE", "Netherlands", "Germany", "France"]


def daily_sales_by_country(df: pd.DataFrame, countries=APP_COUNTRIES) -> pd.DataFrame:
    """Daily sales for the whole business ("All countries") and the main markets,
    on the same continuous calendar."""
    calendar = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    frames = []
    for c in [None, *countries]:
        d = daily_sales(df, c).set_index("date").reindex(calendar, fill_value=0)
        d.index.name = "date"
        frames.append(d.reset_index().assign(country=c or "All countries"))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    sales, quality = clean(raw)
    print(quality.to_string(index=False))
    daily_sales(sales).to_csv(PROCESSED_DIR / "daily_sales.csv", index=False)
    daily_sales_by_country(sales).to_csv(PROCESSED_DIR / "daily_sales_by_country.csv", index=False)
    print("Saved daily_sales.csv and daily_sales_by_country.csv in", PROCESSED_DIR)
