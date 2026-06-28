import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

CUSTOMERS_FILE = "data/customers.csv"
INVOICES_FILE = "data/invoices.csv"
PAYMENTS_FILE = "data/customerpayments.csv"

OUTPUT_FILE = "data/weekly_business_metrics.csv"


# ==========================================================
# Load Data
# ==========================================================


def load_data():
    customers_df = pd.read_csv(CUSTOMERS_FILE, parse_dates=["join_date", "churn_date"])

    invoices_df = pd.read_csv(INVOICES_FILE, parse_dates=["invoice_date", "due_date"])

    payments_df = pd.read_csv(PAYMENTS_FILE, parse_dates=["payment_date"])

    return customers_df, invoices_df, payments_df


# ==========================================================
# Validation
# ==========================================================


def validate_input(customers_df, invoices_df, payments_df):
    print("\nValidating input files...\n")

    assert not customers_df.empty
    assert not invoices_df.empty
    assert not payments_df.empty
    assert customers_df["customer_id"].is_unique
    assert invoices_df["invoice_id"].is_unique
    assert payments_df["payment_id"].is_unique

    print("Validation passed.")


# ==========================================================
# Date Helpers
# ==========================================================


def week_start(date_series):
    return (
        date_series - pd.to_timedelta(date_series.dt.weekday, unit="D")
    ).dt.normalize()


def build_week_calendar(invoices_df):
    first_week = week_start(invoices_df["invoice_date"]).min()
    last_week = week_start(invoices_df["invoice_date"]).max()

    calendar = pd.DataFrame(
        {"week": pd.date_range(start=first_week, end=last_week, freq="W-MON")}
    )

    return calendar


# ==========================================================
# Revenue Aggregation
# ==========================================================


def aggregate_revenue(invoices_df):
    revenue = invoices_df.copy()
    revenue["week"] = week_start(revenue["invoice_date"])

    revenue = revenue.groupby("week", as_index=False).agg(
        weekly_revenue=("invoice_amount", "sum"),
        invoice_count=("invoice_id", "count"),
        average_invoice=("invoice_amount", "mean"),
    )

    return revenue


# ==========================================================
# Payment Aggregation
# ==========================================================


def aggregate_payments(payments_df):
    payments = payments_df.copy()
    payments["week"] = week_start(payments["payment_date"])

    payments = payments.groupby("week", as_index=False).agg(
        weekly_payments=("payment_amount", "sum"),
        payment_count=("payment_id", "count"),
        average_payment=("payment_amount", "mean"),
    )

    return payments


# ==========================================================
# Customer Aggregation
# ==========================================================


def aggregate_new_customers(customers_df):
    new_customers = customers_df.copy()
    new_customers["week"] = week_start(new_customers["join_date"])

    new_customers = new_customers.groupby("week", as_index=False).agg(
        new_customers=("customer_id", "count")
    )

    return new_customers


def aggregate_churn(customers_df):
    churn = customers_df.dropna(subset=["churn_date"]).copy()
    churn["week"] = week_start(churn["churn_date"])

    churn = churn.groupby("week", as_index=False).agg(
        churned_customers=("customer_id", "count")
    )

    return churn


def aggregate_active_customers(customers_df, calendar):
    active_counts = []

    for week in calendar["week"]:
        active = customers_df[
            (customers_df["join_date"] <= week)
            & (customers_df["churn_date"].isna() | (customers_df["churn_date"] > week))
        ].shape[0]
        active_counts.append(active)

    active_df = pd.DataFrame(
        {"week": calendar["week"], "active_customers": active_counts}
    )

    return active_df


# ==========================================================
# Merge Weekly Metrics
# ==========================================================


def build_weekly_dataset(
    calendar, revenue, payments, new_customers, churn, active_customers
):
    weekly = calendar.copy()
    weekly = weekly.merge(revenue, on="week", how="left")
    weekly = weekly.merge(payments, on="week", how="left")
    weekly = weekly.merge(new_customers, on="week", how="left")
    weekly = weekly.merge(churn, on="week", how="left")
    weekly = weekly.merge(active_customers, on="week", how="left")

    count_columns = [
        "weekly_revenue",
        "weekly_payments",
        "invoice_count",
        "payment_count",
        "new_customers",
        "churned_customers"
    ]
    weekly[count_columns] = (
        weekly[count_columns]
        .fillna(0)
    )
    return weekly 

# ==========================================================
# Export
# ==========================================================


def export_dataset(weekly_df):
    weekly_df["week"] = pd.to_datetime(
        weekly_df["week"]
    )
    weekly_df = weekly_df.sort_values(
        "week"
    ).reset_index(
        drop=True
    )
    weekly_df.to_csv(
        OUTPUT_FILE,
        index=False
    )
    return weekly_df

# ==========================================================
# Output Validation
# ==========================================================


def validate_output(weekly_df):
    print("\nValidating weekly dataset...\n")

    assert not weekly_df.empty
    assert weekly_df["week"].is_unique
    assert (weekly_df["weekly_revenue"] >= 0).all()
    assert (weekly_df["weekly_payments"] >= 0).all()
    assert (weekly_df["invoice_count"] >= 0).all()
    assert (weekly_df["payment_count"] >= 0).all()
    assert (weekly_df["active_customers"] >= 0).all()

    print("Weekly dataset validation passed.")


# ==========================================================
# Summary
# ==========================================================


def print_summary(weekly_df):
    print("\n==============================")
    print("Weekly Dataset Summary")
    print("==============================")
    print(f"Weeks               : {len(weekly_df)}")
    print(f"Total Revenue       : {weekly_df['weekly_revenue'].sum():,.2f}")
    print(f"Average Weekly Rev. : {weekly_df['weekly_revenue'].mean():,.2f}")
    print(f"Average Customers   : {weekly_df['active_customers'].mean():.0f}")
    print(f"Average Invoices    : {weekly_df['invoice_count'].mean():.1f}")
    print(f"Average Payments    : {weekly_df['payment_count'].mean():.1f}")
    print("==============================")


# ==========================================================
# Main
# ==========================================================


def main():
    print("\nStarting aggregation...\n")

    customers_df, invoices_df, payments_df = load_data()
    validate_input(customers_df, invoices_df, payments_df)

    calendar = build_week_calendar(invoices_df)
    revenue = aggregate_revenue(invoices_df)
    payments = aggregate_payments(payments_df)
    new_customers = aggregate_new_customers(customers_df)
    churn = aggregate_churn(customers_df)
    active_customers = aggregate_active_customers(customers_df, calendar)

    weekly_df = build_weekly_dataset(
        calendar, revenue, payments, new_customers, churn, active_customers
    )

    weekly_df = export_dataset(weekly_df)

    validate_output(weekly_df)
    print_summary(weekly_df)

    print("\nweekly_business_metrics.csv generated successfully.")


if __name__ == "__main__":
    main()  