#______importing libraries__________
import pandas as pd

#______loading customer payments data__________
payments_df = pd.read_csv("data/customerpayments.csv")

#______filtering successful payments________
payments_df = payments_df[payments_df["status"] == "success"]

#______converting date to datetime________
payments_df["date"] = pd.to_datetime(payments_df["date"])

#______setting date as week start using Monday as anchor________
payments_df["week_start"] = payments_df["date"].dt.to_period("W-SUN").dt.start_time

#______aggregating amount_applied to weekly revenue________
weekly_revenue = payments_df.groupby("week_start")["amount_applied"].sum().reset_index()
weekly_revenue.columns = ["week_start", "revenue"]

#______creating a complete date range with no missing weeks________
all_weeks = pd.date_range(start="2022-01-01", end="2025-12-31", freq="W-MON")
all_weeks_df = pd.DataFrame({"week_start": all_weeks})

#______merging and filling missing weeks with zero________
weekly_revenue = all_weeks_df.merge(weekly_revenue, on="week_start", how="left")
weekly_revenue.columns = ["week_start", "revenue"]
weekly_revenue["revenue"] = weekly_revenue["revenue"].fillna(0)

#______sorting by date to ensure chronological order________
weekly_revenue = weekly_revenue.sort_values("week_start").reset_index(drop=True)

#______saving weekly revenue to data folder__________
weekly_revenue.to_csv("data/weekly_revenue.csv", index=False)

print(f"Total weeks: {len(weekly_revenue)}")
print(weekly_revenue.head())
