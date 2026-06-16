import pandas as pd 
invoices_df=pd.read_csv("data/invoices.csv")

#______filtering paid invoices____________
invoices_df=invoices_df[invoices_df["status"]=="paid"]

#______converting payment_date to datetime______
invoices_df["payment_date"]=pd.to_datetime(invoices_df["payment_date"])

#______setting payment_date as week start using Monday as anchor___________
invoices_df["week_start"]=invoices_df["payment_date"].dt.to_period("W-SUN").dt.start_time

#______aggregating amount_paid to weekly revenue _________________
weekly_revenue=invoices_df.groupby("week_start")["amount_paid"].sum().reset_index()
weekly_revenue.columns=["week_start","revenue"]

#______creating a complete data range with no missing weeks___________________
all_weeks=pd.date_range(start="2022-01-01",end="2025-12-31",freq="W-MON")
all_weeks_df=pd.DataFrame({"week_start":all_weeks})

#______merging and filling the missing weeks with zero_______________
weekly_revenue=all_weeks_df.merge(weekly_revenue,on="week_start",how="left")
weekly_revenue.columns = ["week_start", "revenue"]
weekly_revenue["revenue"]=weekly_revenue["revenue"].fillna(0)

#______sorting by date to ensure chronological order_______________
weekly_revenue=weekly_revenue.sort_values("week_start").reset_index(drop=True)

#______saving weekly revenue to data folder___________________
weekly_revenue.to_csv("data/weekly_revenue.csv",index=False)

print(f"Total weeks::{len(weekly_revenue)}")
print(weekly_revenue.head())
