#______importing libraries_______
import pandas as pd
import numpy as np 

#______loading weekly revenue data__________
df=pd.read_csv("data/weekly_revenue.csv")
df["week_start"]=pd.to_datetime(df["week_start"])

#______lag features__________
df["lag_1"]=df["revenue"].shift(1)
df["lag_2"]=df["revenue"].shift(2)
df["lag_4"]=df["revenue"].shift(4)
df["lag_8"]=df["revenue"].shift(8)
df["lag_12"]=df["revenue"].shift(12)
df["lag_52"]=df["revenue"].shift(52)

#______rolling statistics features___________
df["rolling_mean_4"]=df["revenue"].shift(1).rolling(window=4).mean()
df["rolling_mean_12"]=df["revenue"].shift(1).rolling(window=12).mean()
df["rolling_mean_26"]=df["revenue"].shift(1).rolling(window=26).mean()
df["rolling_std_4"] = df["revenue"].shift(1).rolling(window=4).std()
df["rolling_std_12"] = df["revenue"].shift(1).rolling(window=12).std()
df["rolling_max_4"] = df["revenue"].shift(1).rolling(window=4).max()

#______calendar features__________
df["week_of_year"]=df["week_start"].dt.isocalendar().week.astype(int)
df["month"]=df["week_start"].dt.month
df["quarter"]=df["week_start"].dt.quarter
df["is_month_end"]=df["week_start"].dt.is_month_end.astype(int)
df["is_quarter_end"]=df["quarter"].diff().fillna(0).astype(bool).astype(int)
df["is_december"]=(df["month"]==12).astype(int)
df["is_january"]=(df["month"]==1).astype(int)

#______Trend features__________
df["weeks_elapsed"]=range(len(df))
df["wow_change"]=df["revenue"].pct_change(1)
df["revenue_vs_4w_avg"]=df["revenue"]/df["rolling_mean_4"]
df["revenue_pct_change_4"]=df["revenue"].pct_change(4)
df["revenue_pct_change_52"]=df["revenue"].pct_change(52)

#______dropping rows wiwth NaN values__________
df=df.replace([np.inf,-np.inf],np.nan)
df=df.dropna()
df=df.reset_index(drop=True)

#______saving featured dataframe to data folder__________
df.to_csv("data/featured_weekly_revenue.csv",index=False)

print(f"Total weeks after dropping NaN:{len(df)}")
print(df.head())
