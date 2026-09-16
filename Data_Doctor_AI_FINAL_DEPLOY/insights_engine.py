from __future__ import annotations
import pandas as pd

def summarize_business(df: pd.DataFrame, mapping: dict[str,str]) -> dict:
    out={"positives":[],"attention":[],"actions":[]}
    if "revenue" in df:
        rev=pd.to_numeric(df.revenue,errors="coerce").sum(); out["positives"].append(f"Total revenue is {rev:,.2f} across {len(df):,} records.")
    if "profit" in df:
        profit=pd.to_numeric(df.profit,errors="coerce").sum(); out["positives"].append(f"Recorded profit is {profit:,.2f}.")
        if "revenue" in df:
            rev=pd.to_numeric(df.revenue,errors="coerce").sum(); margin=100*profit/rev if rev else 0; out["positives"].append(f"Overall margin is {margin:.1f}%.")
    if "item" in df and "revenue" in df:
        q=df.groupby("item",dropna=True).revenue.sum().sort_values(ascending=False)
        if not q.empty:
            out["positives"].append(f"Top revenue driver is {q.index[0]} at {q.iloc[0]:,.2f}.")
            if len(q)>=3:
                out["actions"].append(f"Review the bottom-selling items: {', '.join(map(str,q.head(0).index))}" if False else f"Review the lowest-revenue items before giving them more shelf, menu or marketing space.")
    if "quantity" in df and "item" in df:
        q=df.groupby("item").quantity.sum().sort_values(ascending=False)
        if not q.empty: out["positives"].append(f"Most units/items come from {q.index[0]} ({q.iloc[0]:,.0f}).")
    if "stock" in df and "item" in df:
        s=pd.to_numeric(df.stock,errors="coerce"); low=df.loc[s<=0,"item"].dropna().astype(str).unique().tolist()
        if low:
            out["attention"].append(f"{len(low)} item(s) show zero or negative stock values.")
            out["actions"].append("Check replenishment and stock recording for the affected items.")
    if "datetime" in df:
        dt=pd.to_datetime(df.datetime,errors="coerce").dropna()
        if not dt.empty:
            out["positives"].append(f"The dataset spans {dt.min().date()} to {dt.max().date()}.")
            if "revenue" in df:
                tmp=df.copy(); tmp["_dt"]=pd.to_datetime(tmp.datetime,errors="coerce"); tmp["_rev"]=pd.to_numeric(tmp.revenue,errors="coerce").fillna(0); byh=tmp.dropna(subset=["_dt"]).groupby(tmp.dropna(subset=["_dt"])["_dt"].dt.hour)._rev.sum();
                if not byh.empty: out["positives"].append(f"Peak revenue hour is around {int(byh.idxmax()):02d}:00.")
    if "profit" in df and "item" in df:
        g=df.groupby("item").profit.sum().sort_values();
        if not g.empty and g.iloc[0] < 0:
            out["attention"].append(f"Some items are producing negative recorded profit; review pricing and cost data.")
            out["actions"].append("Review negative-profit items and verify purchase costs, prices and discounts.")
    if not out["actions"]:
        out["actions"].append("Focus on the biggest revenue drivers first, then investigate weak or unusual segments.")
    return out
