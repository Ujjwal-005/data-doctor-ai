from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

def candidate_targets(df: pd.DataFrame) -> list[str]:
    """Return plausible prediction targets without treating IDs/descriptors as outcomes."""
    excluded_exact = {
        "id","item","product","category","location","store","customer",
        "patient","employee","gender","status","datetime","date","time",
        "transaction_id","order_id","invoice_id","customer_id","patient_id","employee_id"
    }
    out=[]
    for c in df.columns:
        name=str(c).strip().lower()
        normalized=re.sub(r"[^a-z0-9]+","_",name).strip("_") if name else name
        if c in excluded_exact or normalized in excluded_exact:
            continue
        if normalized.endswith("_id") or normalized in {"id","invoice_id","order_number","order_no"}:
            continue
        s=df[c]
        nun=int(s.nunique(dropna=True))
        if nun < 2:
            continue
        if pd.api.types.is_numeric_dtype(s):
            out.append(c)
        elif 2 <= nun <= min(12, max(3, int(len(df)*0.1))):
            out.append(c)
    priority=[]
    for c in out:
        n=str(c).lower()
        score=0
        if any(k in n for k in ["revenue","sales","profit","income","amount","quantity","qty","rating","score","salary","price","cost"]): score+=10
        if any(k in n for k in ["target","outcome","result","default","attrition","churn","delay","status"]): score+=8
        priority.append((score,str(c),c))
    priority.sort(key=lambda x:(-x[0],x[1]))
    return [x[2] for x in priority[:25]]




def _leakage_features(df: pd.DataFrame, target: str) -> list[str]:
    """Find predictors that are obvious deterministic/derived sources of the target.

    This is intentionally conservative: it targets common business-measure leakage
    patterns (profit from revenue/cost/quantity; totals from line-item measures) and
    near-perfect one-feature linear relationships.
    """
    drop=[]
    tname=re.sub(r"[^a-z0-9]+","_",str(target).lower()).strip("_")
    if tname in {"profit","gross_profit","net_profit","gross_income","profit_amount"}:
        for c in df.columns:
            n=re.sub(r"[^a-z0-9]+","_",str(c).lower()).strip("_")
            if n in {"revenue","sales","sales_amount","net_sales","total","total_amount","amount","cost","cogs","quantity","qty","price","unit_price"}:
                if c != target and pd.api.types.is_numeric_dtype(df[c]):
                    drop.append(c)
    # Generic near-perfect linear leakage check.
    if pd.api.types.is_numeric_dtype(df[target]):
        y=pd.to_numeric(df[target],errors="coerce")
        for c in df.columns:
            if c==target or c in drop or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            x=pd.to_numeric(df[c],errors="coerce")
            mask=x.notna() & y.notna()
            if mask.sum()<30:
                continue
            try:
                corr=float(abs(x[mask].corr(y[mask])))
            except Exception:
                continue
            if corr>=0.9999:
                drop.append(c)
    return list(dict.fromkeys(drop))

def _prepare_supervised(df: pd.DataFrame, target: str):
    x=df.drop(columns=[target]).copy(); y=df[target].copy()
    # Remove obvious ID columns and business-measure leakage from predictors.
    drop=[]
    leakage=_leakage_features(df,target)
    for c in x.columns:
        name=str(c).lower()
        if c != target and (name.endswith("_id") or name in {"id","invoice id","order id","transaction id"}): drop.append(c)
    drop=list(dict.fromkeys(drop+leakage))
    x=x.drop(columns=drop,errors="ignore")
    # Dates become compact calendar features rather than raw datetimes.
    for c in list(x.columns):
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x[f"{c}__year"]=x[c].dt.year
            x[f"{c}__month"]=x[c].dt.month
            x[f"{c}__dow"]=x[c].dt.dayofweek
            x=x.drop(columns=[c])
    numeric=[c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical=[c for c in x.columns if c not in numeric]
    numeric_pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    categorical_pipe=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore", min_frequency=2, max_categories=50))])
    prep=ColumnTransformer([("num",numeric_pipe,numeric),("cat",categorical_pipe,categorical)],remainder="drop")
    return x,y,prep


def auto_supervised(df: pd.DataFrame, target: str) -> dict:
    x,y,prep=_prepare_supervised(df,target)
    mask=y.notna()
    x,y=x.loc[mask],y.loc[mask]
    if len(x)<30: raise ValueError("Need at least 30 usable rows for supervised model comparison.")
    # Keep interactive AutoML responsive on very large business files.
    if len(x) > 50000:
        sample_idx = x.sample(50000, random_state=42).index
        x, y = x.loc[sample_idx], y.loc[sample_idx]
        x, y = x.sort_index(), y.loc[x.index]
    is_num=pd.api.types.is_numeric_dtype(y)
    if is_num:
        y=pd.to_numeric(y,errors="coerce"); mask=y.notna(); x,y=x.loc[mask],y.loc[mask]
        models={
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=180,min_samples_leaf=2,random_state=42,n_jobs=1),
            "Extra Trees": ExtraTreesRegressor(n_estimators=180,min_samples_leaf=2,random_state=42,n_jobs=1),
        }
        split=int(len(x)*0.8); split=max(20,min(split,len(x)-5))
        results=[]
        for name,model in models.items():
            pipe=Pipeline([("prep",prep),("model",model)])
            pipe.fit(x.iloc[:split],y.iloc[:split]); pred=pipe.predict(x.iloc[split:])
            results.append({"model":name,"MAE":float(mean_absolute_error(y.iloc[split:],pred)),"R2":float(r2_score(y.iloc[split:],pred))})
        results=sorted(results,key=lambda r:r["MAE"])
        return {"problem":"Regression","target":target,"metric":"MAE","results":pd.DataFrame(results),"best":results[0],"excluded_leakage":_leakage_features(df,target)}
    else:
        y=y.astype(str)
        if y.nunique()<2 or y.nunique()>12: raise ValueError("Classification target should have 2–12 useful classes.")
        le=LabelEncoder(); yy=le.fit_transform(y)
        models={
            "Logistic Regression": LogisticRegression(max_iter=1200),
            "Random Forest": RandomForestClassifier(n_estimators=180,min_samples_leaf=2,random_state=42,n_jobs=1,class_weight="balanced"),
            "Extra Trees": ExtraTreesClassifier(n_estimators=180,min_samples_leaf=2,random_state=42,n_jobs=1,class_weight="balanced"),
        }
        split=int(len(x)*0.8); split=max(20,min(split,len(x)-5))
        results=[]
        for name,model in models.items():
            pipe=Pipeline([("prep",prep),("model",model)])
            pipe.fit(x.iloc[:split],yy[:split]); pred=pipe.predict(x.iloc[split:])
            results.append({"model":name,"Accuracy":float(accuracy_score(yy[split:],pred)),"F1":float(f1_score(yy[split:],pred,average="weighted"))})
        results=sorted(results,key=lambda r:r["F1"],reverse=True)
        return {"problem":"Classification","target":target,"metric":"F1","classes":list(le.classes_),"results":pd.DataFrame(results),"best":results[0],"excluded_leakage":_leakage_features(df,target)}


def anomaly_detection(df: pd.DataFrame) -> dict | None:
    numeric=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True)>2]
    if len(df)<30 or len(numeric)<2: return None
    x=df[numeric].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(0)
    model=IsolationForest(n_estimators=300,contamination=0.03,random_state=42)
    flags=model.fit_predict(x); scores=model.decision_function(x)
    out=df.copy(); out["unusual"]=flags==-1; out["anomaly_score"]=scores
    return {"data":out,"features":numeric,"count":int((flags==-1).sum())}


def cluster_data(df: pd.DataFrame) -> dict | None:
    numeric=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True)>2]
    if len(df)<30 or len(numeric)<2: return None
    x=df[numeric].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(0)
    k=min(4,max(2,int(np.sqrt(len(df)/50))))
    model=KMeans(n_clusters=k,random_state=42,n_init=10)
    labels=model.fit_predict(StandardScaler().fit_transform(x))
    return {"data":df.assign(cluster=labels),"features":numeric,"k":k}


def forecast_numeric(df: pd.DataFrame, date_col: str, value_col: str, horizon: int=7) -> dict | None:
    d=pd.DataFrame({"date":pd.to_datetime(df[date_col],errors="coerce"),"value":pd.to_numeric(df[value_col],errors="coerce")}).dropna()
    if len(d)<30: return None
    d["day"] = d["date"].dt.floor("D")
    d=d.groupby("day",as_index=False).value.sum().rename(columns={"day":"date"}).sort_values("date")
    if len(d)<21: return None
    d["lag1"]=d.value.shift(1); d["lag7"]=d.value.shift(7); d["roll7"]=d.value.shift(1).rolling(7).mean(); d["dow"]=d.date.dt.dayofweek
    feat=d.dropna().copy();
    if len(feat)<14:return None
    split=max(10,len(feat)-max(7,min(14,len(feat)//5)))
    model=RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=1)
    cols=["lag1","lag7","roll7","dow"]
    model.fit(feat.iloc[:split][cols],feat.iloc[:split].value)
    pred=np.maximum(0,model.predict(feat.iloc[split:][cols])); mae=float(mean_absolute_error(feat.iloc[split:].value,pred))
    hist=d[["date","value"]].copy(); fut=[]
    for _ in range(horizon):
        nxt=hist.date.iloc[-1]+pd.Timedelta(days=1)
        row=pd.DataFrame([{"lag1":hist.value.iloc[-1],"lag7":hist.value.iloc[-7],"roll7":hist.value.tail(7).mean(),"dow":nxt.dayofweek}])
        val=float(max(0,model.predict(row[cols])[0])); hist=pd.concat([hist,pd.DataFrame([{"date":nxt,"value":val}])],ignore_index=True); fut.append({"date":nxt.date(),"forecast":round(val,2)})
    return {"mae":mae,"validation":pd.DataFrame({"date":feat.iloc[split:].date.dt.date,"actual":feat.iloc[split:].value.round(2),"predicted":pred.round(2)}),"forecast":pd.DataFrame(fut),"method":"Random Forest lag model with chronological validation"}
