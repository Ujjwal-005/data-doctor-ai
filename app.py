from __future__ import annotations
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from schema_engine import canonicalize, detect_domain, profile, LABELS
from insights_engine import summarize_business
from ml_engine import auto_supervised, anomaly_detection, cluster_data, candidate_targets, forecast_numeric

st.set_page_config(page_title="Data Doctor AI", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{
  --bg:#0a0f16;
  --sidebar:#0d141d;
  --panel:#111923;
  --panel-soft:#131e2a;
  --border:#263547;
  --border-soft:#1c2a39;
  --text:#eef3f6;
  --muted:#93a3b1;
  --accent:#32c7ad;
  --accent-soft:#173f3a;
  --blue:#69aaf0;
  --warning:#e3ad58;
}
.stApp{background:var(--bg);color:var(--text)}
.block-container{max-width:1480px;padding:1.25rem 2.4rem 4rem}
[data-testid="stSidebar"]{background:var(--sidebar);border-right:1px solid #1f2c3b}
[data-testid="stSidebar"] > div:first-child{padding:1.2rem 1rem 1.6rem}
[data-testid="stSidebar"] *{color:#d8e2e8}
[data-testid="stSidebar"] .stRadio label{padding:.38rem .15rem;font-size:14px}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:.1rem}

.brand{display:flex;align-items:center;gap:10px;margin:2px 0 6px}
.brand-mark{width:34px;height:34px;border:1px solid #31505e;border-radius:10px;display:flex;align-items:center;justify-content:center;background:#101c24;font-size:18px}
.brand-name{font-size:19px;font-weight:760;letter-spacing:-.25px;color:#f3f7f8}
.brand-sub{font-size:12px;color:#7f919e;margin:0 0 22px 44px}
.sidebar-section{font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:#688091;font-weight:700;margin:18px 0 7px}
.sidebar-note{font-size:12px;color:#78909f;line-height:1.5;margin-top:16px}

.topline{display:flex;align-items:center;justify-content:space-between;margin:2px 0 14px;padding-bottom:9px;border-bottom:1px solid var(--border-soft)}
.topline-left{font-size:12px;color:#7f919e}
.topline-right{font-size:12px;color:#9eb0bc}

.hero{padding:30px 32px;border-radius:22px;background:linear-gradient(115deg,#101b27 0%,#112c2a 100%);border:1px solid #29475a;box-shadow:0 16px 35px rgba(0,0,0,.18);margin:10px 0 18px}
.hero-eyebrow{text-transform:uppercase;letter-spacing:.16em;font-size:11px;color:#71a8bb;font-weight:800;margin-bottom:9px}
.hero h1{font-size:40px;line-height:1.06;margin:0;color:#f7fafb;letter-spacing:-1px}
.hero p{font-size:15px;color:#b7c6d0;margin:.65rem 0 0;max-width:850px;line-height:1.65}
.hero-actions{margin-top:17px}

.pill-row{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 18px}
.pill{display:inline-flex;align-items:center;padding:7px 11px;border:1px solid #2b4557;border-radius:999px;background:#101a24;color:#b9d4e3;font-size:12px}
.pill.accent{background:#102620;border-color:#285148;color:#9ae0d3}

.pagebar{display:flex;justify-content:space-between;align-items:flex-end;gap:18px;margin:7px 0 16px}
.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:10px;color:#6e8798;font-weight:800}
.page-title{font-size:28px;font-weight:760;letter-spacing:-.45px;color:#f2f6f8;margin-top:4px}
.page-sub{font-size:13px;color:var(--muted);margin-top:4px;line-height:1.45}
.status{padding:7px 11px;border:1px solid #2d4752;border-radius:999px;background:#0f191f;color:#a9c3c0;font-size:12px;white-space:nowrap}

.section{font-size:20px;font-weight:760;margin:26px 0 11px;color:#f1f5f7;letter-spacing:-.25px}
.muted{color:var(--muted)}
.card,.panel{padding:18px 20px;border-radius:16px;background:var(--panel);border:1px solid var(--border);box-shadow:0 8px 20px rgba(0,0,0,.10)}
.good{border-left:4px solid var(--accent);background:#0f2522;padding:12px 14px;border-radius:10px;margin:7px 0;color:#d8efea}
.warn{border-left:4px solid var(--warning);background:#241d12;padding:12px 14px;border-radius:10px;margin:7px 0;color:#f0e3cf}
.action{border-left:4px solid var(--blue);background:#111f30;padding:12px 14px;border-radius:10px;margin:7px 0;color:#dceafb}
.kicker{font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:#708999;font-weight:800}

[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--border);padding:15px 16px;border-radius:15px;box-shadow:0 7px 18px rgba(0,0,0,.09)}
[data-testid="stMetricLabel"]{color:#8fa2b0!important}
[data-testid="stMetricValue"]{color:#f2f7fa!important;font-weight:730;font-size:28px}

.stButton button{border-radius:10px;border:1px solid #334858;background:#121c26;color:#e8f0f5;font-weight:650;min-height:42px}
.stButton button:hover{border-color:#497084;color:#fff}
button[kind="primary"]{background:#2a9f8a!important;border-color:#2a9f8a!important;color:#08110f!important}
button[kind="primary"]:hover{background:#31b39b!important;border-color:#31b39b!important}

[data-baseweb="select"] > div{background:#121c26!important;border-color:#2a3b4d!important;color:#eef4f8!important;border-radius:10px!important;min-height:42px}
[data-testid="stFileUploader"]{background:#101923;border:1px dashed #35516a;border-radius:15px;padding:8px}
[data-testid="stFileUploader"] section{background:transparent!important}

.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid #243344}
.stTabs [data-baseweb="tab"]{color:#8599a8;padding:10px 13px;font-size:13px}
.stTabs [aria-selected="true"]{color:#ecf4f6!important;border-bottom:2px solid var(--accent)!important;background:#101a24}

[data-testid="stDataFrame"]{border:1px solid #263547;border-radius:12px;overflow:hidden}
hr{border-color:#213042}

.home-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0}
.home-step{padding:18px 18px;border:1px solid var(--border);background:#101923;border-radius:15px}
.home-step .num{font-size:11px;color:#668193;text-transform:uppercase;letter-spacing:.12em}
.home-step h3{margin:.35rem 0 .25rem;font-size:18px;color:#eef4f7}
.home-step p{margin:0;color:#8fa0ad;font-size:13px;line-height:1.55}
.domain-row{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}
.domain{padding:6px 9px;border:1px solid #243a4a;border-radius:8px;color:#9db1bf;background:#101923;font-size:11px}

@media(max-width:900px){
  .block-container{padding:1rem 1rem 3rem}
  .hero h1{font-size:32px}
  .home-grid{grid-template-columns:1fr}
  .pagebar{align-items:flex-start;flex-direction:column}
}
</style>
""",unsafe_allow_html=True)


def chart(fig,height=360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=18,r=18,t=48,b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif",color="#d9e3ea",size=12),
        title=dict(font=dict(size=16,color="#eef4f7"),x=0.01,xanchor="left"),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#b7c6d1")),
        hoverlabel=dict(bgcolor="#101923",bordercolor="#35506a",font_color="#edf4f7"),
    )
    fig.update_xaxes(showgrid=False,zeroline=False,title_font=dict(color="#92a6b5"),tickfont=dict(color="#8799a8"))
    fig.update_yaxes(gridcolor="#20303f",gridwidth=1,zeroline=False,title_font=dict(color="#92a6b5"),tickfont=dict(color="#8799a8"))
    st.plotly_chart(fig,width="stretch",config={"displaylogo":False,"responsive":True})

def read_upload(upload):
    if upload.name.lower().endswith(".csv"): return pd.read_csv(upload)
    xl=pd.ExcelFile(upload); return pd.read_excel(upload,sheet_name=xl.sheet_names[0])

def load_state(upload):
    raw=read_upload(upload); clean,mapping,warnings=canonicalize(raw); domain,conf,scores=detect_domain(raw,mapping)
    st.session_state.data={"raw":raw,"clean":clean,"mapping":mapping,"warnings":warnings,"domain":domain,"conf":conf,"scores":scores,"file":upload.name,"profile":profile(raw),"token":(upload.name,len(upload.getvalue()))}



from relationships import infer_relationships, build_multi_sql

def owner_page():
    d=st.session_state.data["datasets"][st.session_state.data["primary"]]
    st.markdown(f'<div class="pagebar"><div><div class="eyebrow">Business owner dashboard</div><div class="page-title">Your business at a glance</div><div class="page-sub">{d["file"]} · {d["domain"]} · focused on money, customers and what needs attention</div></div><div class="status">● Data loaded</div></div>',unsafe_allow_html=True)
    d=st.session_state.data["datasets"][st.session_state.data["primary"]]; df=d["clean"]
    if "revenue" in df: rev=float(pd.to_numeric(df.revenue,errors="coerce").sum())
    else: rev=None
    if "profit" in df: profit=float(pd.to_numeric(df.profit,errors="coerce").sum())
    else: profit=None
    c=st.columns(6); c[0].metric("Records",f"{len(df):,}"); c[1].metric("Fields",f"{len(d['raw'].columns):,}"); c[2].metric("Revenue",f"{rev:,.0f}" if rev is not None else "Not available"); c[3].metric("Profit",f"{profit:,.0f}" if profit is not None else "Not available"); c[4].metric("Items / Groups",f"{df['item'].nunique():,}" if 'item' in df else "—"); c[5].metric("Domain",d["domain"])
    st.caption(f"Business context: {d['domain']} · confidence {int(d['conf']*100)}% · dashboard adapts to the fields found in this file")
    filters=[]
    if "category" in df and df.category.nunique()<=100: filters.append(("Category","category"))
    if "location" in df and df.location.nunique()<=100: filters.append(("Location","location"))
    filtered=df.copy(); fc=st.columns(max(1,min(3,len(filters)))) if filters else []
    for i,(label,col) in enumerate(filters[:3]):
        vals=["All"]+sorted(df[col].dropna().astype(str).unique().tolist())
        sel=fc[i].selectbox(label,vals,key=f"owner_filter_{col}")
        if sel!="All": filtered=filtered[filtered[col].astype(str)==sel]
    if "revenue" in filtered and "datetime" in filtered:
        x=filtered.copy(); x["_dt"]=pd.to_datetime(x.datetime,errors="coerce"); x["_rev"]=pd.to_numeric(x.revenue,errors="coerce"); g=x.dropna(subset=["_dt"]).groupby(x.dropna(subset=["_dt"])["_dt"].dt.date)._rev.sum().reset_index(name="Revenue")
        if not g.empty: chart(px.line(g,x="_dt",y="Revenue",markers=True,title="Revenue trend"),380)
    col1,col2=st.columns(2)
    if "item" in filtered and "revenue" in filtered:
        q=filtered.groupby("item").revenue.sum().nlargest(12).sort_values(); chart(px.bar(q,x=q.values,y=q.index,orientation="h",title="Top revenue drivers"),400)
    elif "category" in filtered:
        q=filtered.category.value_counts().head(12).sort_values(); chart(px.bar(q,x=q.values,y=q.index,orientation="h",title="Largest groups"),400)
    if "item" in filtered and "profit" in filtered:
        q=filtered.groupby("item").agg(Revenue=("revenue","sum") if "revenue" in filtered else ("profit","count"),Profit=("profit","sum")).reset_index().sort_values("Revenue" if "revenue" in filtered else "Profit",ascending=False).head(15)
        chart(px.bar(q,x="item",y="Profit",title="Profit by item"),400)
    bi=summarize_business(filtered,d["mapping"])
    st.markdown('<div class="section">💡 What the owner should know</div>',unsafe_allow_html=True)
    for s in bi["positives"][:6]: st.markdown(f'<div class="good">✓ {s}</div>',unsafe_allow_html=True)
    for s in bi["attention"][:5]: st.markdown(f'<div class="warn">⚠ {s}</div>',unsafe_allow_html=True)
    st.markdown('<div class="section">🎯 Recommended next actions</div>',unsafe_allow_html=True)
    for s in bi["actions"][:6]: st.markdown(f'<div class="action">→ {s}</div>',unsafe_allow_html=True)

    st.markdown('<div class="section">🤖 Forecast & risk signals</div>',unsafe_allow_html=True)
    ml1,ml2=st.columns(2)
    with ml1:
        st.markdown('<div class="card"><div class="kicker">FORECAST</div><h3>What may happen next?</h3><p class="muted">Uses the history in your file to estimate the next few periods. No forecast is shown when the data is not strong enough.</p></div>',unsafe_allow_html=True)
        if "datetime" in filtered.columns:
            numeric=[c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
            values=[c for c in ["quantity","revenue","profit"] if c in numeric] or numeric[:3]
            if values:
                v=st.selectbox("What should we forecast?",values,key="owner_forecast_value")
                if st.button("Run forecast",key="owner_forecast_run",type="primary"):
                    try:
                        r=forecast_numeric(filtered,"datetime",v,7); st.metric("Validation MAE",f"{r['mae']:.2f}"); chart(px.line(r["validation"],x="date",y=["actual","predicted"],title="Model validation"),320); st.dataframe(r["forecast"],width="stretch",hide_index=True)
                    except Exception as e: st.warning(str(e))
            else: st.info("No numeric measure available for forecasting.")
        else: st.info("Add a date/time column to unlock forecasting.")
    with ml2:
        st.markdown('<div class="card"><div class="kicker">RISK CHECK</div><h3>Anything unusual?</h3><p class="muted">Highlights records that look different from the normal pattern. An unusual record is not automatically fraudulent.</p></div>',unsafe_allow_html=True)
        an=anomaly_detection(filtered)
        if an:
            st.metric("Unusual records",f"{an['count']:,}"); st.caption("Signals used: "+", ".join(an["features"]));
            if an["count"]: st.dataframe(an["data"].loc[an["data"].unusual].head(20),width="stretch",hide_index=True)
        else: st.info("Need at least 30 rows and two useful numeric fields.")
    if "stock" in filtered:
        st.markdown('<div class="section">📦 Stock & availability</div>',unsafe_allow_html=True)
        stock=pd.to_numeric(filtered.stock,errors="coerce"); show_cols=[c for c in ["item","stock","quantity","reorder"] if c in filtered.columns]
        st.dataframe(filtered.loc[stock<=0,show_cols].head(30),width="stretch",hide_index=True) if (stock<=0).any() else st.success("No zero/negative stock rows were detected.")
    with st.expander("🔎 What Data Doctor understood"):
        st.write(f"Detected context: {d['domain']} ({int(d['conf']*100)}% confidence)")
        st.dataframe(pd.DataFrame([{"Business field":LABELS.get(k,k),"Source column":v} for k,v in d["mapping"].items()]),width="stretch",hide_index=True)
        for w in d["warnings"]: st.warning(w)


def analyst_page():
    state=st.session_state.data; d=state["datasets"][state["primary"]]; raw,df=d["raw"],d["clean"]
    st.markdown(f'<div class="pagebar"><div><div class="eyebrow">Analyst workspace</div><div class="page-title">📊 Data / Power BI Analyst</div><div class="page-sub">{d["file"]} · profiling, cleaning, visual analysis, AutoML, relationships, SQL and business story</div></div><div class="status">● Dataset ready</div></div>',unsafe_allow_html=True)
    tabs=st.tabs(["Overview","Data Check","Visual Analysis","ML Lab","SQL & Data Model","Business Story","Export"])
    with tabs[0]:
        c=st.columns(6); vals=[("Rows",len(raw)),("Columns",len(raw.columns)),("Missing",f"{d['profile']['missing_pct']:.1f}%"),("Duplicates",d['profile']['duplicates']),("Domain",d['domain']),("Confidence",f"{int(d['conf']*100)}%")] 
        for x,(a,b) in zip(c,vals): x.metric(a,b)
        st.markdown('<div class="section">Column inventory</div>',unsafe_allow_html=True)
        inv=[]
        for col in raw.columns:
            s=raw[col]; inv.append({"Column":str(col),"Type":str(s.dtype),"Rows":len(s),"Non-null":int(s.notna().sum()),"Missing %":round(100*s.isna().mean(),1),"Unique":int(s.nunique(dropna=True))})
        st.dataframe(pd.DataFrame(inv),width="stretch",hide_index=True)
    with tabs[1]:
        st.markdown('<div class="section">🧹 Cleaning audit</div>',unsafe_allow_html=True)
        st.write(f"Detected {d['profile']['duplicates']:,} duplicate rows and {d['profile']['missing_cells']:,} missing cells ({d['profile']['missing_pct']:.1f}% of all cells).")
        for w in d["warnings"]: st.markdown(f'<div class="good">✓ {w}</div>',unsafe_allow_html=True)
        st.markdown('<div class="section">Automatic field mapping</div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Business role":LABELS.get(k,k),"Source column":v} for k,v in d["mapping"].items()]),width="stretch",hide_index=True)
        st.markdown('<div class="section">Data type audit</div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Column":raw.columns.astype(str),"Type":[str(raw[c].dtype) for c in raw.columns],"Missing":[int(raw[c].isna().sum()) for c in raw.columns],"Unique":[int(raw[c].nunique(dropna=True)) for c in raw.columns]}),width="stretch",hide_index=True)
    with tabs[2]:
        nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]; dates=[c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dates and "revenue" in df: 
            x=df.dropna(subset=[dates[0]]).copy(); x["day"]=x[dates[0]].dt.date; g=x.groupby("day").revenue.sum().reset_index(); chart(px.line(g,x="day",y="revenue",markers=True,title="Revenue over time",labels={"day":"Date","revenue":"Revenue"}),380)
        elif dates:
            x=df.dropna(subset=[dates[0]]).copy(); x["day"]=x[dates[0]].dt.date; g=x.groupby("day").size().reset_index(name="records"); chart(px.line(g,x="day",y="records",markers=True,title="Records over time",labels={"day":"Date","records":"Records"}),380)
        if "item" in df and "revenue" in df: chart(px.bar(df.groupby("item").revenue.sum().nlargest(15).sort_values(),orientation="h",title="Top items by revenue",labels={"value":"Revenue","item":"Item"}),400)
        if len(nums)>=2: chart(px.imshow(df[nums].corr(numeric_only=True),text_auto=".2f",aspect="auto",title="Numeric relationship map",color_continuous_scale="Viridis"),430)
    with tabs[3]:
        st.markdown('<div class="section">🤖 AutoML Lab — many candidate models</div>',unsafe_allow_html=True)
        st.caption("Data Doctor tests multiple supervised models against a time-ordered holdout when possible. It does not force a model when the dataset cannot support the problem.")
        targets=candidate_targets(df)
        if targets:
            target=st.selectbox("What should the model predict?",targets,key="analyst_target")
            if st.button("Train & compare models",type="primary",key="automl_run"):
                try:
                    r=auto_supervised(df,target); st.success(f"{r['problem']} problem · target: {target} · best validation model: {r['best']['model']}"); st.dataframe(r["results"],width="stretch",hide_index=True)
                    if r.get("excluded_leakage"):
                        st.info("Leakage check: these fields were excluded because they can directly give away the target: " + ", ".join(map(str,r["excluded_leakage"])))
                except Exception as e: st.warning(str(e))
        else: st.info("No suitable supervised target was automatically identified. Use anomaly detection or clustering instead.")
        st.markdown('<div class="section">Unsupervised ML</div>',unsafe_allow_html=True)
        an=anomaly_detection(df)
        if an: st.metric("Unusual records",f"{an['count']:,}"); st.dataframe(an["data"].loc[an["data"].unusual].head(30),width="stretch",hide_index=True)
        cl=cluster_data(df)
        if cl: st.write(f"Clustering available with {cl['k']} groups using: {', '.join(cl['features'])}"); st.dataframe(cl["data"].head(30),width="stretch",hide_index=True)
        if "datetime" in df:
            nums=[c for c in ["quantity","revenue","profit"] if c in df]
            if nums:
                v=st.selectbox("Forecast a time-based measure",nums,key="analyst_forecast_value")
                if st.button("Run time forecast",key="analyst_forecast_run"):
                    try:
                        r=forecast_numeric(df,"datetime",v); st.metric("Validation MAE",f"{r['mae']:.2f}"); chart(px.line(r["validation"],x="date",y=["actual","predicted"],title="Chronological validation"),350); st.dataframe(r["forecast"],width="stretch",hide_index=True)
                    except Exception as e: st.warning(str(e))
    with tabs[4]:
        all_ds=state["datasets"]
        st.markdown('<div class="section">🗄️ Suggested relational model</div>',unsafe_allow_html=True)
        if len(all_ds)>1:
            st.markdown("### Uploaded datasets")
            st.dataframe(pd.DataFrame([{"Dataset":n,"Rows":len(v["raw"]),"Columns":len(v["raw"].columns),"Domain":v["domain"]} for n,v in all_ds.items()]),width="stretch",hide_index=True)
            rels=infer_relationships(all_ds)
            st.markdown("### 🔗 Inferred relationships")
            if rels:
                st.dataframe(pd.DataFrame(rels),width="stretch",hide_index=True)
                sql_all=build_multi_sql(all_ds,rels)
                st.code(sql_all,language="sql")
                st.download_button("Download multi-dataset SQL model",sql_all.encode(),file_name="data_doctor_multi_dataset.sql")
            else:
                st.info("No strong cross-dataset relationship was found from identifier names and shared values. Upload files that share an ID/code column to infer joins.")
        st.caption("The model is inferred from your columns. Key candidates are suggestions, not declarations.")

        # Strong key candidates: identifier-like names + near-unique values; exclude business measures.
        idcols=[]
        id_tokens={"id","code","number","no","uuid","key","invoice","order","transaction","receipt","record","patient","employee","customer","account","case"}
        measure_words={"revenue","sales","total","tax","profit","income","cost","cogs","price","quantity","qty","rating","amount","margin","percentage"}
        for c in raw.columns:
            s=raw[c]
            rate=s.nunique(dropna=True)/max(1,len(raw))
            cname=str(c).strip().lower()
            tokens=set(re.split(r"[^a-z0-9]+",cname))
            looks_like_id=bool(tokens & id_tokens) or cname.endswith("_id") or cname.endswith(" id")
            looks_like_measure=bool(tokens & measure_words)
            if rate>=.98 and looks_like_id and not looks_like_measure:
                idcols.append((str(c),round(rate*100,1),"Strong key candidate" if rate==1 else "Possible key"))

        if idcols:
            st.dataframe(pd.DataFrame(idcols,columns=["Column","Unique %","Role"]),width="stretch",hide_index=True)
            primary=idcols[0][0]
        else:
            st.dataframe(pd.DataFrame([{"Column":"No strong identifier candidate found","Unique %":"—","Role":"Review manually"}]),width="stretch",hide_index=True)
            primary=None

        # Keep an editable, analyst-friendly relational design instead of treating
        # every high-cardinality measure as a key.
        safe_cols=[]
        for c in raw.columns:
            safe=str(c).replace('"','""')
            typ="TEXT" if not pd.api.types.is_numeric_dtype(raw[c]) else "REAL"
            safe_cols.append((safe,typ))

        table="business_records"
        lines=[f'CREATE TABLE {table} (']
        for safe,typ in safe_cols:
            lines.append(f'  "{safe}" {typ},')
        if primary:
            safe_primary=str(primary).replace('"','""')
            lines.append(f'  PRIMARY KEY ("{safe_primary}")')
        else:
            lines[-1]=lines[-1].rstrip(',')
        lines.append(');')
        sql='\n'.join(lines)
        st.code(sql,language="sql")

        st.markdown('### Suggested business model')
        st.caption("A practical star-style model is suggested only when the required business fields are present.")
        rel=[]
        if primary:
            rel.append(f"business_records.{primary} is the suggested row identifier.")
        if "item" in d["mapping"]:
            rel.append(f"dim_item can use {d['mapping']['item']} as the business item label.")
        if "customer" in d["mapping"]:
            rel.append(f"dim_customer can use {d['mapping']['customer']} as the customer field.")
        if "location" in d["mapping"]:
            rel.append(f"dim_location can use {d['mapping']['location']} as the location field.")
        if rel:
            for x in rel:
                st.markdown(f"- {x}")
        else:
            st.info("No separate dimension could be inferred safely from the available fields.")

        st.download_button("Download SQL",sql.encode(),file_name="data_doctor_model.sql")
    with tabs[5]:
        bi=summarize_business(df,d["mapping"])
        st.markdown("### What happened")
        if bi["positives"]:
            for x in bi["positives"]:
                st.markdown(f"- {x}")
        else:
            st.info("No strong positive pattern was identified from the available fields.")

        st.markdown("### What needs attention")
        if bi["attention"]:
            for x in bi["attention"]:
                st.markdown(f"- {x}")
        else:
            st.success("No major issue was identified from the available business fields.")

        st.markdown("### What to do next")
        if bi["actions"]:
            for x in bi["actions"]:
                st.markdown(f"- {x}")
    with tabs[6]:
        st.download_button("Download cleaned data",df.to_csv(index=False).encode(),file_name="data_doctor_cleaned.csv")
        st.download_button("Download field mapping",pd.DataFrame([{"Business role":LABELS.get(k,k),"Source column":v} for k,v in d["mapping"].items()]).to_csv(index=False).encode(),file_name="field_mapping.csv")
        st.dataframe(df.head(200),width="stretch",hide_index=True)


def main():
    # ------------------------------------------------------------
    # Sidebar: restrained, human-style product navigation
    # ------------------------------------------------------------
    st.sidebar.markdown(
        '<div class="brand"><div class="brand-mark">⌁</div><div class="brand-name">Data Doctor AI</div></div>'
        '<div class="brand-sub">Business intelligence, explained clearly.</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown('<div class="sidebar-section">Workspace</div>', unsafe_allow_html=True)
    mode=st.sidebar.radio("",["Business Owner","Data / Power BI Analyst"],label_visibility="collapsed")
    st.sidebar.markdown('<div class="sidebar-section">Data source</div>', unsafe_allow_html=True)
    st.sidebar.caption("Upload one or more CSV / Excel files. Your data stays the source of truth.")
    st.sidebar.markdown('<div class="sidebar-note">Designed for structured business data. The dashboard adapts to the fields available in each file.</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="topline"><div class="topline-left">Data Doctor AI</div>'
        f'<div class="topline-right">{mode} · Structured data workspace</div></div>',
        unsafe_allow_html=True
    )

    uploads=st.file_uploader(
        "Upload business data",
        type=["csv","xlsx","xls"],
        accept_multiple_files=True,
        key="universal_upload",
        label_visibility="visible"
    )
    c1,c2=st.columns([1,1])
    with c1:
        if uploads and st.button("Load / Replace dataset(s)",type="primary",width="stretch"):
            datasets={}
            for up in uploads:
                raw=read_upload(up); clean,mapping,warnings=canonicalize(raw); domain,conf,scores=detect_domain(raw,mapping)
                datasets[up.name]={"raw":raw,"clean":clean,"mapping":mapping,"warnings":warnings,"domain":domain,"conf":conf,"scores":scores,"file":up.name,"profile":profile(raw)}
            primary=next(iter(datasets))
            st.session_state.data={"datasets":datasets,"primary":primary,"files":[u.name for u in uploads],"token":[(u.name,len(u.getvalue())) for u in uploads]}
            st.rerun()
    with c2:
        if st.button("Clear dashboard",width="stretch"):
            st.session_state.pop("data",None); st.rerun()

    if "data" not in st.session_state:
        st.markdown(
            '<div class="hero">'
            '<div class="hero-eyebrow">DATA → DECISIONS</div>'
            '<h1>Data Doctor AI</h1>'
            '<p>Drop in a business dataset and get a tailored dashboard, useful patterns, forecasts and practical next steps — built from the information your file actually contains.</p>'
            '<div class="hero-actions"><span class="pill accent">Power BI-style visuals</span> <span class="pill">Automatic field discovery</span> <span class="pill">Business recommendations</span> <span class="pill">Machine learning where supported</span></div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="section">A simple workflow</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="home-grid">'
            '<div class="home-step"><div class="num">01 · Upload</div><h3>Bring your data</h3><p>CSV or Excel files. One file or several related tables.</p></div>'
            '<div class="home-step"><div class="num">02 · Understand</div><h3>See what matters</h3><p>Fields, data quality, business measures and useful patterns are identified automatically.</p></div>'
            '<div class="home-step"><div class="num">03 · Decide</div><h3>Act with confidence</h3><p>Use the dashboard, predictions and recommendations to decide what to review next.</p></div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="section">Built for different kinds of business data</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="domain-row">'
            '<span class="domain">Grocery & Retail</span><span class="domain">Restaurants</span><span class="domain">Healthcare</span><span class="domain">Finance</span><span class="domain">HR</span><span class="domain">E-commerce</span><span class="domain">Logistics</span><span class="domain">Other tabular data</span>'
            '</div>',
            unsafe_allow_html=True
        )
        return

    if len(st.session_state.data.get("datasets",{}))>1:
        st.info(f"{len(st.session_state.data['datasets'])} datasets loaded together. Analyst mode can infer relationships and build a combined SQL model.")
    owner_page() if mode=="Business Owner" else analyst_page()

main()
