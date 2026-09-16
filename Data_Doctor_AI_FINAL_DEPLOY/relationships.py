from __future__ import annotations
import re
import pandas as pd

def _value_set(df: pd.DataFrame, col: str, limit: int = 200_000) -> set:
    if col not in df.columns:
        return set()
    s = df[col].dropna()
    if len(s) > limit:
        s = s.head(limit)
    return set(s.astype(str).str.strip())


def infer_relationships(datasets: dict[str, dict]) -> list[dict]:
    """Infer likely PK/FK relationships across uploaded datasets.
    Conservative rules: identifier-like columns, non-empty overlap,
    child column need not be unique, parent column should be unique.
    """
    id_words = {"id","code","number","no","uuid","key","invoice","order","transaction","receipt","patient","employee","customer","account","case","product","sku","department","branch","store","city"}
    measure_words = {"revenue","sales","total","tax","profit","income","cost","cogs","price","quantity","qty","rating","amount","margin","percentage","age"}
    candidates=[]
    for ds_name, d in datasets.items():
        df=d["raw"]
        for col in df.columns:
            name=str(col)
            norm=re.sub(r"[^a-z0-9]+","_",name.lower()).strip("_")
            tokens=set(norm.split("_"))
            nun=df[col].nunique(dropna=True)
            unique_ratio=nun/max(1,len(df))
            id_like=bool(tokens & id_words) or norm.endswith("_id") or norm in {"id","code"}
            not_measure=not bool(tokens & measure_words)
            if id_like and not_measure and nun>0:
                candidates.append({"dataset":ds_name,"column":name,"norm":norm,"unique_ratio":unique_ratio,"values":_value_set(df,col)})
    rel=[]
    for child in candidates:
        for parent in candidates:
            if child["dataset"]==parent["dataset"] and child["column"]==parent["column"]:
                continue
            # Naming signal: identical normalized name or shared stem.
            name_match = child["norm"]==parent["norm"] or child["norm"].replace("_id","")==parent["norm"].replace("_id","")
            overlap = len(child["values"] & parent["values"])
            parent_unique = parent["unique_ratio"] >= 0.95
            if not name_match or overlap < 2 or not parent_unique:
                continue
            overlap_rate = overlap/max(1,len(child["values"]))
            score=min(100,int((0.65*min(1,overlap_rate)+0.35*(1 if parent_unique else 0))*100))
            rel.append({"From Dataset":child["dataset"],"From Column":child["column"],"To Dataset":parent["dataset"],"To Column":parent["column"],"Shared Values":overlap,"Match %":round(overlap_rate*100,1),"Confidence":score})
    # Deduplicate opposite/symmetric entries and retain strongest.
    out=[]; seen=set()
    for r in sorted(rel,key=lambda x:(-x["Confidence"],-x["Shared Values"])):
        key=(r["From Dataset"],r["From Column"],r["To Dataset"],r["To Column"])
        reverse=(r["To Dataset"],r["To Column"],r["From Dataset"],r["From Column"])
        if key in seen or reverse in seen: continue
        seen.add(key); out.append(r)
    return out[:20]


def build_multi_sql(datasets: dict[str, dict], relationships: list[dict]) -> str:
    lines=["-- Data Doctor AI: inferred multi-dataset relational model", ""]
    for ds_name,d in datasets.items():
        table="".join(ch if ch.isalnum() else "_" for ch in ds_name.lower()).strip("_") or "dataset"
        table=re.sub(r"_+","_",table)
        if table[0].isdigit(): table="t_"+table
        raw=d["raw"]
        # pick strong PK from id-like + unique
        pk=None
        for c in raw.columns:
            norm=re.sub(r"[^a-z0-9]+","_",str(c).lower()).strip("_")
            if (norm.endswith("_id") or norm in {"id","code"}) and raw[c].nunique(dropna=True)==len(raw):
                pk=str(c); break
        lines.append(f'CREATE TABLE "{table}" (')
        for c in raw.columns:
            typ="REAL" if pd.api.types.is_numeric_dtype(raw[c]) else "TEXT"
            lines.append(f'  "{str(c).replace(chr(34), chr(34)*2)}" {typ},')
        if pk:
            lines.append(f'  PRIMARY KEY ("{pk.replace(chr(34), chr(34)*2)}")')
        else:
            lines[-1]=lines[-1].rstrip(',')
        lines.append(");\n")
    for r in relationships:
        ft="".join(ch if ch.isalnum() else "_" for ch in r["From Dataset"].lower()).strip("_") or "dataset"
        tt="".join(ch if ch.isalnum() else "_" for ch in r["To Dataset"].lower()).strip("_") or "dataset"
        fc=r["From Column"].replace('"','""'); tc=r["To Column"].replace('"','""')
        lines.append(f'-- Inferred relationship ({r["Confidence"]}% confidence)')
        lines.append(f'ALTER TABLE "{ft}" ADD FOREIGN KEY ("{fc}") REFERENCES "{tt}" ("{tc}");\n')
    return "\n".join(lines)

