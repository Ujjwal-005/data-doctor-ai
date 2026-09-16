from __future__ import annotations
import re
from dataclasses import dataclass
import pandas as pd

ROLE_ALIASES = {
    "id": ["id", "order_id", "order number", "order_no", "invoice", "invoice_id", "transaction_id", "bill_id", "patient_id", "customer_id", "employee_id", "record_id", "case_id"],
    "datetime": ["date", "time", "datetime", "timestamp", "order_date", "sale_date", "transaction_date", "invoice_date", "admission_date", "created_at", "date_time"],
    "item": ["product", "product_name", "item", "item_name", "sku", "sku_name", "dish", "menu_item", "service", "product_line", "description"],
    "category": ["category", "product_category", "department", "department_name", "ward", "speciality", "specialty", "segment", "class"],
    "location": ["store", "store_name", "branch", "outlet", "location", "city", "region", "country", "hospital", "facility"],
    "quantity": ["quantity", "qty", "units", "units_sold", "items", "count", "volume"],
    "price": ["unit_price", "price", "selling_price", "sale_price", "sell_price", "rate", "amount_per_unit"],
    "cost": ["cost", "cost_price", "purchase_cost", "purchase_price", "buy_price", "unit_cost", "cogs"],
    "revenue": ["revenue", "sales", "sales_amount", "net_sales", "gross_sales", "turnover", "total", "total_amount", "bill_amount", "amount"],
    "profit": ["profit", "gross_profit", "net_profit", "gross_income", "profit_amount", "margin_amount"],
    "stock": ["stock", "stock_qty", "inventory", "available_qty", "on_hand", "closing_stock", "current_stock"],
    "reorder": ["reorder_level", "reorder_point", "min_stock", "minimum_stock", "safety_stock"],
    "customer": ["customer", "customer_id", "member", "client"],
    "patient": ["patient", "patient_id"],
    "employee": ["employee", "employee_id", "staff"],
    "status": ["status", "payment_status", "order_status", "outcome", "result", "disposition"],
    "gender": ["gender", "sex"],
    "age": ["age", "patient_age", "employee_age"],
}

LABELS = {
    "id":"ID / Record Identifier", "datetime":"Date / Time", "item":"Item / Product / Service", "category":"Category / Group",
    "location":"Location / Branch", "quantity":"Quantity / Count", "price":"Price / Rate", "cost":"Cost", "revenue":"Revenue / Sales",
    "profit":"Profit / Income", "stock":"Current Stock", "reorder":"Reorder Level", "customer":"Customer / Client", "patient":"Patient",
    "employee":"Employee / Staff", "status":"Status / Outcome", "gender":"Gender", "age":"Age"
}

def normalize(name: object) -> str:
    s = str(name).strip().lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")

def _token_set(s: str) -> set[str]:
    return set(normalize(s).split("_")) - {""}

def _parse_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def _score(role: str, alias: str, col: str) -> float:
    if alias == col:
        return 1000
    aw, cw = _token_set(alias), _token_set(col)
    if not aw or not aw.issubset(cw):
        return -1
    score = 100 * len(aw)
    if col.startswith(alias): score += 35
    if col.endswith(alias): score += 15
    # Prevent generic aliases from stealing semantically stronger fields.
    if role == "price" and cw & {"sell","selling","sale","retail"}: score += 120
    if role == "cost" and cw & {"buy","purchase","cost","cogs"}: score += 120
    if role == "revenue" and cw & {"revenue","sales","total","amount"}: score += 90
    if role == "profit" and "profit" in cw: score += 120
    if role == "item" and cw & {"product","item","dish","sku","service"}: score += 100
    if role == "datetime" and cw & {"date","time","datetime","timestamp"}: score += 100
    return score

def infer_roles(raw: pd.DataFrame) -> tuple[dict[str,str], dict[str,float]]:
    cols = {normalize(c): c for c in raw.columns}
    mapping: dict[str,str] = {}
    confidence: dict[str,float] = {}
    used: set[str] = set()
    # Role priority makes high-value semantic roles claim columns first.
    priority = ["id","datetime","item","category","location","quantity","revenue","profit","cost","price","stock","reorder","customer","patient","employee","status","gender","age"]
    for role in priority:
        best = None
        best_score = -1
        for alias in ROLE_ALIASES[role]:
            na = normalize(alias)
            for nc, original in cols.items():
                if original in used: continue
                sc = _score(role, na, nc)
                if sc > best_score:
                    best, best_score = original, sc
        if best is not None and best_score >= 100:
            mapping[role] = str(best)
            confidence[role] = min(1.0, best_score / 1200)
            used.add(best)
    # Date + time: combine later if both are detected as separate columns.
    return mapping, confidence

def detect_domain(raw: pd.DataFrame, mapping: dict[str,str]) -> tuple[str,float,dict[str,float]]:
    text = " ".join(normalize(c) for c in raw.columns)
    signals = {
        "Retail / Grocery": ["product","item","sku","stock","inventory","sales","quantity","price"],
        "Restaurant / Food": ["dish","menu","restaurant","table","food","ingredient","recipe","waiter"],
        "Healthcare": ["patient","diagnosis","admission","discharge","doctor","ward","hospital","treatment"],
        "Finance": ["loan","credit","debit","account","balance","interest","payment","transaction","bank"],
        "HR / People": ["employee","salary","attrition","hire","joining","performance","job","staff"],
        "E-commerce": ["order","customer","shipping","delivery","cart","product","quantity","payment"],
        "Logistics": ["delivery","distance","vehicle","driver","route","shipment","warehouse"],
    }
    scores = {d: sum(k in text for k in ks) for d, ks in signals.items()}
    # Mapping evidence gives a softer domain signal.
    if "stock" in mapping: scores["Retail / Grocery"] += 2
    if "patient" in mapping: scores["Healthcare"] += 3
    if "employee" in mapping: scores["HR / People"] += 3
    if "profit" in mapping and "revenue" in mapping: scores["Retail / Grocery"] += 1
    if not scores or max(scores.values()) == 0:
        return "General Business", 0.0, scores
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = min(1.0, scores[best] / max(1, total) + 0.25)
    return best, round(confidence, 2), scores

def canonicalize(raw: pd.DataFrame) -> tuple[pd.DataFrame,dict[str,str],list[str]]:
    df = raw.copy()
    mapping, _ = infer_roles(df)
    warnings: list[str] = []
    clean = pd.DataFrame(index=df.index)
    used = set()
    for role, src in mapping.items():
        clean[role] = df[src]
        used.add(src)
    # Merge separate date/time fields where possible.
    raw_norm = {normalize(c): c for c in df.columns}
    if "datetime" in mapping:
        clean["datetime"] = _parse_datetime(clean["datetime"])
    if "time" in raw_norm and "datetime" in clean.columns:
        t = df[raw_norm["time"]].astype("string").str.strip()
        base = clean["datetime"]
        combo = pd.to_datetime(base.dt.strftime("%Y-%m-%d") + " " + t.fillna(""), errors="coerce")
        clean["datetime"] = combo.fillna(base)
        used.add(raw_norm["time"])
        warnings.append(f"Combined {mapping.get('datetime')} and {raw_norm['time']} into Date / Time.")
    for role in ["quantity","price","cost","revenue","profit","stock","reorder","age"]:
        if role in clean: clean[role] = _parse_numeric(clean[role])
    for role in ["id","item","category","location","customer","patient","employee","status","gender"]:
        if role in clean: clean[role] = clean[role].astype("string").str.strip()
    if "revenue" not in clean and {"quantity","price"}.issubset(clean.columns):
        clean["revenue"] = clean["quantity"] * clean["price"]
        warnings.append("Revenue was derived from Quantity × Price because no revenue field was supplied.")
    if "profit" not in clean and {"revenue","cost","quantity"}.issubset(clean.columns):
        clean["profit"] = clean["revenue"] - clean["cost"] * clean["quantity"]
        warnings.append("Profit was derived from Revenue − Cost × Quantity because no profit field was supplied.")
    if "id" not in clean:
        clean["id"] = [f"ROW-{i+1:08d}" for i in range(len(df))]
        warnings.append("No reliable ID was detected; temporary row identifiers were created.")
    # Preserve every original source column so the analyst can inspect it.
    for c in df.columns:
        if c not in clean.columns and c not in used:
            clean[str(c)] = df[c]
    return clean.dropna(how="all").reset_index(drop=True), mapping, warnings

def profile(df: pd.DataFrame) -> dict:
    rows, cols = len(df), len(df.columns)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    return {"rows":rows,"columns":cols,"missing_cells":missing_cells,"missing_pct":100*missing_cells/max(1,rows*cols),"duplicates":duplicate_rows}
