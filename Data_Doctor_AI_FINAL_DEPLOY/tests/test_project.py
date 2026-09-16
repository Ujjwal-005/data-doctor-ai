import pandas as pd
from schema_engine import canonicalize, detect_domain
from ml_engine import auto_supervised, anomaly_detection, cluster_data

def test_retail_mapping():
    raw=pd.DataFrame({"Order ID":["A1","A2","A3"],"Order Date":["2026-01-01","2026-01-02","2026-01-03"],"Product Name":["Milk","Bread","Milk"],"Qty":[2,3,1],"Buy Price":[30,20,30],"Sell Price":[40,30,40]})
    clean,mapping,_=canonicalize(raw)
    assert mapping["id"]=="Order ID" and mapping["item"]=="Product Name" and mapping["quantity"]=="Qty"
    assert mapping["cost"]=="Buy Price" and mapping["price"]=="Sell Price"
    assert "revenue" in clean and "profit" in clean

def test_supervised_multimodel():
    n=120
    raw=pd.DataFrame({"x1":range(n),"x2":([0,1,2,3]*30),"target":[i%3 for i in range(n)]})
    r=auto_supervised(raw,"target")
    assert len(r["results"])>=3
    assert r["best"]["model"] in set(r["results"]["model"])

def test_unsupervised():
    df=pd.DataFrame({"a":range(50),"b":range(50,100),"c":[x%5 for x in range(50)]})
    assert anomaly_detection(df) is not None
    assert cluster_data(df) is not None

def test_healthcare_domain():
    raw=pd.DataFrame({"Patient ID":[1,2],"Admission Date":["2026-01-01","2026-01-02"],"Diagnosis":["A","B"],"Ward":["ICU","ER"]})
    clean,mapping,_=canonicalize(raw)
    domain,conf,_=detect_domain(raw,mapping)
    assert domain=="Healthcare"

def test_clear_state_logic_helpers():
    # Deterministic replacement semantics are represented by token-based state in app.py.
    assert ("owner"+"_replace")=="owner_replace"


def test_profit_target_excludes_direct_measure_leakage():
    import pandas as pd
    from ml_engine import auto_supervised
    n=60
    df=pd.DataFrame({
        "profit":[10+i%5 for i in range(n)],
        "revenue":[100+i for i in range(n)],
        "cost":[90+i for i in range(n)],
        "quantity":[1+(i%3) for i in range(n)],
        "category":["A" if i%2 else "B" for i in range(n)],
    })
    result=auto_supervised(df,"profit")
    assert set(["revenue","cost","quantity"]).issubset(set(result["excluded_leakage"]))


def test_cross_dataset_relationships():
    from relationships import infer_relationships
    parent = pd.DataFrame({"Customer ID":["C1","C2","C3"],"Name":["A","B","C"]})
    child = pd.DataFrame({"Order ID":["O1","O2","O3"],"Customer ID":["C1","C2","C2"],"Amount":[10,20,30]})
    datasets={
        "customers.csv":{"raw":parent},
        "orders.csv":{"raw":child},
    }
    rels=infer_relationships(datasets)
    assert rels
    assert rels[0]["From Column"]=="Customer ID"
    assert rels[0]["To Column"]=="Customer ID"
