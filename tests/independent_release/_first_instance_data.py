import numpy as np, pandas as pd
def make(n=600, seed=1, returned_col=False):
    r=np.random.default_rng(seed)
    d=pd.DataFrame({
     "order_id":[f"O{i}" for i in range(n)],
     "order_date":pd.date_range("2025-01-01",periods=n,freq="D").astype(str),
     "customer_id":[f"C{i%120}" for i in range(n)],
     "region":r.choice(["North","South","East","West"],n),
     "customer_segment":r.choice(["Corporate","Consumer","Small Business"],n),
     "sales_channel":r.choice(["Online","Retail","Partner"],n),
     "product_category":r.choice(["A","B","C"],n),
     "units":r.integers(1,9,n),
     "unit_list_price":r.uniform(10,200,n).round(2),
     "gross_sales":r.uniform(50,2000,n).round(2),
     "discount_pct":r.uniform(0,.3,n).round(3),
     "order_status":r.choice(["Delivered","Returned","Cancelled","Pending"],n,p=[.6,.15,.15,.1]),
    })
    if returned_col: d["is_returned"]=r.choice(["Yes","No"],n,p=[.2,.8])
    return d
