"""Generate Realistic Enterprise Seed Datasets for Benchmarks & Development."""
import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

SEED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "seed")
os.makedirs(SEED_DIR, exist_ok=True)


def generate_all_datasets():
    print("Generating benchmark seed datasets...")

    # 1. Products
    products = [
        {"product_id": "PRD-101", "product_name": "UltraBook Pro 15", "category": "Laptops", "unit_cost": 750.0, "list_price": 1200.0, "stock_level": 450},
        {"product_id": "PRD-102", "product_name": "CloudPad 10", "category": "Tablets", "unit_cost": 220.0, "list_price": 400.0, "stock_level": 800},
        {"product_id": "PRD-103", "product_name": "Product X Pro Hub", "category": "Accessories", "unit_cost": 45.0, "list_price": 150.0, "stock_level": 120},
        {"product_id": "PRD-104", "product_name": "NoiseCancel Elite Headphones", "category": "Audio", "unit_cost": 90.0, "list_price": 250.0, "stock_level": 600},
        {"product_id": "PRD-105", "product_name": "Vision 4K Monitor", "category": "Displays", "unit_cost": 280.0, "list_price": 550.0, "stock_level": 300},
        {"product_id": "PRD-106", "product_name": "ErgoSmart Keyboard", "category": "Accessories", "unit_cost": 35.0, "list_price": 95.0, "stock_level": 1200},
        {"product_id": "PRD-107", "product_name": "HyperSync Mouse", "category": "Accessories", "unit_cost": 20.0, "list_price": 60.0, "stock_level": 1500},
        {"product_id": "PRD-108", "product_name": "Enterprise Server Node", "category": "Enterprise", "unit_cost": 2200.0, "list_price": 4500.0, "stock_level": 80},
    ]
    df_products = pd.DataFrame(products)
    prod_path = os.path.join(SEED_DIR, "products.csv")
    df_products.to_csv(prod_path, index=False)
    print(f"-> Saved {len(df_products)} products to {prod_path}")

    # 2. Customers
    segments = ["Enterprise", "SMB", "Consumer"]
    regions = ["Region A", "Region B", "Region C", "Region D"]
    customers = []
    for i in range(1, 1001):
        cid = f"CUST-{i:04d}"
        seg = random.choices(segments, weights=[0.2, 0.3, 0.5])[0]
        reg = regions[(i - 1) % 4]
        signup = (datetime(2025, 1, 1) + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
        customers.append({
            "customer_id": cid,
            "customer_name": f"Customer {i}",
            "email": f"cust_{i}@example.com",
            "segment": seg,
            "region": reg,
            "signup_date": signup,
        })
    df_customers = pd.DataFrame(customers)
    cust_path = os.path.join(SEED_DIR, "customers.csv")
    df_customers.to_csv(cust_path, index=False)
    print(f"-> Saved {len(df_customers)} customers to {cust_path}")

    # 3. Sales Transactions
    start_date = datetime(2026, 1, 1)
    sales = []
    
    # Controlled transactions per day
    for day in range(120):
        current_dt = start_date + timedelta(days=day)
        month = current_dt.month

        # Base number of transactions per day
        base_tx_count = 150
        for _ in range(base_tx_count):
            cust = random.choice(customers)
            reg = cust["region"]
            prod = random.choice(products)
            pid = prod["product_id"]

            # Ground truth anomaly: In March (month 3), Region B suffers major slump (70% decline)
            if month == 3 and reg == "Region B":
                if random.random() < 0.70:
                    continue

            qty = random.randint(1, 5)
            unit_price = prod["list_price"]
            discount_rate = round(random.choice([0.0, 0.05, 0.10, 0.15]), 2)
            
            gross = round(qty * unit_price, 2)
            revenue = round(gross * (1.0 - discount_rate), 2)
            cost = round(qty * prod["unit_cost"], 2)
            profit = round(revenue - cost, 2)

            sales.append({
                "order_id": f"ORD-{len(sales)+1:06d}",
                "order_date": current_dt.strftime("%Y-%m-%d"),
                "customer_id": cust["customer_id"],
                "product_id": pid,
                "region": reg,
                "quantity": qty,
                "unit_price": unit_price,
                "discount_rate": discount_rate,
                "revenue": revenue,
                "cost": cost,
                "profit": profit,
            })

    df_sales = pd.DataFrame(sales)
    sales_path = os.path.join(SEED_DIR, "sales.csv")
    df_sales.to_csv(sales_path, index=False)
    print(f"-> Saved {len(df_sales)} sales transactions to {sales_path}")

    # 4. Marketing Campaigns
    channels = ["Google Search", "Meta Ads", "LinkedIn", "Email", "Direct"]
    marketing = []
    for day in range(120):
        dt = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")
        for ch in channels:
            spend = round(random.uniform(200.0, 1500.0), 2)
            impr = int(spend * random.uniform(20, 50))
            clicks = int(impr * random.uniform(0.01, 0.05))
            conv = int(clicks * random.uniform(0.02, 0.08))
            marketing.append({
                "date": dt,
                "channel": ch,
                "spend": spend,
                "impressions": impr,
                "clicks": clicks,
                "conversions": conv,
            })
    df_marketing = pd.DataFrame(marketing)
    mkt_path = os.path.join(SEED_DIR, "marketing.csv")
    df_marketing.to_csv(mkt_path, index=False)
    print(f"-> Saved {len(df_marketing)} marketing records to {mkt_path}")

    # 5. Inventory
    inventory = []
    for p in products:
        for r in regions:
            stock = p["stock_level"] // 4
            if r == "Region B" and p["product_id"] == "PRD-103":
                stock = 4
            inventory.append({
                "product_id": p["product_id"],
                "warehouse_region": r,
                "current_stock": stock,
                "reorder_point": 30,
                "last_restock_date": "2026-02-15",
            })
    df_inv = pd.DataFrame(inventory)
    inv_path = os.path.join(SEED_DIR, "inventory.csv")
    df_inv.to_csv(inv_path, index=False)
    print(f"-> Saved {len(df_inv)} inventory records to {inv_path}")
    print("\nAll seed datasets successfully generated!")


if __name__ == "__main__":
    generate_all_datasets()
