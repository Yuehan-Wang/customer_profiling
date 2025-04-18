import pandas as pd
import os
from datetime import timedelta
import re

def clean_order_csv(file_path):
    df = pd.read_csv(file_path)

    fields = ['Order Date', 'Product Name', 'Shipping Address', 'Unit Price']
    for f in fields:
        if f not in df.columns:
            raise ValueError(f"Missing required column: {f}")

    df = df[fields].dropna(subset=['Order Date', 'Product Name'])

    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df = df.dropna(subset=['Order Date'])
    df['Order Date'] = df['Order Date'].dt.date

    df['Unit Price'] = df['Unit Price'].fillna(0).apply(lambda x: f"${float(x):.2f}")

    seen_names = set()
    def simplify_address(addr):
        addr = str(addr)
        name_match = re.match(r"^(.*?)\d{3,5}", addr)
        name = name_match.group(1).strip() if name_match else ""
        stripped_addr = re.sub(r"^.*?\d{3,5}", "", addr)
        stripped_addr = re.sub(r"(?i)united states", "", stripped_addr)
        stripped_addr = re.sub(r"(?i)[a-z ]*house rm \d+[a-z]?", "", stripped_addr)
        cleaned = " ".join(stripped_addr.strip().split())
        if name and name not in seen_names:
            seen_names.add(name)
            return f"{name} {cleaned}".strip()
        return cleaned

    df['Shipping Address'] = df['Shipping Address'].fillna("unknown address").apply(simplify_address)

    df = df.sort_values(by='Order Date')
    filtered_rows = []
    last_seen = {}

    for _, row in df.iterrows():
        address = row['Shipping Address']
        date = row['Order Date']
        last_date = last_seen.get(address)
        if last_date is None or (date - last_date) > timedelta(days=30):
            filtered_rows.append(row)
            last_seen[address] = date

    df = pd.DataFrame(filtered_rows)

    lines = df.apply(
        lambda row: f"{row['Order Date']}: {row['Product Name']} - {row['Unit Price']} - shipped to {row['Shipping Address']}",
        axis=1
    ).tolist()

    return '\n'.join(lines)


if __name__ == "__main__":
    data_path = os.path.join("data", "Retail.OrderHistory.1.csv")
    print(clean_order_csv(data_path))