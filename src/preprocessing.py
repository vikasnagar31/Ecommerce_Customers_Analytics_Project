"""
preprocessing.py
----------------
Loads the 3 raw tables (Customer, Transactions, Product), cleans them, and produces a merged 
tran2 DataFrame ready for Customer360 feature engineering.
"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------------------
# 1. LOAD RAW DATA
# ------------------------------------------------------------------------------------------

def load_raw_data(data_dir = "data/"):
    cust = pd.read_csv(f"{data_dir}Customer.csv")
    tran = pd.read_csv(f"{data_dir}Transactions.csv")
    prod = pd.read_csv(f"{data_dir}Product.csv")
    print(f"Loaded Customer: {cust.shape}, Transactions: {tran.shape}, Product: {prod.shape}")
    return cust, tran, prod


# ------------------------------------------------------------------------------------------
# 2. CLEAN COLUMN NAMES  (strip spaces, lowercase, replace spaces with _)
# ------------------------------------------------------------------------------------------

def clean_column_names(*dfs):
    """
    Standardise column names: strip whitespace, lowercase, replace space with _.
    Accepts any number of DataFrames and returns them in the same order.
    """
    result = []
    for df in dfs:
        df = df.copy()
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        result.append(df)
    return result


# ------------------------------------------------------------------------------------------
# 3. TYPE CASTING
# ------------------------------------------------------------------------------------------

def cast_dtypes(cust, tran):
    """
    Convert date columns from int (YYYYMMDD) to datetime.
    Must be called AFTER clean_column_names.
    """
    cust = cust.copy()
    tran = tran.copy()

    # Customer date columns
    cust['first_date']  = pd.to_datetime(cust['first_date'].astype(str),  format='%Y%m%d')
    cust['recent_date'] = pd.to_datetime(cust['recent_date'].astype(str), format='%Y%m%d')

    # Transaction date column
    tran['order_time'] = pd.to_datetime(tran['order_time'].astype(str), format='%Y%m%d')

    print("Date columns cast to datetime ✓")
    return cust, tran


# ------------------------------------------------------------------------------------------
# 4. BASIC CLEANING
# ------------------------------------------------------------------------------------------

def clean_customer(cust):
    """
    - Fill nulls in points_redeemed with 0  (missing = never redeemed)
    - Remove returned transactions (customer_value < 0)
    - Rename customer_id → user_id for consistent joins
    """
    cust = cust.copy()
    cust['points_redeemed'] = cust['points_redeemed'].fillna(0)
    cust = cust[cust['customer_value'] >= 0].reset_index(drop=True)
    cust.rename(columns={'customer_id': 'user_id'}, inplace=True)
    print(f"Customer after cleaning: {cust.shape}")
    return cust


def clean_transactions(tran):
    """
    - Remove duplicate rows (keep first)
    - Remove returned records (sale_amount < 0)
    - Filter to the relevant 1-year period (from 2012-04-01)
    """
    tran = tran.copy()
    tran = tran.drop_duplicates(keep='first', ignore_index=True)
    tran = tran[tran['sale_amount'] >= 0].reset_index(drop=True)
    tran = tran[tran['order_time'] >= '2012-04-01 00:00:00'].reset_index(drop=True)
    print(f"Transactions after cleaning: {tran.shape}")
    return tran


def clean_product(prod):
    """
    - Rename cost_price column (spaces in name)
    - Group rare categories into 'Others' to reduce dimensionality
     """
    prod = prod.copy()

    # Map verbose category names to short clean labels
    category_map = {
        'Food':                                       'Food',
        'Kitchen cleaning':                           'Kitchen_Clean',
        'Beauty':                                     'Beauty',
        'Imported food':                              'Imported_Food',
        'Drinks':                                     'Drinks',
        'Mother and children':                        'Mother_Child',
        'Home':                                       'Home',
        'Nutrition and health':                       'Nutrition',
        'Household electrical appliances':            'HH_Electrical',
        'Computers, software, office supplies':       'Office_Computer',
        'Digital':                                    'Digital',
        'Mobile phones':                              'Mobiles',
        'Car related products':                       'Cat_Car_prods',
    }
    prod['new_category'] = prod['category_level2_name_eng'].map(category_map).fillna('Others')
    print(f"Product after cleaning: {prod.shape}")
    return prod


# ------------------------------------------------------------------------------------------
# 5. MERGE TABLES
# ------------------------------------------------------------------------------------------

def merge_tables(tran, prod):
    """
    Join transactions with product on (product_id, merchant_id).
    Then engineer transaction-level features.
    """
    tran2 = pd.merge(
        tran,prod[['product_id', 'product_code', 'merchant_id', 'merchant_name_eng','new_category', 'cost_price']],
        on=['product_id', 'merchant_id'],how='inner')

    # Transaction-level feature engineering
    tran2['first_purchase']  = np.where(tran2['order_time'] == tran2['order_time'].min(), 1, 0)
    tran2['recent_purchase'] = np.where(tran2['order_time'] == tran2['order_time'].max(), 1, 0)
    tran2['period']          = np.where(tran2['order_time'] < '2012-10-01', 'p1', 'p2')
    tran2['act_amount']      = tran2['web_portal_price'] * tran2['sale_number']   # MRP
    tran2['cost_amount']     = tran2['cost_price'] * tran2['sale_number']
    tran2['margin']          = tran2['sale_amount'] - tran2['cost_amount']
    tran2['discount']        = tran2['act_amount'] - tran2['sale_amount']
    tran2['weekend_flag']    = np.where(tran2['order_time'].dt.dayofweek <= 5, 'weekday', 'weekend')
    tran2['promo_flag']      = np.where(tran2['activity_id'] == 999999999, 'no promo', 'promo')

    print(f"Merged tran2: {tran2.shape}")
    return tran2


# ------------------------------------------------------------------------------------------
# 6. MASTER PIPELINE FUNCTION
# ------------------------------------------------------------------------------------------

def run_preprocessing(data_dir= "data/"):
    """
    One-call function that runs the full preprocessing pipeline.
    """
    # Load
    cust, tran, prod = load_raw_data(data_dir)

    # Standardise column names
    cust, tran, prod = clean_column_names(cust, tran, prod)

    # Type cast dates
    cust, tran = cast_dtypes(cust, tran)

    # Clean each table
    cust = clean_customer(cust)
    tran = clean_transactions(tran)
    prod = clean_product(prod)

    # Merge
    tran2 = merge_tables(tran, prod)

    print("\n Preprocessing complete.")
    return cust, tran2


# ------------------------------------------------------------------------------------------
# Quick sanity check 
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    cust, tran2 = run_preprocessing(data_dir="data/")
    print(cust.head(2))
    print(tran2.head(2))
