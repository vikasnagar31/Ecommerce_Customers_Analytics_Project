"""
feature_engineering.py
-----------------------
Builds the Customer360 flat table from the cleaned cust and tran2 DataFrames,
computes RFM scores, and defines the two target variables.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ------------------------------------------------------------------------------------------
# 1. CATEGORY-LEVEL AGGREGATIONS
# ------------------------------------------------------------------------------------------

def make_category_features(tran2):
    """
    Build 5 category-level feature blocks, each pivoted wide.
    """
    feats = {}

    # Number of products purchased per user per category

    cat_prod_cnt = (tran2.groupby(['user_id', 'new_category'])['new_category'].count().reset_index(name='freq')
                    .pivot_table(index='user_id', columns='new_category', values='freq').fillna(0).reset_index())
    
    cat_prod_cnt.columns = ['user_id'] + ['cat_prod_cnt_' + c for c in cat_prod_cnt.columns[1:]]
    feats['cat_prod_cnt'] = cat_prod_cnt

    # Total revenue per user per category

    cat_amount = (
        tran2.groupby(['user_id', 'new_category'])[['sale_amount']].sum()
        .add_prefix('tot_').reset_index()
        .pivot_table(index='user_id', columns='new_category', values='tot_sale_amount', aggfunc='sum')
        .fillna(0).reset_index())
    
    cat_amount.columns = ['user_id'] + ['cat_amount_' + c for c in cat_amount.columns[1:]]
    feats['cat_amount'] = cat_amount

    # Category penetration = (distinct orders in category) / (total orders)

    orders_per_user_cat = tran2.groupby(['user_id', 'new_category'])[['order_time']].nunique().add_prefix('tot_').reset_index()
    orders_per_user     = tran2.groupby('user_id').agg(tot_no_of_orders=('order_time', 'nunique')).reset_index()

    cat_pentr = orders_per_user_cat.merge(orders_per_user, on='user_id', how='inner')
    cat_pentr['cat_pentr'] = cat_pentr['tot_order_time'] / cat_pentr['tot_no_of_orders']
    cat_pentr = (cat_pentr.pivot_table(index='user_id', columns='new_category', values='cat_pentr').fillna(0)
                 .reset_index().sort_values('user_id'))
    
    cat_pentr.columns = ['user_id'] + ['cat_pentr_' + c for c in cat_pentr.columns[1:]]
    feats['cat_penetration'] = cat_pentr

    # Binary purchase flag per user per category

    cat_pf = (
        tran2.groupby(['user_id', 'new_category'])[['new_category']].count()
        .add_prefix('tot_').reset_index())
    cat_pf['tot_new_category'] = np.where(cat_pf['tot_new_category'] == 0, 0, 1)
    cat_pf = (
        cat_pf.pivot_table(index='user_id', columns='new_category', values='tot_new_category')
        .fillna(0).reset_index().sort_values('user_id'))
    cat_pf.columns = ['user_id'] + ['purchase_flag_' + c for c in cat_pf.columns[1:]]
    feats['purchase_flag'] = cat_pf

    # First purchase flag per user per category

    cat_fp = (tran2.groupby(['user_id', 'new_category'])[['first_purchase']].sum().add_prefix('tot_').reset_index())
    cat_fp['tot_first_purchase'] = np.where(cat_fp['tot_first_purchase'] == 0, 0, 1)
    cat_fp = (
        cat_fp.pivot_table(index='user_id', columns='new_category', values='tot_first_purchase')
        .fillna(0).reset_index().sort_values('user_id'))
    cat_fp.columns = ['user_id'] + ['first_purchase_' + c for c in cat_fp.columns[1:]]
    feats['first_purchase'] = cat_fp

    print(f"Category feature blocks built: {list(feats.keys())}")
    return feats


# ------------------------------------------------------------------------------------------
# 2. BUILD CUSTOMER 360
# ------------------------------------------------------------------------------------------

def build_customer_360(cust, tran2):
    """
    Merge customer base with all transaction aggregations to produce
    one flat row per customer — the Customer360 table.
    """
    cat_feats = make_category_features(tran2)

    cust_360 = (
        cust
        .merge(tran2.groupby('user_id').agg(no_of_baskets=('order_id', 'nunique')).reset_index(),             on='user_id', how='left')
        .merge(tran2.groupby('user_id').agg(qty=('sale_number', 'sum')).reset_index(),                        on='user_id', how='left')
        .merge(tran2.groupby('user_id').agg(no_of_SKUs=('product_id', 'count')).reset_index(),                on='user_id', how='left')
        .merge(tran2.groupby('user_id').agg(no_of_dist_SKUs=('product_id', 'nunique')).reset_index(),         on='user_id', how='left')
        .merge(tran2.groupby('user_id').agg(no_of_cat=('new_category', 'count')).reset_index(),               on='user_id', how='left')
        .merge(tran2.groupby('user_id').agg(no_of_dist_cat=('new_category', 'nunique')).reset_index(),        on='user_id', how='left')
        .merge(tran2.groupby('user_id').agg(sale_amount=('sale_amount', 'sum')).reset_index(),                on='user_id', how='left')
        .merge(tran2[tran2['period'] == 'p1'].groupby('user_id').agg(p1_sale_amount=('sale_amount', 'sum')).reset_index(), on='user_id', how='left')
        .merge(tran2[tran2['period'] == 'p2'].groupby('user_id').agg(p2_sale_amount=('sale_amount', 'sum')).reset_index(), on='user_id', how='left')
        .merge(tran2.groupby('user_id')[['act_amount', 'cost_amount', 'discount', 'margin']].sum().reset_index(), on='user_id', how='left')
        .merge(tran2[tran2['weekend_flag'] == 'weekday'].groupby('user_id').agg(weekday_orders=('order_time', 'nunique')).reset_index(), on='user_id', how='left')
        .merge(tran2[tran2['weekend_flag'] == 'weekend'].groupby('user_id').agg(weekend_orders=('order_time', 'nunique')).reset_index(), on='user_id', how='left')
        .merge(tran2[tran2['promo_flag'] == 'no promo'].groupby('user_id').agg(without_promo_orders=('order_time', 'nunique')).reset_index(), on='user_id', how='left')
        .merge(tran2[tran2['promo_flag'] == 'promo'].groupby('user_id').agg(promo_orders=('order_time', 'nunique')).reset_index(), on='user_id', how='left')
        .merge(tran2[tran2['promo_flag'] == 'promo'].groupby('user_id').agg(promo_prods=('product_id', 'count')).reset_index(), on='user_id', how='left')
        .merge(tran2[tran2['discount'] > 0].groupby('user_id').agg(prods_with_discount=('product_id', 'count')).reset_index(), on='user_id', how='left')
        .merge(tran2[tran2['discount'] <= 0].groupby('user_id').agg(prods_without_discount=('product_id', 'count')).reset_index(), on='user_id', how='left')
        .merge(cat_feats['cat_prod_cnt'],   on='user_id', how='left')
        .merge(cat_feats['cat_amount'],     on='user_id', how='left')
        .merge(cat_feats['cat_penetration'],on='user_id', how='left')
        .merge(cat_feats['purchase_flag'],  on='user_id', how='left')
        .merge(cat_feats['first_purchase'], on='user_id', how='left')
    )

    # Drop customers with no transaction data (inactive in period)
    cust_360 = cust_360[~cust_360['sale_amount'].isna()].reset_index(drop=True)

    print(f"Customer360 shape: {cust_360.shape}")
    return cust_360


# ------------------------------------------------------------------------------------------
# 3. DERIVED FEATURES + MISSINGS FILL
# ------------------------------------------------------------------------------------------

def add_derived_features(cust_360):
    """
    Adds:
      - Average per-basket metrics
      - Percentage features (margin_pct, discount_pct)
      - Behavioural flags (buyer_flag, multi_cat_flag, reedem_flag, promo_seeker_flag)
      - Decile segments (decile_sale_amount, decile_margin)
      - recency_days
      - customer_segment (heuristic: New / Churn / Up / Down / Inactive)
    Also fills nulls created by conditional merges.
    """
    df = cust_360.copy()

    # ── Fill nulls that mean "no activity happened" with 0 ──
    zero_fill_cols = ['p1_sale_amount', 'p2_sale_amount', 'prods_with_discount',
                      'prods_without_discount', 'promo_orders', 'promo_prods', 'weekend_orders']
    df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

    # Median fill for low-null columns
    for col in ['weekday_orders', 'without_promo_orders']:
        df[col] = df[col].fillna(df[col].median())

    # ── Averages per basket ──
    nb = df['no_of_baskets'].replace(0, np.nan)   # avoid division by zero
    df['avg_sale_amount'] = df['sale_amount']  / nb
    df['avg_qty']         = df['qty']          / nb
    df['avg_no_prods']    = df['no_of_SKUs']   / nb
    df['avg_no_cat']      = df['no_of_cat']    / nb
    df['avg_margin']      = df['margin']       / nb
    df['avg_discount']    = df['discount']     / nb
    df[['avg_sale_amount','avg_qty','avg_no_prods','avg_no_cat','avg_margin','avg_discount']] = \
        df[['avg_sale_amount','avg_qty','avg_no_prods','avg_no_cat','avg_margin','avg_discount']].fillna(0)

    # ── Percentages ──
    df['margin_pct']   = np.where(df['sale_amount'] == 0, 0.0, df['margin'] / df['sale_amount'])
    df['discount_pct'] = np.where(df['act_amount']  == 0, 0.0, df['discount'] / df['act_amount'])

    # ── Behavioural flags ──
    df['buyer_flag']     = np.where(df['no_of_baskets'] == 1, 'one_time_buyer', 'repeat_buyer')
    df['multi_cat_flag'] = np.where(df['no_of_dist_cat'] == 1, 0, 1)
    df['reedem_flag']    = np.where(df['points_redeemed'] > 0, 1, 0)

    # Promo metrics (needs filled promo cols)
    df['promo_prod_pct']         = np.where(df['no_of_SKUs']    == 0, 0.0, df['promo_prods']  / df['no_of_SKUs'])
    df['pct_pur_with_promo_prods'] = np.where(df['no_of_baskets'] == 0, 0.0, df['promo_orders'] / df['no_of_baskets'])
    df['promo_seeker_flag'] = np.where((df['pct_pur_with_promo_prods'] <= 0.80) & (df['promo_prod_pct'] <= 0.50), 0, 1)

    # ── Decile segments ──
    df['decile_sale_amount'] = pd.qcut(df['sale_amount'], q=10, labels=False)
    df['decile_margin']      = pd.qcut(df['margin'],      q=10, labels=False)

    # ── Recency (days since most recent purchase in dataset) ──
    df['recency_days'] = (df['recent_date'].max() - df['recent_date']).dt.days

    # ── Heuristic customer segment ──
    df['customer_segment'] = np.where(
        (df['p1_sale_amount'] == 0) & (df['p2_sale_amount'] > 0), 'New',
        np.where(
            (df['p1_sale_amount'] > 0) & (df['p2_sale_amount'] == 0), 'Churn',
            np.where(
                df['p1_sale_amount'] > df['p2_sale_amount'], 'DownwordMigrator',
                np.where(df['p1_sale_amount'] < df['p2_sale_amount'], 'UpwordMigrator', 'Inactive')
            )
        )
    )

    # ── Rename date columns ──
    df.rename(columns={'first_date': 'first_purchase_date', 'recent_date': 'recent_purchase_date'}, inplace=True)

    print("Derived features added ✓")
    return df


# ------------------------------------------------------------------------------------------
# 4. TARGET VARIABLES
# ------------------------------------------------------------------------------------------

def add_target_variables(cust_360):
    """
    Adds:
      - high_spender_flag : top decile (decile 9) of sale_amount = 1
      - churn_flag        : purchased in p1 but NOT in p2 = 1
    """
    df = cust_360.copy()
    df['high_spender_flag'] = np.where(df['decile_sale_amount'] == 9, 1, 0)
    df['churn_flag']        = np.where((df['p1_sale_amount'] > 0) & (df['p2_sale_amount'] == 0), 1, 0)
    print(f"high_spender_flag distribution:\n{df['high_spender_flag'].value_counts()}")
    print(f"churn_flag distribution:\n{df['churn_flag'].value_counts()}")
    return df


# ------------------------------------------------------------------------------------------
# 5. RFM SEGMENTATION  (KMeans on R, F, M separately)
# ------------------------------------------------------------------------------------------

def _order_cluster(cluster_col, target_col, df, ascending ):
    """
    Re-labels KMeans cluster IDs so that cluster 0 = worst, 4 = best
    for the given metric.
    """
    mapping = (
        df.groupby(cluster_col)[target_col].mean()
        .reset_index()
        .sort_values(by=target_col, ascending=ascending)
        .reset_index(drop=True)
    )
    mapping['new_label'] = mapping.index
    df = df.merge(mapping[[cluster_col, 'new_label']], on=cluster_col)
    df = df.drop(columns=cluster_col).rename(columns={'new_label': cluster_col})
    return df


def run_rfm_segmentation(cust_360_scaled, random_state= 42):
    """
    Run KMeans (k=5) on Recency, Frequency, Monetary separately.
    Compute overall_score = r + f + m.

    """
    df = cust_360_scaled.copy()

    # Recency (lower is better → ascending=False)
    km_r = KMeans(n_clusters=5, max_iter=1000, random_state=random_state)
    df['r_cluster'] = km_r.fit_predict(df[['recency_days']])
    df = _order_cluster('r_cluster', 'recency_days', df, ascending=False)

    # Frequency (higher is better → ascending=True)
    km_f = KMeans(n_clusters=5, max_iter=1000, random_state=random_state)
    df['f_cluster'] = km_f.fit_predict(df[['no_of_baskets']])
    df = _order_cluster('f_cluster', 'no_of_baskets', df, ascending=True)

    # Monetary (higher is better → ascending=True)
    km_m = KMeans(n_clusters=5, max_iter=1000, random_state=random_state)
    df['m_cluster'] = km_m.fit_predict(df[['sale_amount']])
    df = _order_cluster('m_cluster', 'sale_amount', df, ascending=True)

    # Overall RFM score
    df['overall_score'] = df['r_cluster'] + df['f_cluster'] + df['m_cluster']

    # Business segment label
    def _label(score):
        if score >= 10: return 'High Spenders'
        if score >= 8:  return 'Loyal'
        if score >= 6:  return 'Potential'
        if score >= 4:  return 'At Risk'
        if score >= 2:  return 'Inactivate'
        return 'Churned'

    df['segment'] = df['overall_score'].apply(_label)
    print(f"RFM segment distribution:\n{df['segment'].value_counts()}")
    return df


# ------------------------------------------------------------------------------------------
# 6. MASTER PIPELINE FUNCTION
# ------------------------------------------------------------------------------------------

def run_feature_engineering(cust, tran2):
    """
    One-call function: builds Customer360, adds derived features + targets.
    RFM segmentation is kept separate (requires scaling first).
    """
    cust_360 = build_customer_360(cust, tran2)
    cust_360 = add_derived_features(cust_360)
    cust_360 = add_target_variables(cust_360)
    cust_360 = cust_360.reset_index(drop=True)
    print("\n Feature engineering complete.")
    return cust_360


if __name__ == "__main__":
    from preprocessing import run_preprocessing
    cust, tran2 = run_preprocessing(data_dir="data/")   
    cust_360 = run_feature_engineering(cust, tran2)
    cust_360.to_csv("data/customer_360.csv", index=False)   
    print("Saved to data/customer_360.csv")
    print(cust_360.shape)
