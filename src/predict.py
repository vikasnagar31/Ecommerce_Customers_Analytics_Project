"""
predict.py
----------
Loads the saved sklearn pipelines and predicts on new customer data.
"""

import pandas as pd
import numpy as np
import joblib
import os
from train import (LeakageDropper, TypedImputer, OutlierClipper, EncoderTransformer, ColumnScaler,
                    VarianceFilter, CorrDropper, UnionFeatureSelector)

# ------------------------------------------------------------------------------------------
# 1. LOAD SAVED PIPELINES
# ------------------------------------------------------------------------------------------

def load_high_spender_pipeline(model_dir = "models/high_spender/") :
    """Load the high spender pipeline and its best threshold."""
    pipeline  = joblib.load(os.path.join(model_dir, 'high_spender_pipeline.pkl'))
    threshold = joblib.load(os.path.join(model_dir, 'high_spender_threshold.pkl'))
    print(f"High Spender pipeline loaded from {model_dir}")
    return pipeline, threshold


def load_churn_pipeline(model_dir = "models/churn/") :
    """Load the churn pipeline and its best threshold."""
    pipeline  = joblib.load(os.path.join(model_dir, 'churn_pipeline.pkl'))
    threshold = joblib.load(os.path.join(model_dir, 'churn_threshold.pkl'))
    print(f"Churn pipeline loaded from {model_dir}")
    return pipeline, threshold


# ------------------------------------------------------------------------------------------
# 2. PREDICT FUNCTIONS
# ------------------------------------------------------------------------------------------

def predict_high_spender(customer_data, pipeline, threshold):
    """
    Score customers for High Spender probability.
    """
    user_ids = customer_data['user_id'].values if 'user_id' in customer_data.columns else range(len(customer_data))
    proba      = pipeline.predict_proba(customer_data)[:, 1]
    prediction = (proba >= threshold).astype(int)

    result = pd.DataFrame({'user_id': user_ids, 'high_spender_probability': proba.round(4),'high_spender_predicted': prediction,
                           'spend_tier': pd.cut(proba,bins=[0, 0.3, 0.5, 0.8, 1.0], labels=['Low', 'Medium', 'High', 'Very High']) })
    return result


def predict_churn(customer_data, pipeline, threshold):

    user_ids = customer_data['user_id'].values if 'user_id' in customer_data.columns else range(len(customer_data))
    proba      = pipeline.predict_proba(customer_data)[:, 1]
    prediction = (proba >= threshold).astype(int)

    result = pd.DataFrame({'user_id': user_ids,'churn_probability': proba.round(4),'churn_predicted': prediction,
        'churn_risk': pd.cut(proba, bins=[0, 0.3, 0.5, 0.8, 1.0], labels=['Low', 'Medium', 'High', 'Very High'] ) })

    return result


# ------------------------------------------------------------------------------------------
# 3. COMBINED SCORING (score all customers for both targets at once)
# ------------------------------------------------------------------------------------------

def score_all_customers(customer_data ,model_dir_spd = "models/high_spender/",model_dir_chn = "models/churn/"):
    """
    Run both pipelines on the customer data and return a combined scorecard.
    """
    # Load pipelines
    spd_pipeline, spd_threshold = load_high_spender_pipeline(model_dir_spd)
    chn_pipeline, chn_threshold = load_churn_pipeline(model_dir_chn)

    # Score
    spd_results = predict_high_spender(customer_data, spd_pipeline, spd_threshold)
    chn_results = predict_churn(customer_data, chn_pipeline, chn_threshold)

    # Combine
    scorecard = spd_results.merge(chn_results[['user_id', 'churn_probability', 'churn_predicted', 'churn_risk']],
                                  on='user_id', how='left')
    
    # Business Priority Score
    scorecard['priority_score'] = (scorecard['high_spender_probability'] * scorecard['churn_probability']).round(4)
    scorecard = scorecard.sort_values('priority_score',ascending=False).reset_index(drop=True)

    # Business priority flag: high spender who is also at risk of churn = top priority
    scorecard['action_priority'] = np.where(
        (scorecard['high_spender_predicted'] == 1) & (scorecard['churn_predicted'] == 1),
        '🔴 URGENT - Retain High Spender',
        np.where(
            scorecard['churn_predicted'] == 1,
            '🟡 Win-back Campaign',
            np.where(
                scorecard['high_spender_predicted'] == 1,
                '🟢 VIP Programme',
                '⚪ Standard'
            )
        )
    )
    
    print(f"\n Scored {len(scorecard)} customers.")
    print(f"\nAction Priority Distribution:")
    print(scorecard['action_priority'].value_counts())

    return scorecard


# ------------------------------------------------------------------------------------------
# Quick demo 
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append('src')

    # Load cust_360 (the saved file from feature_engineering.py)
    cust_360 = pd.read_csv("data/customer_360.csv")

    # Score first 50 customers as a demo
    sample = cust_360.iloc[:50].copy()
    scorecard = score_all_customers(sample)

    print("\nSample Scorecard:")
    print(scorecard[['user_id', 'high_spender_probability', 'churn_probability', 'action_priority']].head(10))

    # Save
    scorecard.to_csv("data/customer_scorecard.csv", index=False)
    print("\nSaved to data/customer_scorecard.csv")
