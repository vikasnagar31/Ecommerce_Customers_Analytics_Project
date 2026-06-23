"""
train.py
--------
Trains the High Spender and Churn classification models.
Saves the full sklearn Pipeline (preprocessing + model) as .pkl files
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier


# ------------------------------------------------------------------------------------------
# CUSTOM SKLEARN TRANSFORMER CLASSES
# ------------------------------------------------------------------------------------------

class LeakageDropper(BaseEstimator, TransformerMixin):
    """Drop leakage columns and identifiers before any modelling."""
    def __init__(self, leakage_cols):
        self.leakage_cols = leakage_cols

    def fit(self, X, y=None): 
        return self

    def transform(self, X):
        return X.drop(columns=self.leakage_cols, errors='ignore')


class TypedImputer(BaseEstimator, TransformerMixin):
    """Apply separate median (numeric) and mode (categorical) imputers."""
    def __init__(self, num_strategy='median', cat_strategy='most_frequent'):
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy

    def fit(self, X, y=None):
        self._num_cols = X.select_dtypes(include='number').columns.tolist()
        self._obj_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
        self._num_imp  = SimpleImputer(strategy=self.num_strategy).fit(X[self._num_cols])
        self._cat_imp  = SimpleImputer(strategy=self.cat_strategy).fit(X[self._obj_cols])
        return self

    def transform(self, X):
        X = X.copy()
        if self._num_cols:
            X[self._num_cols] = self._num_imp.transform(X[self._num_cols])
        if self._obj_cols:
            X[self._obj_cols] = self._cat_imp.transform(X[self._obj_cols])
        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns to (Q1% - 0.5*std, Q99% + 0.5*std) bounds learned on train."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include='number').columns
        self.clip_cutoffs_ = {}
        for col in num_cols:
            lc = X[col].quantile(0.01) - 0.5 * X[col].std()
            uc = X[col].quantile(0.99) + 0.5 * X[col].std()
            self.clip_cutoffs_[col] = (lc, uc)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lc, uc) in self.clip_cutoffs_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lc, upper=uc)
        return X


class EncoderTransformer(BaseEstimator, TransformerMixin):
    """One-Hot encode nominal columns and Ordinal encode ordered columns."""
    def __init__(self, ohe_cols, ordinal_cols, ordinal_categories):
        self.ohe_cols           = ohe_cols
        self.ordinal_cols       = ordinal_cols
        self.ordinal_categories = ordinal_categories   # list of lists

    def fit(self, X, y=None):
        self._ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        self._ohe.fit(X[self.ohe_cols])
        self._oe  = OrdinalEncoder(categories=self.ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1)
        self._oe.fit(X[self.ordinal_cols])
        return self

    def transform(self, X):
        X = X.copy()
        # OHE
        ohe_names = list(self._ohe.get_feature_names_out(self.ohe_cols))
        ohe_df    = pd.DataFrame(self._ohe.transform(X[self.ohe_cols]), columns=ohe_names, index=X.index)
        X = pd.concat([X.drop(columns=self.ohe_cols), ohe_df], axis=1)
        # Ordinal
        X[self.ordinal_cols] = self._oe.transform(X[self.ordinal_cols])
        return X


class ColumnScaler(BaseEstimator, TransformerMixin):
    """Scale columns with more than 5 unique values (skip binary / low-cardinality)."""
    def fit(self, X, y=None):
        self._scale_cols = [c for c in X.columns if X[c].nunique() > 5]
        self._scaler = StandardScaler().fit(X[self._scale_cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self._scale_cols] = self._scaler.transform(X[self._scale_cols])
        return X


class VarianceFilter(BaseEstimator, TransformerMixin):
    """Remove near-zero-variance columns using VarianceThreshold."""
    def __init__(self, threshold=0.07):
        self.threshold = threshold

    def fit(self, X, y=None):
        self._vt     = VarianceThreshold(threshold=self.threshold)
        self._vt.fit(X)
        self._keep   = X.columns[self._vt.get_support()].tolist()
        return self

    def transform(self, X):
        return X[self._keep]


class CorrDropper(BaseEstimator, TransformerMixin):
    """Drop one column from every pair with |correlation| > threshold."""
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr   = X.corr().abs()
        upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self._drop = [c for c in upper.columns if any(upper[c] > self.threshold)]
        return self

    def transform(self, X):
        return X.drop(columns=self._drop, errors='ignore')


class UnionFeatureSelector(BaseEstimator, TransformerMixin):
    """Keep the union of features selected by RFE and SelectKBest."""
    def __init__(self, n_rfe=40, n_skb=35):
        self.n_rfe = n_rfe
        self.n_skb = n_skb

    def fit(self, X, y):
        rfe = RFE(estimator=XGBClassifier(random_state=42, n_jobs=-1),n_features_to_select=self.n_rfe)
        rfe.fit(X, y)
        skb = SelectKBest(score_func=f_classif, k=self.n_skb)
        skb.fit(X, y)
        self._keep = sorted(set(X.columns[rfe.support_]) | set(X.columns[skb.get_support()]))
        return self

    def transform(self, X):
        return X[[c for c in self._keep if c in X.columns]]


# ------------------------------------------------------------------------------------------
# COLUMN CONSTANTS  
# ------------------------------------------------------------------------------------------

LEAKAGE_COLS_SPD = ['user_id', 'first_purchase_date', 'recent_purchase_date',
                    'sale_amount', 'decile_sale_amount', 'decile_margin',
                    'p1_sale_amount', 'p2_sale_amount', 'act_amount', 'cost_amount',
                    'margin', 'avg_sale_amount', 'customer_value',
                     'churn_flag', 'high_spender_flag' ]

LEAKAGE_COLS_CHN = ['user_id', 'first_purchase_date', 'recent_purchase_date',
                    'sale_amount', 'decile_sale_amount', 'decile_margin',
                    'p1_sale_amount', 'p2_sale_amount', 'act_amount', 'recency_days',
                    'customer_segment', 'churn_flag', 'high_spender_flag']

OHE_COLS      = ['gender', 'buyer_flag']
ORDINAL_COLS  = ['customer_segment']
ORDINAL_CATS  = [['Churn', 'Inactive', 'New', 'DownwordMigrator', 'UpwordMigrator']]

XGB_PARAMS_SPD = dict(max_depth=5, n_estimators=200, learning_rate=0.01,subsample=0.8, colsample_bytree=0.8,
                       reg_lambda=1, min_child_weight=5, random_state=42)

XGB_PARAMS_CHN = dict(max_depth=4, n_estimators=150, learning_rate=0.03, gamma=0.2, min_child_weight=5, 
                      colsample_bytree=0.8, subsample=0.8, reg_lambda=1.5, random_state=42)


# ------------------------------------------------------------------------------------------
# TRAIN HIGH SPENDER MODEL
# ------------------------------------------------------------------------------------------

def train_high_spender(cust_360, model_dir = "models/high_spender/"):
    """
    Train the full High Spender prediction pipeline and save it as a .pkl file.
    """
    os.makedirs(model_dir, exist_ok=True)

    X = cust_360.drop(columns=LEAKAGE_COLS_SPD, errors='ignore').copy()
    y = cust_360['high_spender_flag'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"High Spender — Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class balance: {y_train.mean():.2%}")

    pipeline = Pipeline([
        ('impute',          TypedImputer()),
        ('clip_outliers',   OutlierClipper()),
        ('encode',          EncoderTransformer(OHE_COLS, ORDINAL_COLS, ORDINAL_CATS)),
        ('scale',           ColumnScaler()),
        ('variance_filter', VarianceFilter(threshold=0.07)),
        ('corr_drop',       CorrDropper(threshold=0.9)),
        ('feature_select',  UnionFeatureSelector(n_rfe=40, n_skb=35)),
        ('model',           XGBClassifier(**XGB_PARAMS_SPD)),
    ])
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    print(f"\nHigh Spender → Test ROC-AUC: {auc:.4f}")

    # Optimal threshold (max F1 on test)
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1     = 2 * prec * rec / (prec + rec + 1e-8)
    best_t = thresholds[np.argmax(f1)]
    print(f"Best threshold: {best_t:.4f}")
    print(classification_report(y_test, (y_prob >= best_t).astype(int)))

    # Save
    joblib.dump(pipeline,  os.path.join(model_dir, 'high_spender_pipeline.pkl'))
    joblib.dump(best_t,    os.path.join(model_dir, 'high_spender_threshold.pkl'))
    joblib.dump(LEAKAGE_COLS_SPD, os.path.join(model_dir, 'leakage_cols.pkl'))
    print(f"Saved to {model_dir} ✓")

    return pipeline


# ------------------------------------------------------------------------------------------
# TRAIN CHURN MODEL
# ------------------------------------------------------------------------------------------

def train_churn(cust_360, model_dir = "models/churn/") :
    """
    Train the full Churn prediction pipeline and save it as a .pkl file.
    Note: We filter to active-in-p1 customers only to reduce noise
    (customers with p1_sale_amount = 0 are inactive, not churners).
    """
    os.makedirs(model_dir, exist_ok=True)

    # Filter: only customers who were active in p1
    active_p1 = cust_360[cust_360['p1_sale_amount'] > 0].copy()
    print(f"Active-in-p1 customers for churn model: {active_p1.shape[0]}")

    X = active_p1.drop(columns=LEAKAGE_COLS_CHN, errors='ignore').copy()
    y = active_p1['churn_flag'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Churn — Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Churn rate: {y_train.mean():.2%}")

    pipeline = Pipeline([
        ('impute',          TypedImputer()),
        ('clip_outliers',   OutlierClipper()),
        ('encode',          EncoderTransformer(OHE_COLS, [], [])),   # no ordinal cols for churn
        ('scale',           ColumnScaler()),
        ('variance_filter', VarianceFilter(threshold=0.05)),
        ('corr_drop',       CorrDropper(threshold=0.9)),
        ('feature_select',  UnionFeatureSelector(n_rfe=35, n_skb=40)),
        ('model',           XGBClassifier(**XGB_PARAMS_CHN)),
    ])
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    print(f"\nChurn → Test ROC-AUC: {auc:.4f}")

    from sklearn.metrics import precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1     = 2 * prec * rec / (prec + rec + 1e-8)
    best_t = thresholds[np.argmax(f1)]
    print(f"Best threshold: {best_t:.4f}")
    print(classification_report(y_test, (y_prob >= best_t).astype(int)))

    # Save
    joblib.dump(pipeline,  os.path.join(model_dir, 'churn_pipeline.pkl'))
    joblib.dump(best_t,  os.path.join(model_dir, 'churn_threshold.pkl'))
    joblib.dump(LEAKAGE_COLS_CHN, os.path.join(model_dir, 'leakage_cols.pkl'))
    print(f"Saved to {model_dir}")

    return pipeline


# ------------------------------------------------------------------------------------------
# MASTER TRAIN FUNCTION
# ------------------------------------------------------------------------------------------

def run_training(cust_360):
    """
    Train both models.
    """
    print("\n" + "="*60)
    print("TRAINING HIGH SPENDER MODEL")
    print("="*60)
    spd_pipeline = train_high_spender(cust_360)

    print("\n" + "="*60)
    print("TRAINING CHURN MODEL")
    print("="*60)
    chn_pipeline = train_churn(cust_360)

    print("\nBoth models trained and saved.")
    return {'high_spender': spd_pipeline, 'churn': chn_pipeline}


if __name__ == "__main__":
    import sys
    sys.path.append('src')
    from preprocessing      import run_preprocessing
    from feature_engineering import run_feature_engineering

    cust, tran2  = run_preprocessing(data_dir="data/")
    cust_360     = run_feature_engineering(cust, tran2)
    run_training(cust_360)