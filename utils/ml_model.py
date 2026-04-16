"""
Lead Scoring ML Model — scikit-learn GradientBoosting
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class LeadScoringModel:
    def __init__(self):
        self.model   = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=4, random_state=42
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=50, random_state=42
        )
        self.scaler   = StandardScaler()
        self.encoders = {}
        self.is_trained = False
        self.feature_cols = []
        self.auc_score    = 0.0

    # ─── FEATURE ENGINEERING ──────────────────
    def _build_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        d = df.copy()

        # Numeric features
        numeric = ['budget', 'employees', 'website_visits',
                   'emails_opened', 'meetings_held', 'days_in_pipeline']
        for col in numeric:
            if col not in d.columns:
                d[col] = 0
            d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0)

        # Derived features
        d['engagement_score'] = (
            d['website_visits'] * 0.3 +
            d['emails_opened']  * 0.4 +
            d['meetings_held']  * 0.3
        )
        d['budget_log']        = np.log1p(d['budget'])
        d['budget_per_employee'] = np.where(
            d['employees'] > 0, d['budget'] / d['employees'], 0
        )
        d['velocity'] = np.where(
            d['days_in_pipeline'] > 0,
            d['meetings_held'] / d['days_in_pipeline'], 0
        )

        # Categorical features
        cat_cols = ['industry', 'deal_stage', 'lead_source']
        for col in cat_cols:
            if col not in d.columns:
                d[col] = 'Unknown'
            d[col] = d[col].fillna('Unknown').astype(str)
            if fit:
                enc = LabelEncoder()
                d[col + '_enc'] = enc.fit_transform(d[col])
                self.encoders[col] = enc
            else:
                if col in self.encoders:
                    enc = self.encoders[col]
                    d[col + '_enc'] = d[col].apply(
                        lambda x: enc.transform([x])[0]
                        if x in enc.classes_ else -1
                    )
                else:
                    d[col + '_enc'] = 0

        feature_cols = (
            numeric +
            ['engagement_score', 'budget_log', 'budget_per_employee', 'velocity'] +
            [c + '_enc' for c in cat_cols]
        )

        if fit:
            self.feature_cols = feature_cols

        X = d[self.feature_cols if not fit else feature_cols].values
        return X

    # ─── SYNTHETIC TARGET ─────────────────────
    def _make_target(self, df: pd.DataFrame) -> np.ndarray:
        """Agar 'converted' ustuni bo'lmasa, heuristik bilan yaratadi."""
        if 'converted' in df.columns:
            return df['converted'].astype(int).values

        d = df.copy()
        for col in ['budget','website_visits','emails_opened','meetings_held','days_in_pipeline']:
            if col not in d.columns:
                d[col] = 0
            d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0)

        score = (
            (d['budget'] / d['budget'].max().clip(1)) * 0.3 +
            (d['website_visits'] / d['website_visits'].max().clip(1)) * 0.2 +
            (d['emails_opened']  / d['emails_opened'].max().clip(1))  * 0.25 +
            (d['meetings_held']  / d['meetings_held'].max().clip(1))  * 0.25
        )

        # Stage bonus
        if 'deal_stage' in d.columns:
            stage_bonus = d['deal_stage'].map({
                'Negotiation': 0.3, 'Proposal': 0.2,
                'Demo':        0.1, 'Qualified': 0.05, 'New': 0.0
            }).fillna(0)
            score += stage_bonus

        # Probabilistic conversion
        np.random.seed(42)
        threshold = np.random.uniform(0.4, 0.7, len(score))
        return (score > threshold).astype(int)

    # ─── TRAIN ────────────────────────────────
    def train(self, df: pd.DataFrame):
        X = self._build_features(df, fit=True)
        y = self._make_target(df)

        if len(np.unique(y)) < 2:
            # Edge case: ensure both classes exist
            y[:max(1, len(y)//3)] = 1

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_tr_sc = self.scaler.fit_transform(X_tr)
        X_te_sc = self.scaler.transform(X_te)

        self.model.fit(X_tr_sc, y_tr)
        self.rf_model.fit(X_tr_sc, y_tr)

        try:
            prob = self.model.predict_proba(X_te_sc)[:, 1]
            self.auc_score = roc_auc_score(y_te, prob)
        except Exception:
            self.auc_score = 0.0

        self.is_trained = True

    # ─── PREDICT ──────────────────────────────
    def predict(self, df: pd.DataFrame) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model hali o'qitilmagan. train() chaqiring.")

        X    = self._build_features(df, fit=False)
        X_sc = self.scaler.transform(X)

        gb_prob = self.model.predict_proba(X_sc)[:, 1]
        rf_prob = self.rf_model.predict_proba(X_sc)[:, 1]
        prob    = (gb_prob * 0.6 + rf_prob * 0.4)   # ensemble

        predicted = (prob >= 0.5).astype(int)

        # Priority
        priority = np.where(prob >= 0.7, 'High',
                   np.where(prob >= 0.4, 'Medium', 'Low'))

        # Recommended price (budget * conversion-adjusted factor)
        budget = pd.to_numeric(df.get('budget', pd.Series([0]*len(df))),
                               errors='coerce').fillna(0).values
        rec_price = budget * (0.7 + 0.3 * prob)   # aggressive when high prob

        return {
            'probability':      prob,
            'predicted':        predicted,
            'priority':         priority,
            'recommended_price': rec_price,
            'auc_score':        self.auc_score,
        }

    # ─── FEATURE IMPORTANCE ───────────────────
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_trained:
            return pd.DataFrame()
        imp = self.model.feature_importances_
        return pd.DataFrame({
            'feature':    self.feature_cols,
            'importance': imp
        }).sort_values('importance', ascending=False)