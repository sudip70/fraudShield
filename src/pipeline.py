"""
FraudShield ML Pipeline
-----------------------
Full-stack fraud detection pipeline with:
  - Feature engineering & preprocessing
  - 3-model comparison with proper class-imbalance handling
  - Cross-validation scoring
  - SHAP explainability
  - Threshold analysis
  - Calibration curve
  - Business impact metrics
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ── Try importing SHAP (optional but recommended) ─────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  shap not installed — explainability features disabled. Run: pip install shap")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time features
    df["Hour"]      = pd.to_datetime(df["Transaction_Time"], format="%H:%M", errors="coerce").dt.hour
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    df["DayOfWeek"] = df["Transaction_Date"].dt.dayofweek
    df["Month"]     = df["Transaction_Date"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["IsNight"]   = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)

    # Amount deviation signals
    df["Amount_vs_Avg"] = (
        df["Transaction_Amount (in Million)"] /
        (df["Avg_Transaction_Amount (in Million)"] + 1e-9)
    )
    df["Amount_vs_Max24h"] = (
        df["Transaction_Amount (in Million)"] /
        (df["Max_Transaction_Last_24h (in Million)"] + 1e-9)
    )
    df["Balance_vs_Amount"] = (
        df["Account_Balance (in Million)"] /
        (df["Transaction_Amount (in Million)"] + 1e-9)
    )
    # How much of balance is being spent in one go?
    df["Spend_Ratio"] = (
        df["Transaction_Amount (in Million)"] /
        (df["Account_Balance (in Million)"] + 1e-9)
    ).clip(0, 10)

    # Location risk
    df["Location_Mismatch"] = (
        df["Transaction_Location"] != df["Customer_Home_Location"]
    ).astype(int)

    # Velocity signals
    df["Tx_Velocity_Ratio"] = (
        df["Daily_Transaction_Count"] /
        (df["Weekly_Transaction_Count"] / 7 + 1e-9)
    ).clip(0, 10)

    # Composite risk score (raw, before model)
    df["Risk_Flag_Count"] = (
        (df["Is_International_Transaction"] == "Yes").astype(int) +
        (df["Is_New_Merchant"] == "Yes").astype(int)             +
        (df["Unusual_Time_Transaction"] == "Yes").astype(int)    +
        df["Location_Mismatch"]                                  +
        (df["Failed_Transaction_Count"] > 0).astype(int)         +
        (df["Previous_Fraud_Count"] > 0).astype(int)
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE LISTS
# ══════════════════════════════════════════════════════════════════════════════

NUMERIC_FEATURES = [
    "Transaction_Amount (in Million)",
    "Distance_From_Home",
    "Account_Balance (in Million)",
    "Daily_Transaction_Count",
    "Weekly_Transaction_Count",
    "Avg_Transaction_Amount (in Million)",
    "Max_Transaction_Last_24h (in Million)",
    "Failed_Transaction_Count",
    "Previous_Fraud_Count",
    "Hour",
    "DayOfWeek",
    "Month",
    "IsWeekend",
    "IsNight",
    "Amount_vs_Avg",
    "Amount_vs_Max24h",
    "Balance_vs_Amount",
    "Spend_Ratio",
    "Location_Mismatch",
    "Tx_Velocity_Ratio",
    "Risk_Flag_Count",
]

CATEGORICAL_FEATURES = [
    "Transaction_Type",
    "Merchant_Category",
    "Card_Type",
    "Is_International_Transaction",
    "Is_New_Merchant",
    "Unusual_Time_Transaction",
]

TARGET = "Fraud_Label"


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = engineer_features(df)

    if fit:
        encoders = {}

    cat_encoded = []
    for col in CATEGORICAL_FEATURES:
        if fit:
            le = LabelEncoder()
            le.fit(df[col].fillna("Unknown").astype(str))
            encoders[col] = le
        le = encoders[col]
        encoded = le.transform(
            df[col].fillna("Unknown").astype(str).map(
                lambda x, le=le: x if x in le.classes_ else le.classes_[0]
            )
        )
        cat_encoded.append(pd.Series(encoded, name=col, index=df.index))

    X = pd.concat(
        [df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())]
        + cat_encoded,
        axis=1,
    )

    if fit:
        y = (df[TARGET].fillna("Normal") == "Fraud").astype(int)
        return X, y, encoders
    else:
        return X


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD ANALYSIS  (business-critical for fraud)
# ══════════════════════════════════════════════════════════════════════════════

def compute_threshold_analysis(y_test, y_prob, avg_fraud_amount: float = 5.0):
    """
    At each threshold, compute classifier metrics AND business cost.
    Assumes:
      - FN cost (missed fraud)    = avg fraud transaction amount
      - FP cost (blocked legit)   = investigation overhead (default $50 / ~0.00005M)
    """
    y_test = np.array(y_test)
    y_prob = np.array(y_prob)

    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p  = precision_score(y_test, y_pred, zero_division=0)
        r  = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "threshold": round(float(t), 3),
            "precision": float(p),
            "recall":    float(r),
            "f1":        float(f1),
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
            # Business: scale FP cost and FN cost per transaction unit
            "cost_fn": float(fn) * avg_fraud_amount,      # fraud $$ missed
            "cost_fp": float(fp) * 0.00005,               # investigation cost per FP
        })

    df_thresh = pd.DataFrame(rows)
    df_thresh["net_cost"] = df_thresh["cost_fn"] + df_thresh["cost_fp"]

    optimal_f1_idx    = df_thresh["f1"].idxmax()
    optimal_cost_idx  = df_thresh["net_cost"].idxmin()

    return {
        "data":               df_thresh.to_dict("records"),
        "optimal_f1_threshold":   float(df_thresh.loc[optimal_f1_idx, "threshold"]),
        "optimal_cost_threshold": float(df_thresh.loc[optimal_cost_idx, "threshold"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _cv_roc_auc(model, X, y, class_weight, n_splits=5, needs_sample_weight=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        if needs_sample_weight:
            sw = np.where(ytr == 1, class_weight[1], class_weight[0])
            model.fit(Xtr, ytr, sample_weight=sw)
        else:
            model.fit(Xtr, ytr)
        scores.append(roc_auc_score(yva, model.predict_proba(Xva)[:, 1]))
    return float(np.mean(scores)), float(np.std(scores))


# ══════════════════════════════════════════════════════════════════════════════
# EDA PRE-COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_eda(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["Is_Fraud"] = (df["Fraud_Label"] == "Fraud").astype(int)

    eda = {}

    # ── Overview stats ─────────────────────────────────────────────────────
    eda["total_transactions"] = len(df)
    eda["total_fraud"]        = int(df["Is_Fraud"].sum())
    eda["fraud_rate"]         = float(df["Is_Fraud"].mean())
    eda["total_amount"]       = float(df["Transaction_Amount (in Million)"].sum())
    eda["avg_fraud_amount"]   = float(
        df[df["Is_Fraud"] == 1]["Transaction_Amount (in Million)"].mean()
    )

    # ── Group-by breakdowns ────────────────────────────────────────────────
    def _gb(col):
        return (
            df.groupby(col)["Is_Fraud"]
            .agg(["sum", "mean", "count"])
            .rename(columns={"sum": "fraud_count", "mean": "fraud_rate", "count": "total"})
            .reset_index()
            .to_dict("records")
        )

    eda["fraud_by_type"]          = _gb("Transaction_Type")
    eda["fraud_by_merchant"]      = _gb("Merchant_Category")
    eda["fraud_by_card"]          = _gb("Card_Type")
    eda["fraud_by_location"]      = _gb("Transaction_Location")
    eda["fraud_by_international"] = _gb("Is_International_Transaction")
    eda["fraud_by_new_merchant"]  = _gb("Is_New_Merchant")
    eda["fraud_by_prev_fraud"]    = _gb("Previous_Fraud_Count")

    # ── Hour-level ─────────────────────────────────────────────────────────
    df["Hour"] = pd.to_datetime(
        df["Transaction_Time"], format="%H:%M", errors="coerce"
    ).dt.hour
    eda["fraud_by_hour"] = (
        df.groupby("Hour")["Is_Fraud"]
        .agg(["sum", "mean"])
        .rename(columns={"sum": "fraud_count", "mean": "fraud_rate"})
        .reset_index()
        .to_dict("records")
    )

    # ── Amount & distance distributions ───────────────────────────────────
    eda["amount_normal"] = df[df["Is_Fraud"] == 0]["Transaction_Amount (in Million)"].tolist()
    eda["amount_fraud"]  = df[df["Is_Fraud"] == 1]["Transaction_Amount (in Million)"].tolist()
    eda["distance_normal"] = df[df["Is_Fraud"] == 0]["Distance_From_Home"].dropna().tolist()
    eda["distance_fraud"]  = df[df["Is_Fraud"] == 1]["Distance_From_Home"].dropna().tolist()

    # ── Correlation matrix (numeric features only) ─────────────────────────
    numeric_cols = [
        "Transaction_Amount (in Million)", "Distance_From_Home",
        "Account_Balance (in Million)", "Daily_Transaction_Count",
        "Weekly_Transaction_Count", "Failed_Transaction_Count",
        "Previous_Fraud_Count", "Is_Fraud",
    ]
    corr_df = df[numeric_cols].corr()
    eda["correlation_matrix"] = {
        "columns": numeric_cols,
        "values":  corr_df.values.tolist(),
    }

    # ── Risk-flag combo analysis ───────────────────────────────────────────
    df["intl"] = (df["Is_International_Transaction"] == "Yes").astype(int)
    df["newm"] = (df["Is_New_Merchant"] == "Yes").astype(int)
    df["Combo"] = df["intl"].astype(str) + "_" + df["newm"].astype(str)
    combo_labels = {"0_0": "Neither", "1_0": "Intl Only", "0_1": "New Merch Only", "1_1": "Both"}
    df["Combo"] = df["Combo"].map(combo_labels)
    eda["fraud_by_combo"] = (
        df.groupby("Combo")["Is_Fraud"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fraud_rate", "count": "total"})
        .reset_index()
        .to_dict("records")
    )

    return eda


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(data_path: str, output_dir: str = "models"):
    print("📂  Loading data...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[TARGET])
    print(f"    {len(df):,} rows  |  Fraud rate: {(df[TARGET]=='Fraud').mean():.2%}")

    print("\n🔧  Preprocessing & feature engineering...")
    X, y, encoders = preprocess(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print(f"    Class weights → Normal: {weights[0]:.2f}  |  Fraud: {weights[1]:.2f}")

    # ── Model definitions ──────────────────────────────────────────────────
    model_defs = {
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=4,
                class_weight=class_weight, random_state=42, n_jobs=-1
            ),
            False,   # needs_sample_weight
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            True,    # GBM ignores class_weight → use sample_weight in .fit()
        ),
        "Logistic Regression": (
            LogisticRegression(
                class_weight=class_weight, max_iter=1000, random_state=42
            ),
            False,
        ),
    }

    print("\n🏋️   Training models + 5-fold cross-validation...")
    results = {}
    sample_weights_train = np.where(y_train == 1, weights[1], weights[0])

    for name, (model, needs_sw) in model_defs.items():
        # ── Final fit ──────────────────────────────────────────────────────
        if needs_sw:
            model.fit(X_train, y_train, sample_weight=sample_weights_train)
        else:
            model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # ── Cross-validation ───────────────────────────────────────────────
        cv_mean, cv_std = _cv_roc_auc(
            model.__class__(**model.get_params()),
            X_train, y_train, class_weight,
            needs_sample_weight=needs_sw,
        )

        results[name] = {
            "model":    model,
            "roc_auc":  roc_auc_score(y_test, y_prob),
            "pr_auc":   average_precision_score(y_test, y_prob),
            "brier":    brier_score_loss(y_test, y_prob),
            "cv_mean":  cv_mean,
            "cv_std":   cv_std,
            "report":   classification_report(y_test, y_pred, output_dict=True),
            "cm":       confusion_matrix(y_test, y_pred).tolist(),
            "y_prob":   y_prob.tolist(),
            "y_test":   y_test.tolist(),
        }
        print(
            f"    {name:<22}  "
            f"ROC-AUC={results[name]['roc_auc']:.4f}  "
            f"PR-AUC={results[name]['pr_auc']:.4f}  "
            f"CV={cv_mean:.4f}±{cv_std:.4f}"
        )

    # ── Select best model ──────────────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = results[best_name]["model"]
    print(f"\n🏆  Best model: {best_name}")

    # ── Calibration curve ─────────────────────────────────────────────────
    y_prob_best  = np.array(results[best_name]["y_prob"])
    y_test_arr   = np.array(results[best_name]["y_test"])
    prob_true, prob_pred = calibration_curve(y_test_arr, y_prob_best, n_bins=10)

    # ── Threshold analysis ─────────────────────────────────────────────────
    eda_stats = compute_eda(df)
    avg_fraud_amount = eda_stats["avg_fraud_amount"]
    thresh_analysis = compute_threshold_analysis(y_test_arr, y_prob_best, avg_fraud_amount)

    # ── Feature importance ─────────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature":    X.columns,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False)
    else:
        fi = pd.DataFrame({"feature": X.columns, "importance": np.zeros(len(X.columns))})

    # ── SHAP (optional) ───────────────────────────────────────────────────
    shap_data = None
    shap_explainer = None
    if SHAP_AVAILABLE:
        print("\n🔬  Computing SHAP values (sample of 500)…")
        sample_size = min(500, len(X_test))
        X_shap = X_test.sample(sample_size, random_state=42)

        try:
            if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier)):
                explainer = shap.TreeExplainer(best_model)
                sv = explainer.shap_values(X_shap)
                shap_vals_fraud = sv[1] if isinstance(sv, list) else sv
            else:
                explainer = shap.LinearExplainer(best_model, X_train)
                shap_vals_fraud = explainer.shap_values(X_shap)

            shap_data = {
                "shap_values":  shap_vals_fraud,   # np array (n_samples, n_features)
                "X_shap":       X_shap,
                "feature_names": list(X.columns),
                "mean_abs":      pd.Series(
                    np.abs(shap_vals_fraud).mean(axis=0),
                    index=X.columns,
                ).sort_values(ascending=False),
            }
            shap_explainer = explainer
            print("    SHAP done ✓")
        except Exception as e:
            print(f"    SHAP failed: {e}")

    # ── Bundle artifacts ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    artifacts = {
        "best_model":         best_model,
        "best_name":          best_name,
        "encoders":           encoders,
        "feature_names":      list(X.columns),
        "feature_importance": fi,
        "model_results":      {
            k: {kk: vv for kk, vv in v.items() if kk != "model"}
            for k, v in results.items()
        },
        "X_test":             X_test,
        "y_test":             y_test,
        "eda":                eda_stats,
        "calibration": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
        "threshold_analysis": thresh_analysis,
        "shap_data":          shap_data,
        "shap_explainer":     shap_explainer,
        "class_weight":       class_weight,
        "SHAP_AVAILABLE":     SHAP_AVAILABLE,
    }

    with open(os.path.join(output_dir, "artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n✅  Artifacts saved → {output_dir}/artifacts.pkl")
    return artifacts


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "FraudShield_Banking_Data.csv"
    train(data_path, output_dir="models")