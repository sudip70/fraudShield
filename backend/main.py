"""
FraudShield — FastAPI Backend  v4.1
=====================================
Endpoints:
  GET  /api/health
  GET  /api/version
  GET  /api/eda
  GET  /api/model
  POST /api/predict

Environment variables (set in Render dashboard or render.yaml):
  ARTIFACT_PATH   Path to model.pkl  (default: models/model.pkl relative to repo root)
  CORS_ORIGINS    Comma-separated allowed origins  (default: * — lock down in production)
  LOG_LEVEL       Python logging level  (default: INFO)

Changes in v4.1
---------------
- Pydantic version guard: raises a clear RuntimeError at import time if Pydantic v2
  is detected, rather than letting @validator decorators silently do nothing.
- opt_t safeguard: if the artifact's optimal F1 threshold is abnormally low (< 0.05),
  a minimum of 0.05 is enforced so that the MEDIUM tier remains reachable and
  composite scores are not inflated.
- medium_t is now derived from the clamped opt_t, preventing medium_t == high_t == 0.
- Production CORS warning: a startup log warning is emitted when CORS_ORIGINS is '*',
  making accidental open-CORS production deploys visible immediately in logs.
- Rule-points cap comment: documents that rule_points intentionally exceeds 40 before
  the min() cap; the cap is design behaviour, not a bug.
- _expected_international dead-code comment: the None branch is unreachable because
  tx_location and home_loc are validated against VALID_CITIES == CITY_COUNTRY.keys()
  before this function is called, but the guard is kept for defensive correctness.
"""

import logging
import os
import sys
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Pydantic version guard ─────────────────────────────────────────────────────
# @validator is a Pydantic v1 API.  In Pydantic v2, @validator is deprecated and
# silently ignored by default, meaning all input validation would stop working.
# We raise a clear error at import time rather than letting bad data reach the model.
try:
    import pydantic
    _pydantic_major = int(pydantic.VERSION.split(".")[0])
    if _pydantic_major >= 2:
        raise RuntimeError(
            f"Pydantic v2 ({pydantic.VERSION}) detected. This backend requires Pydantic v1 "
            "(e.g. pydantic>=1.10,<2). Install the correct version with:\n"
            "  pip install 'pydantic>=1.10,<2'\n"
            "Or migrate the @validator decorators to @field_validator (Pydantic v2 API)."
        )
    from pydantic import BaseModel, validator
except ImportError:
    raise RuntimeError("pydantic is not installed. Run: pip install 'pydantic>=1.10,<2'")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import preprocess, _shap_values_for_class1

# ── Environment variables ──────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

_cors_raw    = os.environ.get("CORS_ORIGINS", "*")
if _cors_raw.strip() in ("", "*"):
    CORS_ORIGINS = ["*"]
else:
    CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]
    if not CORS_ORIGINS:
        CORS_ORIGINS = ["*"]

ARTIFACT_PATH = os.environ.get(
    "ARTIFACT_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl"),
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fraudshield")

log.info("CORS_ORIGINS  = %s", CORS_ORIGINS)
log.info("ARTIFACT_PATH = %s", ARTIFACT_PATH)
log.info("LOG_LEVEL     = %s", LOG_LEVEL)

# FIX: Warn loudly when CORS is open to all origins so accidental production
# deploys are immediately visible in logs, even on Render's free tier.
if "*" in CORS_ORIGINS:
    log.warning(
        "CORS is open to ALL origins ('*'). This is fine for demos but should be "
        "restricted in production — set the CORS_ORIGINS env var to your frontend "
        "origin(s), e.g. 'https://yourusername.github.io'."
    )

# ── Load artifacts once at startup ────────────────────────────────────────────
def load_artifacts():
    path = os.path.abspath(ARTIFACT_PATH)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Model artifact not found at {path}.\n"
            "Run locally:  python src/pipeline.py data/FraudShield_Banking_Data.csv\n"
            "Then commit:  git add models/model.pkl && git commit -m 'add model artifact'"
        )
    try:
        with open(path, "rb") as f:
            arts = pickle.load(f)
    except ModuleNotFoundError as exc:
        missing_pkg = exc.name or "unknown"
        raise RuntimeError(
            f"Model artifact requires '{missing_pkg}' but it is not installed in this environment. "
            "Install backend dependencies with: pip install -r backend/requirements.txt"
        ) from exc
    log.info(
        "Artifacts loaded — best model: %s  ROC-AUC: %.4f  test_set_size: %d",
        arts["best_name"],
        arts["model_results"][arts["best_name"]]["roc_auc"],
        arts.get("test_set_size", 0),
    )
    return arts


arts = load_artifacts()

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FraudShield API",
    description="Fraud detection ML backend — EDA stats, model metrics, live scoring",
    version="4.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Utility ────────────────────────────────────────────────────────────────────
def clean_nan(val):
    """Recursively replace NaN / Inf / numpy scalars with JSON-safe equivalents."""
    if isinstance(val, dict):
        return {k: clean_nan(v) for k, v in val.items()}
    if isinstance(val, list):
        return [clean_nan(v) for v in val]
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return None if (np.isnan(val) or np.isinf(val)) else float(val)
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


# ── Tier ranking helper ────────────────────────────────────────────────────────
_TIER_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

def _max_tier(a: str, b: str) -> str:
    return a if _TIER_RANK[a] >= _TIER_RANK[b] else b


# ── Input domain constraints ──────────────────────────────────────────────────
VALID_TX_TYPES = {"Online", "ATM", "POS", "Wire Transfer"}
VALID_MERCHANT_CATS = {
    "Airlines", "ATM", "Clothing", "Electronics", "Entertainment", "Fuel",
    "Grocery", "Healthcare", "Hotel", "Jewelry", "Online Shopping",
    "Pharmacy", "Restaurant", "Travel",
}
VALID_CARD_TYPES = {"Debit", "Credit", "Prepaid"}
CITY_COUNTRY = {
    "New York": "US", "Los Angeles": "US", "Chicago": "US", "Houston": "US",
    "Phoenix": "US", "Philadelphia": "US", "San Antonio": "US", "San Diego": "US",
    "Dallas": "US", "San Francisco": "US", "Austin": "US", "Seattle": "US",
    "Denver": "US", "Nashville": "US", "Louisville": "US", "Portland": "US",
    "Las Vegas": "US", "Memphis": "US", "Atlanta": "US", "Miami": "US",
    "Boston": "US", "Washington DC": "US", "Detroit": "US", "Indianapolis": "US",
    "Columbus": "US", "Charlotte": "US", "Toronto": "CA", "Vancouver": "CA",
    "Montreal": "CA", "Calgary": "CA", "London": "GB", "Paris": "FR",
    "Berlin": "DE", "Madrid": "ES", "Rome": "IT", "Amsterdam": "NL",
    "Dublin": "IE", "Zurich": "CH", "Vienna": "AT", "Brussels": "BE",
    "Copenhagen": "DK", "Stockholm": "SE", "Oslo": "NO", "Helsinki": "FI",
    "Lisbon": "PT", "Istanbul": "TR", "Tokyo": "JP", "Singapore": "SG",
    "Mumbai": "IN", "São Paulo": "BR", "Buenos Aires": "AR",
}
VALID_CITIES = set(CITY_COUNTRY.keys())


def _expected_international(home_loc: str, tx_loc: str):
    """
    Derive the expected is_intl value from the two city names.

    NOTE: The None return branch is technically unreachable in practice because
    both home_loc and tx_loc are validated against VALID_CITIES (== CITY_COUNTRY.keys())
    before this function is called, so CITY_COUNTRY.get() will always find them.
    The None guard is kept for defensive correctness in case VALID_CITIES and
    CITY_COUNTRY drift out of sync in a future edit.
    """
    home_country = CITY_COUNTRY.get(home_loc)
    tx_country   = CITY_COUNTRY.get(tx_loc)
    if home_country is None or tx_country is None:
        return None
    return "Yes" if home_country != tx_country else "No"


# ── Optimal threshold safeguard ────────────────────────────────────────────────
# FIX: Clamp the artifact's optimal F1 threshold to a minimum of 0.05.
# If the threshold is 0 or near-zero (e.g. due to extreme class imbalance in a
# small test set), then medium_t == high_t == 0, making the MEDIUM tier unreachable
# and inflating composite scores — nearly every transaction would score HIGH.
# The minimum of 0.05 corresponds to the lowest step in compute_threshold_analysis.
_RAW_OPT_T = arts["threshold_analysis"]["optimal_f1_threshold"]
if _RAW_OPT_T < 0.05:
    log.warning(
        "Artifact optimal_f1_threshold is abnormally low (%.4f). "
        "Clamping to 0.05 to prevent MEDIUM tier becoming unreachable and "
        "composite scores being inflated. Retrain on a larger dataset to fix.",
        _RAW_OPT_T,
    )
_CLAMPED_OPT_T = max(_RAW_OPT_T, 0.05)


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    """
    Lightweight liveness check — safe to ping every 5 minutes from UptimeRobot
    to prevent Render free-tier spin-down.
    """
    best = arts["best_name"]
    return {
        "status":  "ok",
        "model":   best,
        "roc_auc": arts["model_results"][best]["roc_auc"],
        "shap":    arts.get("shap_data") is not None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VERSION
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/version")
def version():
    """
    Artifact provenance — when the model was trained, on what data, with what
    software versions. Supports model governance and UI freshness indicators.
    """
    meta = arts.get("training_metadata", {})
    return {
        "api_version":          "4.1.0",
        "pipeline_version":     meta.get("pipeline_version", "unknown"),
        "trained_at":           meta.get("trained_at"),
        "sklearn_version":      meta.get("sklearn_version"),
        "lgbm_version":         meta.get("lgbm_version"),
        "n_training_rows":      meta.get("n_rows"),
        "n_features":           meta.get("n_features"),
        "fraud_rate":           meta.get("fraud_rate"),
        "best_model":           meta.get("best_model", arts["best_name"]),
        "test_set_size":        arts.get("test_set_size"),
        "model_selection_metric": meta.get("model_selection_metric"),
        "optimal_f1_threshold": _CLAMPED_OPT_T,
        "raw_optimal_f1_threshold": _RAW_OPT_T,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/eda")
def eda():
    e    = arts["eda"]
    bins = np.linspace(0, 100000, 22).tolist()
    bcs  = ((np.array(bins[:-1]) + np.array(bins[1:])) / 2).tolist()

    def _hist(data):
        counts, _ = np.histogram(data, bins=bins, density=True)
        return {"x": bcs, "y": [clean_nan(float(c)) for c in counts.tolist()]}

    response = {
        "overview": {
            "total_transactions": e["total_transactions"],
            "total_fraud":        e["total_fraud"],
            "fraud_rate":         round(e["fraud_rate"], 6),
            "total_amount":       round(e["total_amount"], 2),
            "avg_fraud_amount":   round(e["avg_fraud_amount"], 4),
        },
        "fraud_by_type":          e["fraud_by_type"],
        "fraud_by_merchant":      e["fraud_by_merchant"],
        "fraud_by_location":      e["fraud_by_location"],
        "fraud_by_international": e["fraud_by_international"],
        "fraud_by_new_merchant":  e["fraud_by_new_merchant"],
        "fraud_by_prev_fraud":    e["fraud_by_prev_fraud"],
        "fraud_by_hour":          e["fraud_by_hour"],
        "fraud_by_combo":         e["fraud_by_combo"],
        "amount_dist": {
            "normal": _hist(e["amount_normal"]),
            "fraud":  _hist(e["amount_fraud"]),
        },
        "distance_dist": {
            "normal_median": clean_nan(float(np.median(e["distance_normal"]))),
            "fraud_median":  clean_nan(float(np.median(e["distance_fraud"]))),
            "normal_p75":    clean_nan(float(np.percentile(e["distance_normal"], 75))),
            "fraud_p75":     clean_nan(float(np.percentile(e["distance_fraud"], 75))),
        },
        "correlation": e.get("correlation_matrix", {}),
    }
    return clean_nan(response)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/model")
def model_info():
    from sklearn.metrics import roc_curve, precision_recall_curve

    results = arts["model_results"]
    best    = arts["best_name"]

    def _ds(arr, n=200):
        arr = np.asarray(arr)
        idx = np.round(np.linspace(0, len(arr) - 1, min(n, len(arr)))).astype(int)
        return arr[idx].tolist()

    curves = {}
    for name, r in results.items():
        y_t = np.array(r["y_test"])
        y_p = np.array(r["y_prob"])
        fpr, tpr, _ = roc_curve(y_t, y_p)
        pre, rec, _ = precision_recall_curve(y_t, y_p)
        curves[name] = {
            "roc": {"fpr": _ds(fpr), "tpr": _ds(tpr)},
            "pr":  {"precision": _ds(pre), "recall": _ds(rec)},
        }

    comparison = []
    for name, r in results.items():
        rep        = r["report"]
        fraud_rep  = rep.get("1", rep.get(1, {}))
        normal_rep = rep.get("0", rep.get(0, {}))
        comparison.append({
            "name":             name,
            "roc_auc":          round(r["roc_auc"],  4),
            "pr_auc":           round(r["pr_auc"],   4),
            "cv_mean":          round(r["cv_mean"],  4),
            "cv_std":           round(r["cv_std"],   4),
            "brier":            round(r["brier"],    4),
            "precision":        round(fraud_rep.get("precision", 0),  4),
            "recall":           round(fraud_rep.get("recall", 0),     4),
            "f1":               round(fraud_rep.get("f1-score", 0),   4),
            "precision_normal": round(normal_rep.get("precision", 0), 4),
            "recall_normal":    round(normal_rep.get("recall", 0),    4),
            "f1_normal":        round(normal_rep.get("f1-score", 0),  4),
            "is_best":          name == best,
        })

    fi_rows = [
        {"feature": row["feature"], "importance": round(float(row["importance"]), 6)}
        for _, row in arts["feature_importance"].head(15).iterrows()
    ]

    shap_global = []
    sd = arts.get("shap_data")
    if sd is not None:
        top_shap = sd["mean_abs"].head(15)
        shap_global = [
            {"feature": k, "value": round(float(v), 6)}
            for k, v in top_shap.items()
        ]

    cal    = arts.get("calibration", {})
    thresh = arts["threshold_analysis"]
    best_r = results[best]

    return {
        "best_name":                   best,
        "comparison":                  comparison,
        "curves":                      curves,
        "confusion_matrix":            best_r["cm"],
        "confusion_matrix_opt":        best_r.get("cm_opt", best_r["cm"]),
        "confusion_matrix_opt_thresh": best_r.get("opt_thresh", 0.5),
        "feature_importance":          fi_rows,
        "shap_global":                 shap_global,
        "calibration":                 cal,
        "threshold_analysis": {
            "optimal_f1_threshold":   _CLAMPED_OPT_T,
            "optimal_cost_threshold": thresh["optimal_cost_threshold"],
            "data": [
                {k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}
                for row in thresh["data"]
            ],
        },
        "fraud_rate":    arts["eda"]["fraud_rate"],
        "test_set_size": arts.get("test_set_size", len(best_r["y_test"])),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
class TransactionInput(BaseModel):
    amount:       float
    balance:      float
    distance:     float
    tx_time:      str
    tx_type:      str
    merchant_cat: str
    card_type:    str
    tx_location:  str
    home_loc:     str
    daily_tx:     int
    weekly_tx:    int
    avg_amount:   float
    max_24h:      float
    failed:       int
    prev_fraud:   int
    is_intl:      str
    is_new:       str
    unusual:      str

    @validator(
        "tx_type", "merchant_cat", "card_type", "tx_location", "home_loc",
        "is_intl", "is_new", "unusual",
        pre=True,
    )
    def strip_string_inputs(cls, v):
        return v.strip() if isinstance(v, str) else v

    @validator("tx_time")
    def validate_time_format(cls, v):
        try:
            datetime.strptime(v, "%H:%M")
        except ValueError:
            raise ValueError("tx_time must be HH:MM format, e.g. '14:30'")
        return v

    @validator("tx_type")
    def validate_tx_type(cls, v):
        if v not in VALID_TX_TYPES:
            raise ValueError(f"tx_type must be one of: {sorted(VALID_TX_TYPES)}")
        return v

    @validator("merchant_cat")
    def validate_merchant_category(cls, v):
        if v not in VALID_MERCHANT_CATS:
            raise ValueError(f"merchant_cat must be one of: {sorted(VALID_MERCHANT_CATS)}")
        return v

    @validator("card_type")
    def validate_card_type(cls, v):
        if v not in VALID_CARD_TYPES:
            raise ValueError(f"card_type must be one of: {sorted(VALID_CARD_TYPES)}")
        return v

    @validator("tx_location", "home_loc")
    def validate_location(cls, v):
        if v not in VALID_CITIES:
            raise ValueError("Location must be one of the supported city values")
        return v

    @validator("is_new", "unusual")
    def validate_yes_no(cls, v):
        if v not in ("Yes", "No"):
            raise ValueError("Must be 'Yes' or 'No'")
        return v

    @validator("is_intl")
    def validate_international_consistency(cls, v, values):
        if v not in ("Yes", "No"):
            raise ValueError("Must be 'Yes' or 'No'")
        expected = _expected_international(
            values.get("home_loc"),
            values.get("tx_location"),
        )
        # expected is None only if a city is missing from CITY_COUNTRY — see
        # _expected_international docstring. In practice this branch is unreachable
        # because both fields are already validated against VALID_CITIES above.
        if expected is not None and v != expected:
            raise ValueError(
                f"is_intl must match selected locations (expected '{expected}')"
            )
        return v

    @validator("amount", "balance", "avg_amount", "max_24h", "distance")
    def validate_non_negative_float(cls, v):
        if v < 0:
            raise ValueError("Must be >= 0")
        return v

    @validator("daily_tx", "weekly_tx")
    def validate_tx_counts(cls, v):
        if v < 1:
            raise ValueError("Transaction counts must be >= 1")
        return v

    @validator("failed", "prev_fraud")
    def validate_non_negative_int(cls, v):
        if v < 0:
            raise ValueError("Must be >= 0")
        return v


@app.post("/api/predict")
def predict(tx: TransactionInput):
    today = datetime.now().strftime("%Y-%m-%d")
    log.info(
        "predict  amount=%.2f  location=%s  intl=%s  new=%s  prev_fraud=%d  failed=%d",
        tx.amount, tx.tx_location, tx.is_intl, tx.is_new, tx.prev_fraud, tx.failed,
    )

    # NOTE: Dummy fields (Transaction_ID, IP_Address, Fraud_Label, etc.) are required
    # by engineer_features' column references but are not used as model features.
    # If engineer_features is ever updated to reference these fields as features,
    # replace these placeholders with real values from the request.
    row = pd.DataFrame([{
        "Transaction_Amount":           tx.amount,
        "Transaction_Time":             tx.tx_time,
        "Transaction_Date":             today,
        "Transaction_Type":             tx.tx_type,
        "Merchant_Category":            tx.merchant_cat,
        "Transaction_Location":         tx.tx_location,
        "Customer_Home_Location":       tx.home_loc,
        "Distance_From_Home":           tx.distance,
        "Card_Type":                    tx.card_type,
        "Account_Balance":              tx.balance,
        "Daily_Transaction_Count":      tx.daily_tx,
        "Weekly_Transaction_Count":     tx.weekly_tx,
        "Avg_Transaction_Amount":       tx.avg_amount,
        "Max_Transaction_Last_24h":     tx.max_24h,
        "Is_International_Transaction": tx.is_intl,
        "Is_New_Merchant":              tx.is_new,
        "Failed_Transaction_Count":     tx.failed,
        "Unusual_Time_Transaction":     tx.unusual,
        "Previous_Fraud_Count":         tx.prev_fraud,
        # Required by engineer_features but not used as model features.
        "Transaction_ID": 0, "Customer_ID": 0, "Merchant_ID": 0,
        "Device_ID": 0, "IP_Address": "0.0.0.0", "Fraud_Label": "Normal",
    }])

    try:
        X = preprocess(row, encoders=arts["encoders"], fit=False)
        X = X.reindex(columns=arts["feature_names"], fill_value=0)
        prob = float(arts["best_model"].predict_proba(X)[0][1])
    except Exception as e:
        log.error("Feature engineering / inference error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # ── ML-tier thresholds ────────────────────────────────────────────────
    # Use the clamped threshold (minimum 0.05) so medium_t is always strictly
    # less than high_t, keeping the MEDIUM tier reachable even when the artifact
    # was trained on a dataset with an extreme class imbalance that drove
    # optimal_f1_threshold close to zero.
    opt_t    = _CLAMPED_OPT_T
    medium_t = round(opt_t * 0.5, 4)

    if prob >= opt_t:
        ml_tier = "HIGH"
    elif prob >= medium_t:
        ml_tier = "MEDIUM"
    else:
        ml_tier = "LOW"

    rule_tier = ml_tier  # tracks the highest tier the rule engine wants to assign

    # ── Rule engine ───────────────────────────────────────────────────────
    rule_fired = False
    rule_name  = None

    critical_flags = sum([
        tx.prev_fraud > 0,
        tx.failed >= 2,
        tx.is_intl == "Yes",
        tx.is_new  == "Yes",
        tx.distance > 2000,
        tx.unusual == "Yes",
    ])

    if tx.prev_fraud > 0 and tx.failed >= 2 and tx.is_intl == "Yes":
        rule_fired = True
        rule_name  = "RULE_01: prior fraud + card-testing (≥2 failed) + international"
        rule_tier  = _max_tier(rule_tier, "HIGH")

    elif tx.distance > 5000 and tx.is_intl == "Yes" and tx.is_new == "Yes":
        rule_fired = True
        rule_name  = f"RULE_02: extreme displacement ({int(tx.distance):,} km) + international + new merchant"
        rule_tier  = _max_tier(rule_tier, "HIGH")

    elif critical_flags >= 5:
        rule_fired = True
        rule_name  = f"RULE_03: {critical_flags}/6 critical risk signals simultaneously active"
        rule_tier  = _max_tier(rule_tier, "HIGH")

    elif critical_flags >= 4 and ml_tier == "LOW":
        rule_fired = True
        rule_name  = f"RULE_04: {critical_flags}/6 critical risk signals — elevated for review"
        rule_tier  = _max_tier(rule_tier, "MEDIUM")

    # ── Composite risk score (0–100) ──────────────────────────────────────
    ml_component = min(60.0, (prob / max(opt_t, 1e-9)) * 60.0)

    # NOTE: Individual rule_points items are intentionally additive and can
    # sum beyond 40 when many signals fire together.  The min(40.0, ...) cap
    # is by design — it bounds the rule component to its allocated budget so the
    # ML component (0–60) always dominates when the model is confident.  The cap
    # discards the excess rather than compressing it, which is acceptable because
    # the tier assignment (HIGH/MEDIUM/LOW) is separate from the numeric score.
    rule_points = 0.0
    if tx.prev_fraud > 0:             rule_points += 12.0
    if tx.failed >= 2:                rule_points += 8.0
    if tx.is_intl == "Yes":           rule_points += 5.0
    if tx.is_new  == "Yes":           rule_points += 5.0
    if tx.distance > 2000:            rule_points += 5.0
    elif tx.distance > 500:           rule_points += 2.0
    if tx.unusual == "Yes":           rule_points += 3.0
    if tx.tx_location != tx.home_loc: rule_points += 2.0
    # Combination bonuses — non-additive risk uplift for co-occurring signals
    if tx.is_intl == "Yes" and tx.is_new == "Yes":  rule_points += 5.0
    if tx.prev_fraud > 0 and tx.is_intl == "Yes":   rule_points += 5.0
    if tx.prev_fraud > 0 and tx.failed >= 2:         rule_points += 5.0
    rule_component = min(40.0, rule_points)

    risk_score = round(ml_component + rule_component, 1)

    # ── Composite tier ────────────────────────────────────────────────────
    score_tier     = "HIGH" if risk_score >= 70 else "MEDIUM" if risk_score >= 35 else "LOW"
    composite_tier = _max_tier(score_tier, rule_tier)

    risk_score_pct = f"{risk_score:.1f}%"

    log.info(
        "scored  ml_prob=%.4f  ml_tier=%s  rule=%s  rule_tier=%s  "
        "score_tier=%s  composite_tier=%s  risk_score=%.1f",
        prob, ml_tier, rule_name or "none", rule_tier,
        score_tier, composite_tier, risk_score,
    )

    # ── Decision trace ────────────────────────────────────────────────────
    decision_trace = {
        "ml_probability":       round(prob, 6),
        "ml_tier":              ml_tier,
        "optimal_f1_threshold": opt_t,
        "medium_threshold":     medium_t,
        "rule_engine": {
            "fired":                 rule_fired,
            "rule_id":               rule_name,
            "tier_override":         rule_tier,
            "critical_flags_active": critical_flags,
        },
        "composite": {
            "ml_component":    round(ml_component, 2),
            "rule_component":  round(rule_component, 2),
            "total":           risk_score,
            "tier_thresholds": {"HIGH": 70, "MEDIUM": 35},
        },
        "final_tier": composite_tier,
    }

    # ── Display flags ─────────────────────────────────────────────────────
    flags = []
    if tx.is_intl == "Yes":
        flags.append({"icon": "🌍", "text": "International transaction"})
    if tx.is_new == "Yes":
        flags.append({"icon": "🏪", "text": "New merchant — no prior history"})
    if tx.unusual == "Yes":
        flags.append({"icon": "🕐", "text": "Unusual transaction time"})
    if tx.tx_location != tx.home_loc:
        flags.append({"icon": "📍", "text": f"Location mismatch: {tx.tx_location} vs home {tx.home_loc}"})
    if tx.distance > 400:
        flags.append({"icon": "📏", "text": f"{int(tx.distance):,} km from home"})
    if tx.failed > 0:
        flags.append({"icon": "❌", "text": f"{tx.failed} failed transaction(s) in session"})
    if tx.prev_fraud > 0:
        flags.append({"icon": "⚠️", "text": f"Prior fraud history: {tx.prev_fraud} incident(s)"})
    if tx.avg_amount > 0 and tx.amount > tx.avg_amount * 2:
        flags.append({
            "icon": "💰",
            "text": f"Amount spike: ${tx.amount:,.0f} vs avg ${tx.avg_amount:,.0f} ({tx.amount / tx.avg_amount:.1f}×)",
        })

    # ── SHAP waterfall ────────────────────────────────────────────────────
    shap_waterfall = []
    expl = arts.get("shap_explainer")
    if expl is not None:
        try:
            source_model = arts["best_model"]
            if hasattr(source_model, "named_steps"):
                X_shap = pd.DataFrame(
                    source_model[:-1].transform(X),
                    columns=arts["feature_names"],
                )
            else:
                X_shap = X

            sv      = expl.shap_values(X_shap)
            sv_arr  = _shap_values_for_class1(sv)
            sv_flat = sv_arr[0]

            series = pd.Series(sv_flat, index=arts["feature_names"])
            top    = pd.concat([series.nlargest(6), series.nsmallest(6)]).sort_values()
            shap_waterfall = [
                {"feature": k, "value": round(float(v), 5)}
                for k, v in top.items()
            ]
        except Exception as e:
            log.warning("SHAP waterfall failed (non-fatal): %s", e)

    # ── Response ──────────────────────────────────────────────────────────
    return {
        # Primary display fields
        "risk_score":         risk_score,
        "risk_score_pct":     risk_score_pct,
        "tier":               composite_tier,

        # Raw ML output
        "ml_probability":     round(prob, 6),
        "ml_probability_pct": f"{prob:.1%}",
        "ml_tier":            ml_tier,

        # Explainability
        "decision_trace":     decision_trace,
        "rule_override":      rule_name,
        "flags":              flags,
        "shap_waterfall":     shap_waterfall,

        # Metadata
        "model":              arts["best_name"],
        "roc_auc":            round(arts["model_results"][arts["best_name"]]["roc_auc"], 4),
        "optimal_threshold":  opt_t,
    }