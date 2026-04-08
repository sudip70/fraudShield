"""
scripts/check_model.py
======================
Check whether the trained model artifact exists.
If it does not, train the pipeline to produce it.

This script is called by render.yaml's buildCommand so the inline Python block
in the YAML is replaced by a proper, readable, testable file.

Usage:
    python scripts/check_model.py

Environment variables:
    ARTIFACT_PATH   Override the default artifact location.
                    Default: models/model.pkl (relative to repo root)
    DATA_PATH       Override the default training data path.
                    Default: data/FraudShield_Banking_Data.csv
"""

import os
import subprocess
import sys

# Resolve paths relative to the repo root (parent of this script's directory).
_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PKL  = os.path.join(_REPO_ROOT, "models", "model.pkl")
_DEFAULT_DATA = os.path.join(_REPO_ROOT, "data", "FraudShield_Banking_Data.csv")

ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH", _DEFAULT_PKL)
DATA_PATH     = os.environ.get("DATA_PATH",     _DEFAULT_DATA)


def main() -> int:
    if os.path.exists(ARTIFACT_PATH):
        print(f"✅ Model artifact found at '{ARTIFACT_PATH}' — skipping training.")
        return 0

    print(f"⚙️  Model artifact not found at '{ARTIFACT_PATH}' — training now…")

    if not os.path.exists(DATA_PATH):
        print(
            f"❌ Training data not found at '{DATA_PATH}'.\n"
            "   Either commit the CSV to your repo or set the DATA_PATH env var.",
            file=sys.stderr,
        )
        return 1

    pipeline_script = os.path.join(_REPO_ROOT, "src", "pipeline.py")
    result = subprocess.run(
        [sys.executable, pipeline_script, DATA_PATH],
        cwd=_REPO_ROOT,
    )

    if result.returncode != 0:
        print(
            f"❌ Training failed with exit code {result.returncode}.",
            file=sys.stderr,
        )
    else:
        print(f"✅ Training complete — artifact saved to '{ARTIFACT_PATH}'.")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())