"""
config.py
Central configuration: paths, random seed, dataset constants and display names.
Every script imports this module, so a single change here propagates everywhere and
all experiments stay reproducible.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repository root, so the code runs from any location)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "cicids2017"       # original CSVs (downloaded)
PROC_DIR = DATA_DIR / "processed"                # cleaned, analysis-ready CSVs

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
PRED_DIR = RESULTS_DIR / "preds"
MODELS_DIR = ROOT / "models"

for _d in (RAW_DIR, PROC_DIR, FIG_DIR, TAB_DIR, PRED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset files (CIC-IDS2017, "MachineLearningCVE" CSV release)
# ---------------------------------------------------------------------------
THURSDAY_FILE = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"   # web attacks
MONDAY_FILE = "Monday-WorkingHours.pcap_ISCX.csv"                          # 100% benign day

THURSDAY_ALL = PROC_DIR / "thursday_all.csv.gz"    # all traffic, cleaned (keeps Destination Port)
THURSDAY_HTTP = PROC_DIR / "thursday_http.csv.gz"  # HTTP (port 80) flows only -> main dataset
MONDAY_HTTP = PROC_DIR / "monday_http.csv.gz"      # unseen benign day, HTTP flows only
FEATURES_JSON = RESULTS_DIR / "feature_list.json"  # ordered model input features
PROFILE_JSON = RESULTS_DIR / "data_profile.json"   # row counts at every cleaning stage

# ---------------------------------------------------------------------------
# Experimental constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20      # held-out, previously unseen test split
CV_FOLDS = 5          # stratified k-fold cross-validation
HTTP_PORT = 80        # every web attack in CIC-IDS2017 targets the web server on port 80
TARGET_FPR = 0.01     # alarm budget (1% false alarms) used to calibrate the conventional
                      # detector and to report TPR@FPR for every detector

LABEL_COL = "Label"
BENIGN = "BENIGN"
ATTACKS = ["Brute Force", "XSS", "SQL Injection"]
CLASSES = [BENIGN] + ATTACKS

# ---------------------------------------------------------------------------
# Models (display names used in tables and figures)
# ---------------------------------------------------------------------------
ML_MODELS = ["logreg", "linsvm", "rf", "hgb", "mlp"]
BASELINE = "threshold_ids"
DISPLAY = {
    "logreg": "Logistic Regression",
    "linsvm": "Linear SVM",
    "rf": "Random Forest",
    "hgb": "Gradient-Boosted Trees (HistGB)",
    "mlp": "Neural Network (MLP)",
    "threshold_ids": "Threshold IDS (conventional)",
}
SHORT = {
    "logreg": "LogReg", "linsvm": "Lin-SVM", "rf": "RForest",
    "hgb": "HistGB", "mlp": "MLP", "threshold_ids": "Threshold IDS",
}
