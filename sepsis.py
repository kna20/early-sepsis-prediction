"""Shared code for the sepsis notebooks (physioNet/CinC 2019, Hospitals A and B).
Every notebook imports from here ,so loading, splitting, features, the clinical
utility score and cross-validation are written once and behave the same everywhere.
"""
import gc #garbage collector to help in dealing with limited storage
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# FEATURE GROUPS

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
CLINICAL = VITALS + LABS                  # the 34 measurements that change during a stay
ALL_FEATURES = CLINICAL + DEMOGRAPHICS    # all 40 inputs
UNIT_COLS = ["Unit1", "Unit2"]
CATEGORICAL = ["Gender"] + UNIT_COLS      # 0/1 codes: standardizing them would add nothing

HOURS_SINCE_CAP = 48   # "not measured for 48+ hours" and "never measured yet" look the same to the model
SEED = 42

# sepsis_dataset/ is expected next to this file, so the path works on any computer (and on GitHub)
LOCAL_DATA_DIR = Path(__file__).resolve().parent / "sepsis_dataset"


# LOADING

def data_dir():
    """Folder that contains training_setA/ and training_setB/, on Colab or on a laptop."""
    try:
        from google.colab import drive
    except ImportError:
        return LOCAL_DATA_DIR

    import zipfile
    drive.mount("/content/drive")
    folder = Path("/content/sepsis_dataset")

    # Unzip only the first time; later runs find the folders already there.
    if not (folder / "training_setB").exists():
        with zipfile.ZipFile("/content/drive/MyDrive/archive (1).zip") as z:
            members = [f for f in z.namelist()
                       if f.endswith(".psv") and f.startswith(("training_setA/", "training_setB/"))]
            z.extractall(folder, members)
    return folder


def load_hospital(name):
    """All patients of hospital "A" or "B" in ONE table: one row per patient-hour."""
    folder = data_dir() / f"training_set{name}"
    files = sorted(folder.rglob("*.psv"))
    assert files, f"No .psv files found in {folder} - is sepsis_dataset/ next to sepsis.py?"

    frames = []
    for file in files:
        df = pd.read_csv(file, sep="|")
        df["PatientID"] = file.stem             # p000001, ...
        df["Hour"] = np.arange(len(df))         # 0, 1, 2, ... within this stay
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # float32 halves the memory (the laptop ran out with float64); the precision loss does not matter here
    data[ALL_FEATURES] = data[ALL_FEATURES].astype("float32")
    return data

# Splitting: always by PATIENT, never by row

def patient_status(data):
    """1 if the patient is ever septic, else 0 (one value per patient)."""
    return data.groupby("PatientID")["SepsisLabel"].max()
    #This will be used to stratify and ensure that septic patients are represented in both the training and test split


def split_patients(data, test_size=0.2, ids=None):
    """Training and test patient IDs, with the same share of septic patients in both.

    ids=None splits all patients; pass ids to split only those again (e.g. 85% -> 70% + 15%).
    """
    status = patient_status(data)
    if ids is not None:
        status = status.loc[ids]
    # If one patient's hours landed in both parts, the model would be tested on someone it has already seen.
    train_ids, test_ids = train_test_split(status.index, test_size=test_size,random_state=SEED, stratify=status.values)
    return train_ids, test_ids

def make_folds(data, train_ids, n_folds=5):
    """For each development patient, the fold (1..n) in which it is the VALIDATION patient."""
    status = patient_status(data).loc[train_ids]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_of = pd.Series(0, index=status.index)
    for k, (_, val_idx) in enumerate(skf.split(status.index, status.values), start=1):
        fold_of.iloc[val_idx] = k
    assert (fold_of > 0).all()   # every patient is validated exactly once
    return fold_of


# Features computed once, before the folds
def prepare_dev(data, train_ids, fold_of=None, missingness=True):
    """Everything the model needs for the training patients, computed ONCE.
    fold_of is only needed for cross-validation (run_cv); a single split leaves it out.
    Forward fill and the missingness features only look at a patient's own past,
    and they learn nothing from other patients, so doing them before the fold
    split leaks nothing. (Medians and scaling DO learn, so they happen per fold.)
    """
    dev = (data[data["PatientID"].isin(train_ids)]
           .sort_values(["PatientID", "Hour"])          # ffill needs time order
           .reset_index(drop=True))
    patients = dev["PatientID"].to_numpy()
    hours = dev["Hour"].to_numpy()

    out = {
        "fold": None if fold_of is None else dev["PatientID"].map(fold_of).to_numpy(),
        "patients": patients,
        "hours": hours,
        "labels": dev["SepsisLabel"].to_numpy().astype(int),
        "missing": dev[ALL_FEATURES].isna(),    # raw gaps, for the 99%-missing rule
    }

    # Units are not carried forward: a missing unit becomes "unknown" (-1) later instead.
    carry = [c for c in ALL_FEATURES if c not in UNIT_COLS]
    filled = dev[ALL_FEATURES].copy()
    filled[carry] = filled[carry].groupby(patients).ffill()
    out["filled"] = filled

    if missingness:
        # After ffill a copied value looks like a fresh one. These two features give that information back.
        measured = dev[CLINICAL].notna()
        out["flags"] = measured.astype("int8").add_suffix("_measured")

        h = hours.astype("float32")
        last = pd.DataFrame(np.where(measured.to_numpy(), h[:, None], np.nan).astype("float32"))
        last = last.groupby(patients).ffill().to_numpy()               # hour of the latest real measurement
        since = np.nan_to_num(h[:, None] - last, nan=HOURS_SINCE_CAP)  #NaN = never measured so far
        out["since"] = pd.DataFrame(np.minimum(since, HOURS_SINCE_CAP),
                                    columns=[c + "_hours_since" for c in CLINICAL])
    return out

# Model: one train/validation split, and cross-validation built on it
def fit_predict(dev, train, val, cols, flags=False, since=False, class_weight="balanced"):
    """Train a logistic regression on the `train` rows and return P(sepsis) for the `val` rows.

    train and val are True/False masks over the rows of `dev`. Everything that learns
    from data (dropping, medians, scaling, the model) only ever sees the train rows.
    """
    # 1. Drop near-empty columns, judged on the training patients only.
    missing_pct = dev["missing"].loc[train, cols].mean() * 100
    keep = [c for c in cols if missing_pct[c] < 99]
    X_train = dev["filled"].loc[train, keep]
    X_val = dev["filled"].loc[val, keep]

    # 2. What ffill could not fill (no earlier value): training medians, and -1 = unknown ICU unit.
    fill = X_train.drop(columns=UNIT_COLS, errors="ignore").median().to_dict()
    fill.update({c: -1 for c in UNIT_COLS if c in keep})
    X_train, X_val = X_train.fillna(fill), X_val.fillna(fill)
    scale = [c for c in keep if c not in CATEGORICAL]

    # 3. Optional missingness features, only for the vitals/labs that survived step 1.
    kept = [c for c in CLINICAL if c in keep]
    extra = []
    if flags:
        extra.append(dev["flags"][[c + "_measured" for c in kept]])
    if since:
        since_cols = [c + "_hours_since" for c in kept]
        extra.append(dev["since"][since_cols])
        scale += since_cols                    # hours get scaled; 0/1 flags stay as they are
    if extra:
        X_train = pd.concat([X_train] + [e.loc[train] for e in extra], axis=1)
        X_val = pd.concat([X_val] + [e.loc[val] for e in extra], axis=1)
    assert not X_train.isna().any().any() and not X_val.isna().any().any()

    # 4. Mean and std come from the training patients, then the same numbers are applied to validation.
    scaler = StandardScaler().fit(X_train[scale])
    X_train[scale] = scaler.transform(X_train[scale])
    X_val[scale] = scaler.transform(X_val[scale])

    model = LogisticRegression(max_iter=1000, random_state=SEED, class_weight=class_weight)
    model.fit(X_train, dev["labels"][train])
    print(f"trained on {X_train.shape[1]} input features")
    return model.predict_proba(X_val)[:, 1]


def run_cv(dev, cols, **options):
    """Patient-level CV: fit_predict once per fold. Returns the out-of-fold predictions.

    options (flags, since, class_weight) are passed on to fit_predict unchanged.
    """
    oof = []
    for k in np.unique(dev["fold"]):
        val = dev["fold"] == k
        oof.append(pd.DataFrame({
            "PatientID": dev["patients"][val],
            "Hour": dev["hours"][val],
            "y_true": dev["labels"][val],
            "y_prob": fit_predict(dev, ~val, val, cols, **options),
            "Fold": k,
        }))
        gc.collect()      # free each fold's copies before the next one (the laptop ran out of memory without it)
    return pd.concat(oof, ignore_index=True)

# ----------------------------------------------------------------------------
# Clinical utility score (PhysioNet/CinC 2019 normalized utility)
#   1 = perfect alarm timing, 0 = same as never raising an alarm, <0 = worse than that
# Constants and formulas follow the official evaluation code:
# https://github.com/physionetchallenges/evaluation-2019
# ----------------------------------------------------------------------------
DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1, -2, -0.05, 0

M1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY)
B1 = -M1 * DT_EARLY
M2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL)
B2 = -M2 * DT_LATE
M3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL)
B3 = -M3 * DT_OPTIMAL


def make_utility_scorer(patient_ids, labels, hour_in_stay):
    """Return a function that scores 0/1 hourly predictions for exactly these hours."""
    patient_ids = np.asarray(patient_ids)
    labels = np.asarray(labels).astype(int)
    hour_in_stay = np.asarray(hour_in_stay)

    # First hour with SepsisLabel = 1 for each patient (infinity = never septic)
    first_positive = (pd.Series(np.where(labels == 1, hour_in_stay, np.inf))
                      .groupby(patient_ids).transform("min").to_numpy())
    is_septic = np.isfinite(first_positive)

    # The label switches on 6 h BEFORE onset, so onset = first positive + 6 h
    rel = hour_in_stay - (first_positive - DT_OPTIMAL)

    # What each hour is worth if we raise an alarm / stay quiet
    u_alarm = np.zeros(len(labels))
    u_quiet = np.zeros(len(labels))
    u_alarm[~is_septic] = U_FP
    u_quiet[~is_septic] = U_TN
    early = is_septic & (rel <= DT_OPTIMAL)
    late = is_septic & (rel > DT_OPTIMAL) & (rel <= DT_LATE)
    u_alarm[early] = np.maximum(M1 * rel[early] + B1, U_FP)
    u_alarm[late] = M2 * rel[late] + B2
    u_quiet[late] = M3 * rel[late] + B3

    inaction = u_quiet.sum()
    best_alarms = is_septic & (rel >= DT_EARLY) & (rel <= DT_LATE)
    best = np.where(best_alarms, u_alarm, u_quiet).sum()

    def score(predictions):
        observed = np.where(np.asarray(predictions) == 1, u_alarm, u_quiet).sum()
        return (observed - inaction) / (best - inaction)

    # The normalization only makes sense if these two hold
    assert abs(score(np.zeros(len(labels), dtype=int))) < 1e-12
    assert abs(score(best_alarms.astype(int)) - 1) < 1e-12
    return score


# ----------------------------------------------------------------------------
# Result tables
# ----------------------------------------------------------------------------
def threshold_metrics(y_true, pred):
    """F1, precision, recall and false positive rate of 0/1 predictions."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "F1": f1_score(y_true, pred, zero_division=0),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "FPR": fp / (fp + tn),
    }


def compare_auc(oofs):
    """AUROC/AUPRC of every feature set: (mean and std over folds, fold-by-fold difference to the FIRST set).

    The folds are the same for every set, so differences per fold remove the
    fold-to-fold noise that the plain means still contain.
    """
    rows = [{"Features": name, "Fold": k,
             "AUROC": roc_auc_score(g["y_true"], g["y_prob"]),
             "AUPRC": average_precision_score(g["y_true"], g["y_prob"])}
            for name, oof in oofs.items() for k, g in oof.groupby("Fold")]
    per_fold = pd.DataFrame(rows)
    summary = per_fold.groupby("Features", sort=False)[["AUROC", "AUPRC"]].agg(["mean", "std"])

    names = list(oofs)
    diffs = {}
    for metric in ["AUROC", "AUPRC"]:
        wide = per_fold.pivot(index="Fold", columns="Features", values=metric)[names]
        d = wide.sub(wide[names[0]], axis=0).drop(columns=names[0])
        diffs[metric] = pd.concat([d, d.mean().to_frame("mean").T, d.std().to_frame("std").T])
    return summary, pd.concat(diffs, axis=1)


CUTOFF_GRID = np.round(np.arange(0.10, 0.951, 0.05), 2)


def best_cutoff_table(oofs, grid=CUTOFF_GRID):
    """Each feature set at ITS OWN best cutoff (highest pooled clinical utility)."""
    rows = []
    for name, oof in oofs.items():
        y, prob = oof["y_true"].to_numpy(), oof["y_prob"].to_numpy()
        utility = make_utility_scorer(oof["PatientID"], y, oof["Hour"])
        best = max(grid, key=lambda c: utility((prob >= c).astype(int)))
        pred = (prob >= best).astype(int)

        # Same cutoff inside every fold: shows how much the utility moves between folds
        fold_u = [make_utility_scorer(g["PatientID"], g["y_true"], g["Hour"])((g["y_prob"].to_numpy() >= best).astype(int))
                  for _, g in oof.groupby("Fold")]

        rows.append({"Features": name, "Best cutoff": best, **threshold_metrics(y, pred),
                     "Clinical utility": utility(pred),
                     "Utility, fold mean": np.mean(fold_u),
                     "Utility, fold std": np.std(fold_u, ddof=1)})
    return pd.DataFrame(rows).set_index("Features")