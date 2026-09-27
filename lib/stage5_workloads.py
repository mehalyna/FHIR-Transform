"""
Frozen W1-W4 workload query logic, extracted verbatim from
notebooks/04_analytical_fidelity.ipynb, plus thin CSV loaders for the persistent R1/R2
representations (results/stage3/*.csv). Every query function below takes only the already-loaded
representation data it needs and returns the workload's result set (a set of Patient IDs for
W1-W3, a set of Encounter IDs for W4) -- exactly the frozen Stage 4 definitions, unmodified.

There is deliberately no w*_r2 function for W4: R2 lacks Encounter.period.end (Stage 3/4 finding),
so W4/R2 is not feasible and must never be approximated.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from lib.stage5_transform import ref_id

SNOMED = "http://snomed.info/sct"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
BRONCHITIS_CODE = "10509002"
BRONCHITIS_DISPLAY = "Acute bronchitis (disorder)"
# Frozen Stage 4 W2 medication selection (results/stage4/selected_w2_medication.json) -- not
# rederived here; Stage 4 froze it as an R0-only, pre-comparison selection.
SELECTED_MED_SYSTEM = RXNORM
SELECTED_MED_CODE = "106892"
THRESHOLD_HOURS = 24.0


def read_csv_str(path: Path) -> pd.DataFrame:
    """Same frozen loader NB04 uses for R1/R2 CSVs: dtype=str so concept_id joins are exact."""
    return pd.read_csv(path, dtype=str, keep_default_na=True, na_values=["", "nan", "NaN"])


def load_r1_tables_from_csv(stage3_dir: Path, names):
    return {name: read_csv_str(stage3_dir / f"{name}.csv") for name in names}


def load_r2_tables_from_csv(stage3_dir: Path, names):
    return {name: read_csv_str(stage3_dir / f"{name}.csv") for name in names}


def norm_id(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    x = str(x)
    if x == "" or x.lower() == "nan":
        return None
    return x[:-2] if x.endswith(".0") else x


def matching_concept_ids(r1_coding_df, system, code):
    """concept_id set for coding rows matching (system, code).

    This is a vectorized reformulation of the frozen NB04 lookup (build a full
    concept_id -> {(system, code), ...} reverse index over every row of r1_coding_df, then keep
    concepts whose set contains the target pair). Both formulations join through the same
    normalized coding table and return the identical set of matching concept_ids; this version
    filters directly to the rows that match instead of materializing an index over the ~163k
    coding rows that describe every other field (Encounter.class, Observation components, etc.),
    which is unrelated to the single (system, code) pair a given workload is resolving.
    """
    mask = (r1_coding_df["system"] == system) & (r1_coding_df["code"] == code)
    return set(r1_coding_df.loc[mask, "concept_id"].map(norm_id))


# ---------------------------------------------------------------------------
# W1: single-concept clinical cohort (Acute bronchitis, SNOMED 10509002)
# ---------------------------------------------------------------------------

def w1_r0(resources_by_target_type):
    patients = set()
    for c in resources_by_target_type["Condition"]:
        codings = (c.get("code") or {}).get("coding", []) or []
        if any(co.get("system") == SNOMED and co.get("code") == BRONCHITIS_CODE for co in codings):
            pid = ref_id(c.get("subject"))
            if pid:
                patients.add(pid)
    return patients


def w1_r1(r1_condition_df, r1_coding_df):
    matched_concepts = matching_concept_ids(r1_coding_df, SNOMED, BRONCHITIS_CODE)
    mask = r1_condition_df["code_concept_id"].map(norm_id).isin(matched_concepts)
    return set(r1_condition_df.loc[mask, "patient_id"].dropna())


def w1_r2(r2_condition_df):
    mask = (r2_condition_df["code_system"] == SNOMED) & (r2_condition_df["code_code"] == BRONCHITIS_CODE)
    return set(r2_condition_df.loc[mask, "patient_id"].dropna())


# ---------------------------------------------------------------------------
# W2: cross-resource cohort (W1 bronchitis patients also on the frozen RxNorm 106892 medication)
# ---------------------------------------------------------------------------

def w2_r0(resources_by_target_type):
    s0_w1 = w1_r0(resources_by_target_type)
    patients = set()
    for m in resources_by_target_type["MedicationRequest"]:
        codings = (m.get("medicationCodeableConcept") or {}).get("coding", []) or []
        if codings and (codings[0].get("system"), codings[0].get("code")) == (SELECTED_MED_SYSTEM, SELECTED_MED_CODE):
            pid = ref_id(m.get("subject"))
            if pid:
                patients.add(pid)
    return s0_w1 & patients


def w2_r1(r1_condition_df, r1_coding_df, r1_medication_request_df):
    s1_w1 = w1_r1(r1_condition_df, r1_coding_df)
    matched_concepts = matching_concept_ids(r1_coding_df, SELECTED_MED_SYSTEM, SELECTED_MED_CODE)
    mask = r1_medication_request_df["medication_codeable_concept_id"].map(norm_id).isin(matched_concepts)
    return s1_w1 & set(r1_medication_request_df.loc[mask, "patient_id"].dropna())


def w2_r2(r2_condition_df, r2_medication_request_df):
    s2_w1 = w1_r2(r2_condition_df)
    mask = (r2_medication_request_df["medication_system"] == SELECTED_MED_SYSTEM) & (r2_medication_request_df["medication_code"] == SELECTED_MED_CODE)
    return s2_w1 & set(r2_medication_request_df.loc[mask, "patient_id"].dropna())


# ---------------------------------------------------------------------------
# W3: MedicationRequest.reasonReference -> Condition temporal query
# ---------------------------------------------------------------------------

def add_satisfaction(df):
    df = df.copy()
    df["authored_on_dt"] = pd.to_datetime(df["authored_on"], utc=True, errors="coerce")
    df["onset_datetime_dt"] = pd.to_datetime(df["onset_datetime"], utc=True, errors="coerce")
    df["missing_timestamp"] = df["authored_on_dt"].isna() | df["onset_datetime_dt"].isna()
    df["satisfies"] = df["resolved"] & (~df["missing_timestamp"]) & (df["authored_on_dt"] >= df["onset_datetime_dt"])
    return df


def w3_generic(medreq_df, cond_df, authored_col, onset_col, reason_col, cond_id_col, cond_onset_col, patient_col):
    sub = medreq_df[medreq_df[reason_col].notna()][[reason_col, authored_col, patient_col]].copy()
    sub = sub.rename(columns={reason_col: "condition_id", authored_col: "authored_on", patient_col: "patient_id"})
    cond_lookup = cond_df.set_index(cond_id_col)[cond_onset_col].to_dict()
    sub["resolved"] = sub["condition_id"].isin(cond_lookup.keys())
    sub["onset_datetime"] = sub["condition_id"].map(cond_lookup)
    return sub[["patient_id", "resolved", "authored_on", "onset_datetime"]]


def w3_r0(resources_by_target_type):
    condition_by_id_r0 = {c["id"]: c for c in resources_by_target_type["Condition"]}
    rows = []
    for m in resources_by_target_type["MedicationRequest"]:
        rr = m.get("reasonReference")
        if not rr:
            continue
        target_id = ref_id(rr[0])
        cond = condition_by_id_r0.get(target_id)
        rows.append({
            "patient_id": ref_id(m.get("subject")), "resolved": cond is not None,
            "authored_on": m.get("authoredOn"), "onset_datetime": cond.get("onsetDateTime") if cond is not None else None,
        })
    df = add_satisfaction(pd.DataFrame(rows))
    return set(df.loc[df["satisfies"], "patient_id"].dropna())


def w3_r1(r1_medication_request_df, r1_condition_df):
    df = add_satisfaction(w3_generic(
        r1_medication_request_df, r1_condition_df, "authored_on", "onset_datetime",
        "reason_condition_id", "condition_id", "onset_datetime", "patient_id",
    ))
    return set(df.loc[df["satisfies"], "patient_id"].dropna())


def w3_r2(r2_medication_request_df, r2_condition_df):
    df = add_satisfaction(w3_generic(
        r2_medication_request_df, r2_condition_df, "event_timestamp", "event_timestamp",
        "reason_condition_id", "condition_id", "event_timestamp", "patient_id",
    ))
    return set(df.loc[df["satisfies"], "patient_id"].dropna())


# ---------------------------------------------------------------------------
# W4: Encounter duration > 24h. No w4_r2 -- not feasible (Encounter.period.end not in R2).
# ---------------------------------------------------------------------------

def w4_r0(resources_by_target_type):
    rows = [{"encounter_id": e["id"], "start": (e.get("period") or {}).get("start"), "end": (e.get("period") or {}).get("end")}
            for e in resources_by_target_type["Encounter"]]
    df = pd.DataFrame(rows)
    df["start_dt"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
    df["duration_hours"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 3600.0
    return set(df.loc[df["duration_hours"] > THRESHOLD_HOURS, "encounter_id"])


def w4_r1(r1_encounter_df):
    df = r1_encounter_df.copy()
    df["start_dt"] = pd.to_datetime(df["period_start"], utc=True, errors="coerce")
    df["end_dt"] = pd.to_datetime(df["period_end"], utc=True, errors="coerce")
    df["duration_hours"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 3600.0
    return set(df.loc[df["duration_hours"] > THRESHOLD_HOURS, "encounter_id"])


# ---------------------------------------------------------------------------
# Per-workload/representation minimum required persistent files (Section 7 documentation).
# ---------------------------------------------------------------------------

REQUIRED_R0_TYPES = {
    "W1": ["Condition"],
    "W2": ["Condition", "MedicationRequest"],
    "W3": ["Condition", "MedicationRequest"],
    "W4": ["Encounter"],
}

REQUIRED_R1_TABLES = {
    "W1": ["r1_condition", "r1_coding"],
    "W2": ["r1_condition", "r1_coding", "r1_medication_request"],
    "W3": ["r1_medication_request", "r1_condition"],
    "W4": ["r1_encounter"],
}

REQUIRED_R2_TABLES = {
    "W1": ["r2_condition"],
    "W2": ["r2_condition", "r2_medication_request"],
    "W3": ["r2_medication_request", "r2_condition"],
    # W4/R2: not feasible -- no table set.
}
