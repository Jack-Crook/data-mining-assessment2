"""Pre-processing pipeline for the diabetes 130-US-hospitals dataset (3804ICT A2).

Steps run in the order fixed in the implementation plan: row exclusions before the
first-encounter dedup, so that a patient whose earliest encounter is a hospice
discharge is represented by their next surviving encounter rather than dropped.

Row counts are logged after every step; they are checked against known targets at
the end. A mismatch means a bug in the step above it, not in the target.
"""

import pandas as pd

DATA = "data/diabetic_data.csv"
OUT = "results/cleaned.csv"

# discharge dispositions meaning death or hospice: these patients cannot be readmitted
NON_READMITTABLE = [11, 13, 14, 19, 20]

# no variance at all (a single distinct value across all 101,766 rows)
ZERO_VARIANCE = ["examide", "citoglipton"]

# >99.9% one value; carry no usable signal but do let a tree split on noise
NEAR_CONSTANT = [
    "acetohexamide",
    "glimepiride-pioglitazone",
    "metformin-pioglitazone",
    "metformin-rosiglitazone",
    "troglitazone",
    "glipizide-metformin",
    "tolbutamide",
    "miglitol",
    "tolazamide",
    "chlorpropamide",
]

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

steps = []


def log(label, df):
    """Record and print rows + positives after a pipeline step."""
    pos = int((df["readmitted"] == "<30").sum())
    steps.append((label, len(df), pos, 100 * pos / len(df)))
    print(f"{label:<45} {len(df):>7,} rows  {pos:>6,} <30  ({100 * pos / len(df):5.2f}%)")


def group_icd9(code):
    """Bin an ICD-9 code into one of 9 diagnosis groups (Strack et al., 2014).

    V (supplementary) and E (external cause) codes are non-numeric and must be
    branched on before any numeric comparison, or the float cast throws.
    """
    if pd.isna(code):
        return "Other"
    code = str(code).strip()
    if code.startswith(("V", "E")):
        return "Other"
    try:
        value = float(code)
    except ValueError:
        return "Other"
    if 250 <= value < 251:
        return "Diabetes"
    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 140 <= value <= 239:
        return "Neoplasms"
    if 800 <= value <= 999:
        return "Injury"
    return "Other"


# 1. load, collapsing '?' and real NaN into one representation
df = pd.read_csv(
    DATA,
    na_values="?",
    dtype={c: "string" for c in [*DIAG_COLS, "payer_code"]},
)
log("0. raw", df)

# 2. exclusions before dedup (see module docstring)
df = df[~df["discharge_disposition_id"].isin(NON_READMITTABLE)]
log("1. drop death/hospice discharges", df)

# 3. three records carry a placeholder gender
df = df[df["gender"] != "Unknown/Invalid"]
log("2. drop invalid gender", df)

# 4. one row per patient. No timestamp exists; encounter_id is assumed to increase
#    chronologically, so the minimum per patient is their first encounter.
df = df.loc[df.groupby("patient_nbr")["encounter_id"].idxmin()]
log("3. dedup to first encounter per patient", df)

# 5. identifiers were needed for the dedup, not as features
df = df.drop(columns=["encounter_id", "patient_nbr", "weight", *ZERO_VARIANCE, *NEAR_CONSTANT])

# 6. missingness that is itself meaningful becomes an explicit level
df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
df["payer_code"] = df["payer_code"].fillna("Unknown")
df["max_glu_serum"] = df["max_glu_serum"].fillna("Not tested")
df["A1Cresult"] = df["A1Cresult"].fillna("Not tested")

# 7. remaining missingness is small enough to drop rather than impute
df = df.dropna(subset=["race", *DIAG_COLS])
log("4. drop missing race/diag_1/diag_2/diag_3", df)

# 8. ~700-800 distinct codes per column down to 9 groups
for col in DIAG_COLS:
    df[col] = df[col].map(group_icd9)

# 9. binary target: readmission within 30 days vs. everything else
df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
df = df.drop(columns=["readmitted"])

df.to_csv(OUT, index=False)

print(f"\nwrote {OUT}: {len(df):,} rows x {df.shape[1]} columns")
print(f"positive rate {100 * df['readmitted_binary'].mean():.2f}%")
print(f"majority-class baseline accuracy {100 * (1 - df['readmitted_binary'].mean()):.2f}%")

print("\ndiagnosis group counts (diag_1):")
print(df["diag_1"].value_counts().to_string())

# verified independently from the raw CSV; a mismatch is a bug in this script
TARGETS = [
    ("0. raw", 101_766, 11_357),
    ("1. drop death/hospice discharges", 99_343, 11_314),
    ("2. drop invalid gender", 99_340, 11_314),
    ("3. dedup to first encounter per patient", 69_987, 6_285),
    ("4. drop missing race/diag_1/diag_2/diag_3", 66_859, 6_082),
]

print("\ncheck against verified targets:")
for (label, rows, pos, _), (_, want_rows, want_pos) in zip(steps, TARGETS):
    ok = rows == want_rows and pos == want_pos
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<45} "
          f"got {rows:,}/{pos:,}  want {want_rows:,}/{want_pos:,}")
