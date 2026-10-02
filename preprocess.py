"""Clean diabetic_data.csv into results/cleaned.csv, checking row counts after
every step against independently verified targets."""

import pandas as pd

DATA = "data/diabetic_data.csv"
OUT = "results/cleaned.csv"

# death or hospice: cannot be readmitted
NON_READMITTABLE = [11, 13, 14, 19, 20]

# one value in every row
ZERO_VARIANCE = ["examide", "citoglipton"]

# over 99.9% one value
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
    """Print and record rows and <30 count after a step."""
    pos = int((df["readmitted"] == "<30").sum())
    steps.append((label, len(df), pos, 100 * pos / len(df)))
    print(f"{label:<45} {len(df):>7,} rows  {pos:>6,} <30  ({100 * pos / len(df):5.2f}%)")


def group_icd9(code):
    """Bin an ICD-9 code into one of 9 groups (Strack et al., 2014)."""
    if pd.isna(code):
        return "Other"
    code = str(code).strip()
    if code.startswith(("V", "E")):  # not numeric, check before the float cast
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


# 1. load, treating '?' as missing
df = pd.read_csv(
    DATA,
    na_values="?",
    dtype={c: "string" for c in [*DIAG_COLS, "payer_code"]},
)
log("0. raw", df)

# 2. drop death/hospice before dedup
df = df[~df["discharge_disposition_id"].isin(NON_READMITTABLE)]
log("1. drop death/hospice discharges", df)

# 3. drop placeholder gender
df = df[df["gender"] != "Unknown/Invalid"]
log("2. drop invalid gender", df)

# 4. first encounter per patient (assumes encounter_id is chronological)
df = df.loc[df.groupby("patient_nbr")["encounter_id"].idxmin()]
log("3. dedup to first encounter per patient", df)

# 5. drop ids and unused columns
df = df.drop(columns=["encounter_id", "patient_nbr", "weight", *ZERO_VARIANCE, *NEAR_CONSTANT])

# 6. missing values become their own level
df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
df["payer_code"] = df["payer_code"].fillna("Unknown")
df["max_glu_serum"] = df["max_glu_serum"].fillna("Not tested")
df["A1Cresult"] = df["A1Cresult"].fillna("Not tested")

# 7. drop the few rows still missing race or a diagnosis
df = df.dropna(subset=["race", *DIAG_COLS])
log("4. drop missing race/diag_1/diag_2/diag_3", df)

# 8. ICD-9 codes to 9 groups
for col in DIAG_COLS:
    df[col] = df[col].map(group_icd9)

# 9. binary target: <30 vs everything else
df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
df = df.drop(columns=["readmitted"])

df.to_csv(OUT, index=False)

print(f"\nwrote {OUT}: {len(df):,} rows x {df.shape[1]} columns")
print(f"positive rate {100 * df['readmitted_binary'].mean():.2f}%")
print(f"majority-class baseline accuracy {100 * (1 - df['readmitted_binary'].mean()):.2f}%")

print("\ndiagnosis group counts (diag_1):")
print(df["diag_1"].value_counts().to_string())

# targets counted independently from the raw CSV
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
