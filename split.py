import pandas as pd
from sklearn.model_selection import train_test_split

CLEANED = "results/cleaned.csv"
TARGET = "readmitted_binary"
TEST_SIZE = 0.2
SEED = 42

df = pd.read_csv(CLEANED)

train, test = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df[TARGET],
    random_state=SEED,
)

train.to_csv("results/train.csv", index=False)
test.to_csv("results/test.csv", index=False)

for name, part in (("train", train), ("test", test)):
    pos = int(part[TARGET].sum())
    print(f"{name:<6} {len(part):>6,} rows  {pos:>5,} positive  ({100 * pos / len(part):5.2f}%)")

# stratification should hold the rate to within a fraction of a percentage point
assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.005