# Pump it Up — Implementation Guide

Predict water pump status (`functional`, `functional needs repair`, `non functional`) for Tanzanian water points. Goal: a simple, interpretable pipeline built on well-established, "golden" tabular ML models — not an exotic stack.

Dataset: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/

---

## 0. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
```

Project layout:

```
ml_pump_it_up/
├── data/
│   ├── raw/                 # train_values.csv, train_labels.csv, test_values.csv
│   └── processed/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── models/                  # saved model artifacts
└── IMPLEMENTATION_GUIDE.md
```

---

## 1. Load the data

```python
import pandas as pd

values = pd.read_csv("data/raw/train_values.csv", index_col="id")
labels = pd.read_csv("data/raw/train_labels.csv", index_col="id")
df = values.join(labels)

test = pd.read_csv("data/raw/test_values.csv", index_col="id")
```

---

## 2. Quick EDA (sanity checks, not exhaustive)

```python
df["status_group"].value_counts(normalize=True)   # class balance
df.isna().mean().sort_values(ascending=False)      # missingness per column
df.nunique().sort_values(ascending=False)           # cardinality per column
```

What to look for:
- Class imbalance (`functional needs repair` is a small minority — keep this in mind for metrics/resampling).
- High-missingness columns (`scheme_name`, `funder`, `installer`, `public_meeting`, `permit`).
- High-cardinality columns (`wpt_name`, `subvillage`, `funder`, `installer`) — these need to be dropped or heavily grouped, not one-hot encoded as-is.

---

## 3. Feature selection

Use the reduced, interpretable feature set agreed on earlier — one representative per redundant hierarchy, no high-cardinality free text, no constant/ID-like columns.

```python
FEATURES = [
    "quantity_group",
    "amount_tsh",
    "extraction_type_class",
    "waterpoint_type_group",
    "construction_year",   # -> engineered into pump_age
    "date_recorded",       # -> used only to compute pump_age, then dropped
    "quality_group",
    "source_type",
    "management_group",
    "payment_type",
    "permit",
    "public_meeting",
    "basin",
    "gps_height",
    "population",
]

TARGET = "status_group"
```

---

## 4. Feature engineering

```python
def engineer_features(df):
    df = df.copy()

    # Pump age (construction_year has many 0s meaning "unknown" — treat as missing)
    df["date_recorded"] = pd.to_datetime(df["date_recorded"])
    year_recorded = df["date_recorded"].dt.year
    df["construction_year"] = df["construction_year"].replace(0, pd.NA)
    df["pump_age"] = year_recorded - df["construction_year"]

    # amount_tsh is heavily zero-inflated -> add a "was it reported" flag
    df["amount_tsh_reported"] = (df["amount_tsh"] > 0).astype(int)

    # booleans arrive as True/False/NaN -> make NaN an explicit category
    for col in ["permit", "public_meeting"]:
        df[col] = df[col].astype("object").fillna("unknown")

    return df.drop(columns=["date_recorded"])

df = engineer_features(df)
test = engineer_features(test)
```

---

## 5. Preprocessing pipeline

Keep it in an sklearn `Pipeline` / `ColumnTransformer` so train and test (and future data) are transformed identically, and so the model can be shipped as one artifact.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numeric_features = ["amount_tsh", "amount_tsh_reported", "pump_age", "gps_height", "population"]
categorical_features = [
    "quantity_group", "extraction_type_class", "waterpoint_type_group",
    "quality_group", "source_type", "management_group", "payment_type",
    "permit", "public_meeting", "basin",
]

numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])
```

> Note: tree models (Random Forest, XGBoost, LightGBM) don't need scaling — only imputation and encoding. That's part of why they're the "golden" choice here: less preprocessing, fewer things to get wrong.

---

## 6. Train/validation split

```python
from sklearn.model_selection import train_test_split

X = df[numeric_features + categorical_features]
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Stratify on `y` because of the class imbalance noted in EDA.

---

## 7. Model 1 (baseline): Random Forest

Why: minimal tuning needed, robust to messy tabular data, gives free feature importances for interpretability, and is a strong baseline on this exact competition historically.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )),
])

rf_pipeline.fit(X_train, y_train)
val_preds = rf_pipeline.predict(X_val)

print(accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds))
```

Competition metric is classification accuracy, so `accuracy_score` is the number to optimize — but check `classification_report` too, since the minority class (`functional needs repair`) is easy to ignore while still scoring well on accuracy.

---

## 8. Model 2 (stronger): Gradient Boosting (XGBoost / LightGBM)

Why: gradient-boosted trees are the other "golden" default for structured/tabular data and typically edge out Random Forest by a couple of points on this kind of dataset. Use whichever is already installed/familiar — they perform similarly here.

```python
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

xgb_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )),
])

xgb_pipeline.fit(X_train, y_train_enc)
val_preds_enc = xgb_pipeline.predict(X_val)

print(accuracy_score(y_val_enc, val_preds_enc))
```

Keep both models — compare them on the same validation split and pick the better one (or ensemble by averaging predicted probabilities, a simple and effective trick if the two models make different kinds of errors).

---

## 9. Cross-validation (more reliable than a single split)

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(scores.mean(), scores.std())
```

---

## 10. Light hyperparameter tuning

Don't over-invest here — a small randomized search around sane defaults is enough for a simple-but-effective solution.

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_leaf": [1, 2, 4],
}

search = RandomizedSearchCV(
    rf_pipeline, param_dist, n_iter=10, cv=3,
    scoring="accuracy", random_state=42, n_jobs=-1,
)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

---

## 11. Interpretability check

```python
import matplotlib.pyplot as plt

model = rf_pipeline.named_steps["model"]
feature_names = rf_pipeline.named_steps["preprocess"].get_feature_names_out()

importances = pd.Series(model.feature_importances_, index=feature_names)
importances.sort_values(ascending=False).head(15).plot.barh()
plt.gca().invert_yaxis()
plt.title("Top 15 feature importances")
plt.tight_layout()
plt.savefig("models/feature_importances.png")
```

Confirms whether `quantity_group`, `waterpoint_type_group`, and `pump_age` come out on top as expected from domain reasoning — if they don't, that's a signal to revisit preprocessing before trusting the model.

---

## 12. Generate predictions on the test set

```python
best_pipeline = search.best_estimator_   # or xgb_pipeline, whichever wins on CV

test_X = test[numeric_features + categorical_features]
test_preds = best_pipeline.predict(test_X)

submission = pd.DataFrame({"id": test.index, "status_group": test_preds})
submission.to_csv("data/processed/submission.csv", index=False)
```

---

## 13. Next steps (only if the simple pipeline underperforms)

- Add back `funder`/`installer` after grouping rare categories into `"other"`.
- Target-encode high-cardinality geography (`ward`, `lga`) instead of dropping them.
- Try a stacked ensemble of Random Forest + XGBoost/LightGBM.
- Address class imbalance more directly (e.g. `class_weight`, or oversampling the minority class) if the `functional needs repair` recall is the weak point.

---

## Summary

| Step | Choice | Why |
|---|---|---|
| Models | Random Forest → XGBoost/LightGBM | Industry-standard defaults for tabular data; minimal preprocessing, strong out-of-the-box performance |
| Preprocessing | Median impute (numeric), constant-impute + one-hot (categorical) | Trees don't need scaling; keeps pipeline simple |
| Validation | Stratified 80/20 split + 5-fold CV | Class imbalance makes stratification necessary |
| Metric | Accuracy (competition metric) + per-class report | Accuracy alone can hide poor minority-class performance |
| Tuning | Small randomized search | Diminishing returns beyond this for a "simple but effective" solution |
