import pandas as pd
import joblib
import matplotlib.pyplot as plt

CSV_FILE = "placement.csv"
TARGET = "PlacementStatus"
DROP_COLUMNS = ["StudentID"]
SOURCE_NAME = "Placement Prediction Dataset (Kaggle, Ruchika Kumbhar)"
SOURCE_URL = "https://www.kaggle.com/datasets/ruchikakumbhar/placement-prediction-dataset"

df = pd.read_csv(CSV_FILE)
print("Rows, columns:", df.shape)

df = df.drop(columns=DROP_COLUMNS)
df = df.drop_duplicates()
df = df.dropna(subset=[TARGET])

y = df[TARGET].astype(str).str.strip().str.lower().isin(["placed", "1", "yes"]).astype(int)
X = df.drop(columns=[TARGET])
X = X.loc[:, X.nunique() > 1]
print("Placed %:", round(y.mean() * 100, 1))
print("Features:", list(X.columns))

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

number_cols = X.select_dtypes(include="number").columns.tolist()
text_cols = [c for c in X.columns if c not in number_cols]
print("Number columns:", number_cols)
print("Text columns:", text_cols)

prep = ColumnTransformer([
    ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), number_cols),
    ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), text_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", len(X_train), "Test:", len(X_test))

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15),
}

results = []
for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    score = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
    results.append({"Model": name, "ROC-AUC": round(score, 3)})
    print(name, "->", round(score, 3))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
best_name = results_df.iloc[0]["Model"]
print("Best model:", best_name)

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.inspection import permutation_importance

best_model = Pipeline([("prep", prep), ("model", models[best_name])])
best_model.fit(X_train, y_train)

pred = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1]
metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
    "ROC-AUC": roc_auc_score(y_test, proba),
}
for k, v in metrics.items():
    print(k, ":", round(v, 3))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))

imp = permutation_importance(best_model, X_test, y_test, scoring="roc_auc",
                             n_repeats=5, random_state=42)
importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
importance.plot(kind="barh", title="Feature importance")
plt.tight_layout()
plt.savefig("feature_importance.png")

info = {}
for c in X.columns:
    if c in number_cols:
        whole = bool((X[c].dropna() % 1 == 0).all())
        info[c] = {"type": "number", "min": float(X[c].min()), "max": float(X[c].max()),
                   "default": float(X[c].median()), "step": 1.0 if whole else 0.1}
    else:
        info[c] = {"type": "text", "options": sorted(X[c].dropna().astype(str).unique().tolist())}

joblib.dump({
    "model": best_model, "columns": list(X.columns), "info": info,
    "metrics": metrics, "results": results_df, "best_name": best_name,
    "importance": importance, "y_test": y_test.values, "test_proba": proba, "source_name": SOURCE_NAME, "source_url": SOURCE_URL,
}, "placement_model.joblib")
print("Model saved: placement_model.joblib")

from sklearn.model_selection import learning_curve

lc_pipe = Pipeline([("prep", prep), ("model", models[best_name])])
sizes, train_scores, val_scores = learning_curve(
    lc_pipe, X_train, y_train, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
    cv=5, scoring="roc_auc", shuffle=True, random_state=42)
joblib.dump({
    "sizes": sizes,
    "train": train_scores.mean(axis=1),
    "val": val_scores.mean(axis=1),
}, "learning_curve.joblib")
print("Learning curve saved: learning_curve.joblib")
print("Final train score:", round(train_scores.mean(axis=1)[-1], 3))
print("Final validation score:", round(val_scores.mean(axis=1)[-1], 3))

err_df = X_test.copy()
err_df["Actual"] = y_test.values
err_df["Probability"] = proba
joblib.dump(err_df, "error_analysis.joblib")
print("Error analysis saved: error_analysis.joblib")




