import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("diabetes.csv")

# Kolom berikut memiliki missing value berupa nilai 0
missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df[missing_cols] = df[missing_cols].replace(0, np.nan)

for col in missing_cols:
    df[col] = df[col].fillna(df[col].mean())


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

smote = SMOTE(k_neighbors=3, random_state=2024, n_jobs=-1)
X_over, y_over = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_over, y_over, test_size=0.2, random_state=2024
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "knn": KNeighborsClassifier(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC(),
}


model_results = pd.DataFrame(
    index=models.keys(), columns=["train", "validation", "test"]
)
for name, model in models.items():
    results = cross_validate(
        model,
        X_train,
        y_train,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    model.fit(X_train, y_train)

    model_results.loc[name, "train"] = np.mean(results["train_score"])
    model_results.loc[name, "validation"] = np.mean(results["test_score"])
    model_results.loc[name, "test"] = f1_score(y_test, model.predict(X_test))

print(model_results)

param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [16, 32],
    "min_samples_split": [2, 5],
    "ccp_alpha": [0.001, 0.005, 0.007],
}

baseline = RandomForestClassifier(random_state=2024, n_jobs=-1)
grid = GridSearchCV(
    baseline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="f1",
    verbose=1,
    return_train_score=True,
)

grid.fit(X_train, y_train)

best_estimator = grid.best_estimator_
best_estimator.fit(X_train, y_train)

y_pred = best_estimator.predict(X_test)
final_result = pd.DataFrame(
    {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    },
    index=["RandomForest"],
)

print(final_result.round(2))
