import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import  train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
# Load the dataset
df = pd.read_csv("student.csv")

# Separate features and target
X = df[["studyhour", "attendance"]]
y = df["result"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
numeric_features = ["studyhour", "attendance"]

numeric_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_preprocessor, numeric_features)
    ]
)
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        )
    ]
)


# 1. Create 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 2. Define settings to test
param_grid = {
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__class_weight": [None, "balanced"]
}

# 3. Search for the best settings
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True
)

# 4. Train and compare settings using training data only
grid_search.fit(X_train, y_train)

# 5. Select the best pipeline
best_model = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation F1:", round(grid_search.best_score_, 4))

# 6. Evaluate on the test set
y_pred = best_model.predict(X_test)


metrics_score = pd.DataFrame({
    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ],
    "score": [
        accuracy_score(y_test, y_pred),
        precision_score(
            y_test, y_pred, pos_label="Pass"
        ),
        recall_score(
            y_test, y_pred, pos_label="Pass"
        ),
        f1_score(
            y_test, y_pred, pos_label="Pass"
        )
    ]
})
 
print(metrics_score.round(4))
print("Confusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Fail", "Pass"]
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=["Fail", "Pass"]
))

joblib.dump(best_model, "model.pkl")
print("Model pipeline saved successfully!")