
"""
PROJECT: Iris Flower Classification
AUTHOR: Your Name
DESCRIPTION: End-to-end Machine Learning Classification Project
"""

import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ==========================================
# 2. DATASET LOADING
# ==========================================

def load_dataset():

    iris = load_iris(as_frame=True)

    X = iris.data
    y = iris.target

    df = X.copy()
    df["target"] = y

    df.to_csv("dataset/iris.csv", index=False)

    print("\nDATASET LOADED SUCCESSFULLY")
    print("Dataset shape:", df.shape)

    return X, y, iris.target_names


# ==========================================
# 3. EXPLORATORY DATA ANALYSIS
# ==========================================

def perform_eda(X, y):

    print("\nDATASET INFORMATION")
    print(X.info())

    print("\nSTATISTICAL SUMMARY")
    print(X.describe())

    print("\nMISSING VALUES")
    print(X.isnull().sum())

    print("\nCLASS DISTRIBUTION")
    print(y.value_counts())

    # Correlation heatmap
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        X.corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    plt.savefig("outputs/correlation_matrix.png")
    plt.close()


# ==========================================
# 4. DATA PREPROCESSING
# ==========================================

def create_pipeline(classifier):

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])


# ==========================================
# 5. MODEL CONFIGURATION
# ==========================================

def get_models():

    return {

        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {
                "classifier__C": [0.1, 1, 10]
            }
        ),

        "Support Vector Machine": (
            SVC(),
            {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["linear", "rbf"]
            }
        ),

        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10]
            }
        )
    }


# ==========================================
# 6. MODEL TRAINING AND COMPARISON
# ==========================================

def train_models(X_train, y_train):

    models = get_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    best_score = -1
    best_model = None
    best_name = None

    print("\nMODEL TRAINING STARTED")

    for name, (classifier, parameters) in models.items():

        pipeline = create_pipeline(classifier)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        score = search.best_score_

        results.append({
            "Model": name,
            "Cross Validation F1": score,
            "Best Parameters": str(search.best_params_)
        })

        print(f"\n{name}: {score:.4f}")

        if score > best_score:

            best_score = score
            best_model = search.best_estimator_
            best_name = name

    # Save comparison results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "outputs/model_comparison.csv",
        index=False
    )

    print("\nBEST MODEL:", best_name)

    return best_model, best_name


# ==========================================
# 7. MODEL EVALUATION
# ==========================================

def evaluate_model(model, X_test, y_test, target_names):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\nFINAL MODEL PERFORMANCE")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nCLASSIFICATION REPORT")

    print(classification_report(
        y_test,
        predictions,
        target_names=target_names
    ))

    # Save confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Iris Classification Confusion Matrix")

    plt.tight_layout()

    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    with open("outputs/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    return metrics


# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

def save_model(model, model_name, feature_names, target_names):

    artifact = {
        "model": model,
        "model_name": model_name,
        "feature_names": list(feature_names),
        "target_names": list(target_names)
    }

    joblib.dump(
        artifact,
        "models/iris_classifier.pkl"
    )

    print("\nMODEL SAVED SUCCESSFULLY")


# ==========================================
# 9. PREDICTION FUNCTION
# ==========================================

def predict_flower(values):

    artifact = joblib.load(
        "models/iris_classifier.pkl"
    )

    model = artifact["model"]

    input_data = pd.DataFrame(
        [values],
        columns=artifact["feature_names"]
    )

    prediction = model.predict(input_data)[0]

    return artifact["target_names"][prediction]


# ==========================================
# 10. MAIN EXECUTION
# ==========================================

def main():

    print("=" * 50)
    print("IRIS FLOWER CLASSIFICATION SYSTEM")
    print("=" * 50)

    # Load dataset
    X, y, target_names = load_dataset()

    # Perform EDA
    perform_eda(X, y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Train models
    best_model, best_name = train_models(
        X_train,
        y_train
    )

    # Evaluate selected model
    metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
        target_names
    )

    # Save model
    save_model(
        best_model,
        best_name,
        X.columns,
        target_names
    )

    # Sample prediction
    sample = [5.1, 3.5, 1.4, 0.2]

    result = predict_flower(sample)

    print("\nSAMPLE PREDICTION")
    print("Input measurements:", sample)
    print("Predicted species:", result)

    print("\nPROJECT EXECUTED SUCCESSFULLY")


if __name__ == "__main__":
    main()
