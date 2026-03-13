from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
sns.set_style("whitegrid")


def load_dataset() -> pd.DataFrame:
    """Load churn dataset from data/churn.csv."""
    data_path = Path("data") / "churn.csv"

    if not data_path.exists():
        raise FileNotFoundError("Dataset not found. Expected data/churn.csv.")

    df = pd.read_csv(data_path)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nDataset info:")
    print(df.info())
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare raw dataframe for EDA and modeling."""
    df_clean = df.copy()

    # TotalCharges can contain blanks; coerce to numeric and impute.
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    # Drop ID-like column because it does not contain predictive signal.
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # Handle missing numeric and categorical values.
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in categorical_cols:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # Normalize target labels for consistent binary modeling.
    if "Churn" in df_clean.columns:
        df_clean["Churn"] = df_clean["Churn"].map({"No": 0, "Yes": 1})

    return df_clean


def run_eda(df: pd.DataFrame) -> None:
    """Perform exploratory plots requested in the project tasks."""
    Path("visuals").mkdir(exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Churn", data=df)
    plt.title("Churn Distribution (0 = No, 1 = Yes)")
    plt.tight_layout()
    plt.savefig("visuals/churn_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
    plt.title("Monthly Charges vs Churn")
    plt.tight_layout()
    plt.savefig("visuals/monthly_charges_vs_churn.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Churn", y="tenure", data=df)
    plt.title("Tenure vs Churn")
    plt.tight_layout()
    plt.savefig("visuals/tenure_vs_churn.png", dpi=300)
    plt.close()

    # Correlation heatmap is created from one-hot encoded data.
    corr_df = pd.get_dummies(df, drop_first=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_df.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("visuals/correlation_heatmap.png", dpi=300)
    plt.close()


def build_preprocessors(X: pd.DataFrame):
    """Build preprocessing transformers for logistic regression and random forest."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    logistic_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    rf_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return logistic_preprocessor, rf_preprocessor


def evaluate_model(model_name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate model and plot confusion matrix."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 40}")
    print(f"{model_name} Results")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"visuals/{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=300)
    plt.close()

    return accuracy


def get_rf_feature_importance(rf_model: Pipeline, top_n: int = 10) -> pd.DataFrame:
    """Extract and return top feature importances from random forest pipeline."""
    preprocessor: ColumnTransformer = rf_model.named_steps["preprocessor"]
    classifier: RandomForestClassifier = rf_model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)

    print("\nTop feature importances (Random Forest):")
    print(fi_df.head(top_n))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df.head(top_n), x="importance", y="feature", palette="viridis")
    plt.title(f"Top {top_n} Factors Influencing Churn")
    plt.tight_layout()
    plt.savefig("visuals/top_feature_importance.png", dpi=300)
    plt.close()

    return fi_df


def main() -> None:
    df_raw = load_dataset()
    df = clean_data(df_raw)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    run_eda(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    logistic_preprocessor, rf_preprocessor = build_preprocessors(X)

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", logistic_preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", rf_preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    logistic_acc = evaluate_model("Logistic Regression", logistic_model, X_test, y_test)
    rf_acc = evaluate_model("Random Forest", rf_model, X_test, y_test)

    comparison_df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy": [logistic_acc, rf_acc],
        }
    ).sort_values("Accuracy", ascending=False)

    print("\nModel performance comparison:")
    print(comparison_df)

    _ = get_rf_feature_importance(rf_model, top_n=12)

    best_model = rf_model if rf_acc >= logistic_acc else logistic_model
    Path("models").mkdir(exist_ok=True)
    model_path = Path("models") / "best_churn_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nSaved best model to: {model_path}")


if __name__ == "__main__":
    main()
