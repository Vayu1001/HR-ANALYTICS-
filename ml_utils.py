import os
import sys
import io

import joblib
import pandas as pd

# Clean up path resolution to target the actual project setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# FIX: Local imports are handled directly as they are in the same folder
from db_connection import get_employee_data          
from preprocessing import clean_data, get_attrition_data, get_salary_data  

# Simple in-process cache so models/data are loaded once, not per-request.
_cache = {}


def load_resources():
    """
    Loads (and caches) the trained models, encoders, scalers, and the
    cleaned employee dataframe. Safe to call from every request; only
    does real work the first time.
    """
    if _cache:
        return _cache

    _cache["log_model"] = joblib.load(os.path.join(MODELS_DIR, "logistic_model.pkl"))
    _cache["knn_model"] = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
    _cache["dt_model"] = joblib.load(os.path.join(MODELS_DIR, "decision_tree_model.pkl"))
    _cache["rf_model"] = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    _cache["attrition_encoders"] = joblib.load(os.path.join(MODELS_DIR, "encoder_log.pkl"))
    _cache["attrition_scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler_log.pkl"))

    _cache["lin_model"] = joblib.load(os.path.join(MODELS_DIR, "Linear_model.pkl"))
    _cache["salary_encoders"] = joblib.load(os.path.join(MODELS_DIR, "encoder_lin.pkl"))
    _cache["salary_scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler_lin.pkl"))

    _cache["df"] = clean_data(get_employee_data())

    return _cache


def refresh_employee_data():
    """Re-fetches and re-cleans employee data. Call this after inserting
    a new employee so subsequent requests see it."""
    df = clean_data(get_employee_data())
    _cache["df"] = df
    return df


# ---------------------------------------------------------------------------
# ATTRITION PREDICTION
# ---------------------------------------------------------------------------
def build_attrition_input(overrides):
    resources = load_resources()
    df = resources["df"]

    input_dict = {}
    for col in df.columns:
        if col in ["Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]:
            continue
        if df[col].dtype == "object":
            input_dict[col] = df[col].mode()[0]
        else:
            input_dict[col] = df[col].median()

    if overrides.age is not None:
        input_dict["Age"] = overrides.age
    if overrides.monthly_income is not None:
        input_dict["MonthlyIncome"] = overrides.monthly_income
    if overrides.total_working_years is not None:
        input_dict["TotalWorkingYears"] = overrides.total_working_years
    if overrides.years_at_company is not None:
        input_dict["YearsAtCompany"] = overrides.years_at_company
    if overrides.overtime is not None:
        input_dict["OverTime"] = overrides.overtime
    if overrides.job_satisfaction is not None:
        input_dict["JobSatisfaction"] = overrides.job_satisfaction

    return pd.DataFrame([input_dict])


def predict_attrition(overrides):
    resources = load_resources()
    input_df = build_attrition_input(overrides)

    encoders = resources["attrition_encoders"]
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    scaler = resources["attrition_scaler"]
    num_cols = [c for c in input_df.select_dtypes(include=["int64", "float64"]).columns if c != "MonthlyIncome"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    model_map = {
        "logistic": resources["log_model"],
        "knn": resources["knn_model"],
        "decision_tree": resources["dt_model"],
        "random_forest": resources["rf_model"],
    }
    model = model_map[overrides.model]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "model": overrides.model,
        "prediction": "Leave" if prediction == 1 else "Stay",
        "probability_of_leaving": float(probability),
    }


# ---------------------------------------------------------------------------
# SALARY PREDICTION
# ---------------------------------------------------------------------------
def build_salary_input(overrides):
    resources = load_resources()
    df = resources["df"]

    input_dict = {}
    for col in df.columns:
        if col in ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours", "MonthlyIncome"]:
            continue
        if df[col].dtype == "object":
            input_dict[col] = df[col].mode()[0]
        else:
            input_dict[col] = df[col].median()

    if overrides.age is not None:
        input_dict["Age"] = overrides.age
    if overrides.job_level is not None:
        input_dict["JobLevel"] = overrides.job_level
    if overrides.total_working_years is not None:
        input_dict["TotalWorkingYears"] = overrides.total_working_years
    if overrides.years_at_company is not None:
        input_dict["YearsAtCompany"] = overrides.years_at_company
    if overrides.job_role is not None:
        input_dict["JobRole"] = overrides.job_role
    if overrides.education is not None:
        input_dict["Education"] = overrides.education
    input_dict["Attrition"] = 0

    return pd.DataFrame([input_dict])


def predict_salary(overrides):
    resources = load_resources()
    input_df = build_salary_input(overrides)

    encoders = resources["salary_encoders"]
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    scaler = resources["salary_scaler"]
    num_cols = [c for c in input_df.select_dtypes(include=["int64", "float64"]).columns if c != "Attrition"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    predicted_salary = resources["lin_model"].predict(input_df)[0]
    return float(predicted_salary)


# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
def get_model_performance():
    from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

    resources = load_resources()

    X_train_p, X_test_p, y_train_p, y_test_p, _, _ = get_attrition_data()
    log_preds = resources["log_model"].predict(X_test_p)
    knn_preds = resources["knn_model"].predict(X_test_p)
    dt_preds = resources["dt_model"].predict(X_test_p)
    rf_preds = resources["rf_model"].predict(X_test_p)

    X_train_s, X_test_s, y_train_s, y_test_s, _, _ = get_salary_data()
    lin_preds = resources["lin_model"].predict(X_test_s)

    performance = {
        "logistic_accuracy": float(accuracy_score(y_test_p, log_preds)),
        "knn_accuracy": float(accuracy_score(y_test_p, knn_preds)),
        "decision_tree_accuracy": float(accuracy_score(y_test_p, dt_preds)),
        "random_forest_accuracy": float(accuracy_score(y_test_p, rf_preds)),
        "salary_r2_score": float(r2_score(y_test_s, lin_preds)),
    }

    confusion_matrices = {
        "logistic": confusion_matrix(y_test_p, log_preds).tolist(),
        "knn": confusion_matrix(y_test_p, knn_preds).tolist(),
        "decision_tree": confusion_matrix(y_test_p, dt_preds).tolist(),
        "random_forest": confusion_matrix(y_test_p, rf_preds).tolist(),
    }

    try:
        importances = pd.Series(
            resources["rf_model"].feature_importances_, index=X_train_p.columns
        ).sort_values(ascending=False).head(10)
        top_feature_importances = {k: float(v) for k, v in importances.to_dict().items()}
    except Exception:
        top_feature_importances = {}

    return {
        "performance": performance,
        "confusion_matrices": confusion_matrices,
        "top_feature_importances": top_feature_importances,
    }


# ---------------------------------------------------------------------------
# VISUALIZATIONS (returns PNG bytes)
# ---------------------------------------------------------------------------
VALID_GRAPH_TYPES = [
    "department_attrition",
    "age_distribution",
    "overtime_attrition",
    "income_attrition",
    "satisfaction_attrition",
    "correlation_heatmap",
]


def generate_visualization(graph_type):
    import matplotlib
    matplotlib.use("Agg")  # no display available on a server
    import matplotlib.pyplot as plt
    import seaborn as sns

    resources = load_resources()
    df = resources["df"]

    fig, ax = plt.subplots(figsize=(6, 4))

    if graph_type == "department_attrition":
        # FIX: Explicitly handle both integer flags (1) and string flags ('Yes') safely
        dept_attrition = (
            df.groupby("Department")["Attrition"]
            .apply(lambda x: (x.astype(str).str.lower().isin(["yes", "1"])).mean())
            .reset_index(name="AttritionRate")
        )
        sns.barplot(data=dept_attrition, x="Department", y="AttritionRate", ax=ax)
        ax.set_ylabel("Attrition Rate")
        for container in ax.containers:
            ax.bar_label(container, fmt=lambda v: f"{v:.1%}")
        plt.setp(ax.get_xticklabels(), rotation=15)

    elif graph_type == "age_distribution":
        sns.histplot(data=df, x="Age", hue="Attrition", bins=20, multiple="layer", alpha=0.6, ax=ax)

    elif graph_type == "overtime_attrition":
        sns.countplot(data=df, x="OverTime", hue="Attrition", ax=ax)

    elif graph_type == "income_attrition":
        sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", hue="Attrition", legend=False, ax=ax)

    elif graph_type == "satisfaction_attrition":
        sns.countplot(data=df, x="JobSatisfaction", hue="Attrition", ax=ax)

    elif graph_type == "correlation_heatmap":
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        corr = numeric_df.corr()
        sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, annot=False, ax=ax)

    else:
        plt.close(fig)
        raise ValueError(f"Unknown graph_type: {graph_type}")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf