import streamlit as st
import joblib
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")

st.title("HR Analytics · Attrition & Salary Prediction")
st.write("Explore workforce data, predict attrition risk, estimate salary, and check model performance.")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

# Force absolute path resolution for the backend source directory
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "src"))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Backend"))

# Append both possibilities to the system path safely
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from db_connection import get_employee_data, insert_employee
from preprocessing import clean_data, get_attrition_data, get_salary_data

# ---------------------------------------------------------------------------
# CACHED DATA & ARTIFACT LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_ml_artifacts():
    log_model = joblib.load(os.path.join(MODELS_DIR, "logistic_model.pkl"))
    knn_model = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
    dt_model = joblib.load(os.path.join(MODELS_DIR, "decision_tree_model.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    attrition_encoders = joblib.load(os.path.join(MODELS_DIR, "encoder_log.pkl"))
    attrition_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_log.pkl"))
    
    lin_model = joblib.load(os.path.join(MODELS_DIR, "Linear_model.pkl"))
    salary_encoders = joblib.load(os.path.join(MODELS_DIR, "encoder_lin.pkl"))
    salary_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_lin.pkl"))
    
    return (log_model, knn_model, dt_model, rf_model, attrition_encoders, attrition_scaler, 
            lin_model, salary_encoders, salary_scaler)

(log_model, knn_model, dt_model, rf_model, attrition_encoders, attrition_scaler,
 lin_model, salary_encoders, salary_scaler) = load_ml_artifacts()

@st.cache_data(ttl=600)
def fetch_and_clean_workforce():
    raw_df = get_employee_data()
    return clean_data(raw_df)

df = fetch_and_clean_workforce()

# Fetch baseline modeling feature distributions once safely
try:
    X_train_p, X_test_p, y_train_p, y_test_p, _, _ = get_attrition_data()
    X_train_s, X_test_s, y_train_s, y_test_s, _, _ = get_salary_data()
    feature_columns_attrition = X_train_p.columns.tolist()
    feature_columns_salary = X_train_s.columns.tolist()
except Exception as e:
    st.error(f"Failed to extract modeling array footprints: {e}")
    feature_columns_attrition = []
    feature_columns_salary = []

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Dashboard")
    st.caption("Internal HR analytics tool")
    st.metric("Total employees", df.shape[0])
    if "Attrition" in df.columns:
        try:
            leave_rate = (df["Attrition"] == "Yes").mean()
            st.metric("Historical attrition rate", f"{leave_rate:.1%}")
        except Exception:
            pass
    st.caption("Models Active: Logistic Regression · KNN · Decision Tree · Random Forest · Linear Regression")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Data Explorer", "Attrition Predictor (Logistic/KNN)", "Decision Tree & Random Forest",
    "Salary Predictor", "Model Performance", "Visualizations", "Add Employee"
])

# Helpers to guarantee feature footprints align precisely with pipeline requirements
def align_and_transform_pipeline(input_dict, encoders, scaler, columns_blueprint, skip_scale_col="MonthlyIncome"):
    input_df = pd.DataFrame([input_dict])
    
    # Run categorical structural tracking transforms
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])
            
    # Normalize numeric distributions
    num_cols = [c for c in input_df.select_dtypes(include=["int64", "float64"]).columns if c != skip_scale_col]
    if num_cols and scaler:
        input_df[num_cols] = scaler.transform(input_df[num_cols])
        
    # Reindex target structure matching backend model footprints completely
    if columns_blueprint:
        for missing_col in [c for c in columns_blueprint if c not in input_df.columns]:
            input_df[missing_col] = 0
        input_df = input_df[columns_blueprint]
    return input_df

# ---------------------------------------------------------------------------
# TAB 1 — DATA EXPLORER
# ---------------------------------------------------------------------------
with tab1:
    st.header("Data Explorer")
    st.write("Browse the full cleaned employee dataset below.")
    st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — ATTRITION PREDICTOR
# ---------------------------------------------------------------------------
with tab2:
    st.header("Attrition Predictor (Logistic Regression / KNN)")
    model_choice = st.selectbox("Choose model", ["Logistic Regression", "KNN"], key="attrition_model_choice")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 70, 30)
        overtime = st.selectbox("OverTime", ["Yes", "No"])
    with col2:
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    with col3:
        total_working_years = st.slider("Total Working Years", 0, 50, 5)
        years_at_company = st.slider("Years at Company", 0, total_working_years, min(3, total_working_years))

    if st.button("Predict Attrition", key="predict_attrition"):
        input_dict = {}
        for col in df.columns:
            if col in ["Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]:
                continue
            input_dict[col] = df[col].mode()[0] if df[col].dtype == "object" else df[col].median()

        input_dict.update({
            "Age": age, "MonthlyIncome": monthly_income, "TotalWorkingYears": total_working_years,
            "YearsAtCompany": years_at_company, "OverTime": overtime, "JobSatisfaction": job_satisfaction
        })

        processed_input = align_and_transform_pipeline(input_dict, attrition_encoders, attrition_scaler, feature_columns_attrition)
        chosen_model = log_model if model_choice == "Logistic Regression" else knn_model

        prediction = chosen_model.predict(processed_input)[0]
        probability = chosen_model.predict_proba(processed_input)[0][1]

        if prediction == 1:
            st.error(f"({model_choice}) This employee is likely to leave. (Probability: {probability:.2%})")
        else:
            st.success(f"({model_choice}) This employee is likely to stay. (Probability of leaving: {probability:.2%})")

# ---------------------------------------------------------------------------
# TAB 3 — DECISION TREE & RANDOM FOREST PREDICTOR
# ---------------------------------------------------------------------------
with tab3:
    st.header("Decision Tree & Random Forest Predictor")

    col1, col2, col3 = st.columns(3)
    with col1:
        age_t = st.slider("Age", 18, 70, 30, key="age_tree")
        overtime_t = st.selectbox("OverTime", ["Yes", "No"], key="overtime_tree")
    with col2:
        monthly_income_t = st.number_input("Monthly Income", 1000, 20000, 5000, key="income_tree")
        job_satisfaction_t = st.slider("Job Satisfaction (1-4)", 1, 4, 3, key="jobsat_tree")
    with col3:
        total_working_years_t = st.slider("Total Working Years", 0, 50, 5, key="twy_tree")
        years_at_company_t = st.slider("Years at Company", 0, total_working_years_t, min(3, total_working_years_t), key="yac_tree")

    if st.button("Predict Attrition", key="predict_tree"):
        input_dict_t = {}
        for col in df.columns:
            if col in ["Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]:
                continue
            input_dict_t[col] = df[col].mode()[0] if df[col].dtype == "object" else df[col].median()

        input_dict_t.update({
            "Age": age_t, "MonthlyIncome": monthly_income_t, "TotalWorkingYears": total_working_years_t,
            "YearsAtCompany": years_at_company_t, "OverTime": overtime_t, "JobSatisfaction": job_satisfaction_t
        })

        processed_input_t = align_and_transform_pipeline(input_dict_t, attrition_encoders, attrition_scaler, feature_columns_attrition)

        dt_prediction = dt_model.predict(processed_input_t)[0]
        dt_probability = dt_model.predict_proba(processed_input_t)[0][1]
        rf_prediction = rf_model.predict(processed_input_t)[0]
        rf_probability = rf_model.predict_proba(processed_input_t)[0][1]

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("Decision Tree")
            if dt_prediction == 1:
                st.error(f"Likely to leave (Probability: {dt_probability:.2%})")
            else:
                st.success(f"Likely to stay (Probability of leaving: {dt_probability:.2%})")
        with res_col2:
            st.subheader("Random Forest")
            if rf_prediction == 1:
                st.error(f"Likely to leave (Probability: {rf_probability:.2%})")
            else:
                st.success(f"Likely to stay (Probability of leaving: {rf_probability:.2%})")

# ---------------------------------------------------------------------------
# TAB 4 — SALARY PREDICTOR
# ---------------------------------------------------------------------------
with tab4:
    st.header("Salary Predictor")

    col1, col2, col3 = st.columns(3)
    with col1:
        age_s = st.slider("Age", 18, 70, 30, key="age_salary")
        education_s = st.slider("Education (1-5)", 1, 5, 3, key="edu_salary")
    with col2:
        job_level = st.slider("Job Level (1-5)", 1, 5, 2, key="joblevel_salary")
        job_role_s = st.selectbox("Job Role", df["JobRole"].unique(), key="jobrole_salary")
    with col3:
        total_working_years_s = st.slider("Total Working Years", 0, 50, 5, key="twy_salary")
        years_at_company_s = st.slider("Years at Company", 0, total_working_years_s, min(3, total_working_years_s), key="yac_salary")

    if st.button("Predict Salary", key="predict_salary"):
        input_dict_s = {}
        for col in df.columns:
            if col in ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours", "MonthlyIncome"]:
                continue
            input_dict_s[col] = df[col].mode()[0] if df[col].dtype == "object" else df[col].median()

        input_dict_s.update({
            "Age": age_s, "JobLevel": job_level, "TotalWorkingYears": total_working_years_s,
            "YearsAtCompany": years_at_company_s, "JobRole": job_role_s, "Education": education_s, "Attrition": 0
        })

        processed_input_s = align_and_transform_pipeline(input_dict_s, salary_encoders, salary_scaler, feature_columns_salary, skip_scale_col="Attrition")
        predicted_salary = lin_model.predict(processed_input_s)[0]
        st.info(f"Predicted Monthly Income: ${max(0.0, predicted_salary):,.2f}")

# ---------------------------------------------------------------------------
# TAB 5 — MODEL PERFORMANCE
# ---------------------------------------------------------------------------
with tab5:
    st.header("Model Performance")

    log_preds = log_model.predict(X_test_p)
    knn_preds = knn_model.predict(X_test_p)
    dt_preds = dt_model.predict(X_test_p)
    rf_preds = rf_model.predict(X_test_p)
    lin_preds = lin_model.predict(X_test_s)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Logistic Accuracy", f"{accuracy_score(y_test_p, log_preds):.1%}")
    col2.metric("KNN Accuracy", f"{accuracy_score(y_test_p, knn_preds):.1%}")
    col3.metric("Decision Tree Accuracy", f"{accuracy_score(y_test_p, dt_preds):.1%}")
    col4.metric("Random Forest Accuracy", f"{accuracy_score(y_test_p, rf_preds):.1%}")
    col5.metric("Salary R² Score", f"{r2_score(y_test_s, lin_preds):.3f}")

    st.write("")
    c1, c2 = st.columns(2)

    with c1:
        cm_model_choice = st.selectbox(
            "Confusion matrix for", ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"], key="cm_model_choice"
        )
        preds_map = {"Logistic Regression": log_preds, "KNN": knn_preds, "Decision Tree": dt_preds, "Random Forest": rf_preds}
        st.subheader(f"Confusion Matrix ({cm_model_choice})")
        cm = confusion_matrix(y_test_p, preds_map[cm_model_choice])
        cm_df = pd.DataFrame(cm, index=["Actual: Stay", "Actual: Leave"], columns=["Pred: Stay", "Pred: Leave"])
        st.dataframe(cm_df, use_container_width=True)

    with c2:
        st.subheader("Model Comparison")
        comparison_df = pd.DataFrame({
            "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
            "Accuracy": [accuracy_score(y_test_p, log_preds), accuracy_score(y_test_p, knn_preds),
                         accuracy_score(y_test_p, dt_preds), accuracy_score(y_test_p, rf_preds)]
        })
        st.dataframe(comparison_df, use_container_width=True)
        st.bar_chart(comparison_df.set_index("Model"))

    st.write("")
    st.subheader("Random Forest — Top Feature Importances")
    try:
        importances = pd.Series(rf_model.feature_importances_, index=X_train_p.columns).sort_values(ascending=False).head(10)
        st.bar_chart(importances)
    except Exception as e:
        st.warning(f"Could not display feature importances: {e}")

# ---------------------------------------------------------------------------
# TAB 6 — VISUALIZATIONS
# ---------------------------------------------------------------------------
with tab6:
    st.header("Visualizations")
    graph_choice = st.selectbox(
        "Choose a graph to display",
        ["Attrition Rate by Department", "Age Distribution by Attrition", "OverTime vs Attrition",
         "Monthly Income by Attrition", "Job Satisfaction vs Attrition", "Correlation Heatmap (Numeric Features)"],
        key="graph_choice"
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    if graph_choice == "Attrition Rate by Department":
        dept_attrition = df.groupby("Department")["Attrition"].apply(lambda x: (x == "Yes").mean()).reset_index(name="AttritionRate")
        sns.barplot(data=dept_attrition, x="Department", y="AttritionRate", ax=ax)
        ax.set_ylabel("Attrition Rate")
        for container in ax.containers:
            ax.bar_label(container, fmt=lambda v: f"{v:.1%}")
        plt.xticks(rotation=15)
    elif graph_choice == "Age Distribution by Attrition":
        sns.histplot(data=df, x="Age", hue="Attrition", bins=20, multiple="layer", alpha=0.6, ax=ax)
    elif graph_choice == "OverTime vs Attrition":
        sns.countplot(data=df, x="OverTime", hue="Attrition", ax=ax)
    elif graph_choice == "Monthly Income by Attrition":
        sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", hue="Attrition", legend=False, ax=ax)
    elif graph_choice == "Job Satisfaction vs Attrition":
        sns.countplot(data=df, x="JobSatisfaction", hue="Attrition", ax=ax)
    elif graph_choice == "Correlation Heatmap (Numeric Features)":
        sns.heatmap(df.select_dtypes(include=["int64", "float64"]).corr(), cmap="RdBu_r", vmin=-1, vmax=1, annot=False, ax=ax)

    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# TAB 7 — ADD EMPLOYEE
# ---------------------------------------------------------------------------
with tab7:
    st.header("Add New Employee")
    st.caption("Insert a new employee record directly into the database.")

    with st.form("add_employee_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_age = st.slider("Age", 18, 70, 30)
            new_attrition = st.selectbox("Attrition", ["No", "Yes"])
            new_business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            new_daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
            new_department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            new_distance = st.number_input("Distance From Home", 0, 50, 5)
            new_education = st.slider("Education (1-5)", 1, 5, 3)
            new_education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
            new_env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
            new_gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            new_hourly_rate = st.number_input("Hourly Rate", 30, 150, 60)
            new_job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
            new_job_level = st.slider("Job Level (1-5)", 1, 5, 2)
            new_job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
            new_job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            new_marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            new_monthly_income = st.number_input("Monthly Income", 1000, 25000, 5000)
            new_monthly_rate = st.number_input("Monthly Rate", 1000, 30000, 15000)
            new_num_companies = st.number_input("Number of Companies Worked", 0, 15, 1)
            new_overtime = st.selectbox("OverTime", ["No", "Yes"])
        with col3:
            new_salary_hike = st.number_input("Percent Salary Hike", 0, 30, 15)
            new_performance = st.slider("Performance Rating (1-4)", 1, 4, 3)
            new_relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
            new_stock_option = st.slider("Stock Option Level (0-3)", 0, 3, 0)
            new_total_working_years = st.number_input("Total Working Years", 0, 50, 5)
            new_training_times = st.number_input("Training Times Last Year", 0, 10, 2)
            new_wlb = st.slider("Work Life Balance (1-4)", 1, 4, 3)
            new_years_at_company = st.number_input("Years At Company", 0, 50, 3)
            new_years_current_role = st.number_input("Years In Current Role", 0, 30, 2)
            new_years_since_promotion = st.number_input("Years Since Last Promotion", 0, 30, 1)
            new_years_curr_manager = st.number_input("Years With Current Manager", 0, 30, 2)

        submitted = st.form_submit_button("Add Employee")

    if submitted:
        # Cross-field validation to protect database row logic integrity
        if new_years_at_company > new_total_working_years:
            st.error(f"❌ Structural Mismatch: 'Years At Company' ({new_years_at_company}) cannot exceed 'Total Working Years' ({new_total_working_years}).")
        elif new_years_current_role > new_years_at_company:
            st.error(f"❌ Structural Mismatch: 'Years In Current Role' ({new_years_current_role}) cannot exceed 'Years At Company' ({new_years_at_company}).")
        elif new_years_since_promotion > new_years_at_company:
            st.error(f"❌ Structural Mismatch: 'Years Since Last Promotion' ({new_years_since_promotion}) cannot exceed 'Years At Company' ({new_years_at_company}).")
        elif new_years_curr_manager > new_years_at_company:
            st.error(f"❌ Structural Mismatch: 'Years With Current Manager' ({new_years_curr_manager}) cannot exceed 'Years At Company' ({new_years_at_company}).")
        else:
            new_employee = {
                "Age": new_age, "Attrition": new_attrition, "BusinessTravel": new_business_travel, "DailyRate": new_daily_rate,
                "Department": new_department, "DistanceFromHome": new_distance, "Education": new_education, "EducationField": new_education_field,
                "EnvironmentSatisfaction": new_env_satisfaction, "Gender": new_gender, "HourlyRate": new_hourly_rate, "JobInvolvement": new_job_involvement,
                "JobLevel": new_job_level, "JobRole": new_job_role, "JobSatisfaction": new_job_satisfaction, "MaritalStatus": new_marital_status,
                "MonthlyIncome": new_monthly_income, "MonthlyRate": new_monthly_rate, "NumCompaniesWorked": new_num_companies, "OverTime": new_overtime,
                "PercentSalaryHike": new_salary_hike, "PerformanceRating": new_performance, "RelationshipSatisfaction": new_relationship_satisfaction,
                "StockOptionLevel": new_stock_option, "TotalWorkingYears": new_total_working_years, "TrainingTimesLastYear": new_training_times,
                "WorkLifeBalance": new_wlb, "YearsAtCompany": new_years_at_company, "YearsInCurrentRole": new_years_current_role,
                "YearsSinceLastPromotion": new_years_since_promotion, "YearsWithCurrManager": new_years_curr_manager
            }
            try:
                new_id = insert_employee(new_employee)
                st.success(f"🎉 Employee added successfully! Assigned EmployeeNumber = {new_id}")
                st.cache_data.clear()  # Drop cache to pull fresh insertions immediately
            except Exception as e:
                st.error(f"Failed to add employee: {e}")