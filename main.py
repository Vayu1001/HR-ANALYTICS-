import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    EmployeeIn,
    EmployeeAddedResponse,
    EmployeeListResponse,
    AttritionPredictionRequest,
    AttritionPredictionResponse,
    SalaryPredictionRequest,
    SalaryPredictionResponse,
    ModelPerformanceResponse,
)
from ml_utils import (
    load_resources,
    refresh_employee_data,
    predict_attrition,
    predict_salary,
    get_model_performance,
    generate_visualization,
    VALID_GRAPH_TYPES,
)

# FIX: Drop the sys.path.append manipulation since db_connection is in the same folder
from db_connection import insert_employee  

app = FastAPI(
    title="HR Analytics API",
    description=(
        "REST API mirroring the HR Analytics Streamlit dashboard: employee "
        "data, attrition/salary predictions, model performance, and visualizations."
    ),
    version="1.0.0",
)

# Allow browser-based frontends to call this API during development.
# Tighten allow_origins to your actual frontend URL(s) before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    # Load models + data once at startup instead of on the first request.
    load_resources()


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "HR Analytics API is running"}


# ---------------------------------------------------------------------------
# EMPLOYEES
# ---------------------------------------------------------------------------
@app.get("/employees", response_model=EmployeeListResponse, tags=["Employees"])
def list_employees(limit: int = 100, offset: int = 0):
    """Returns cleaned employee records, paginated."""
    resources = load_resources()
    df = resources["df"]
    records = df.iloc[offset: offset + limit].to_dict(orient="records")
    return {"count": len(records), "total": df.shape[0], "employees": records}


@app.get("/employees/{employee_number}", tags=["Employees"])
def get_employee(employee_number: int):
    resources = load_resources()
    df = resources["df"]
    match = df[df["EmployeeNumber"] == employee_number]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Employee {employee_number} not found")
    return match.iloc[0].to_dict()


@app.post("/employees", response_model=EmployeeAddedResponse, status_code=201, tags=["Employees"])
def add_employee(employee: EmployeeIn):
    try:
        new_id = insert_employee(employee.dict(exclude_none=True))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add employee: {e}")

    refresh_employee_data()
    return {"message": "Employee added successfully", "employee_number": new_id}


# ---------------------------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------------------------
@app.post("/predict/attrition", response_model=AttritionPredictionResponse, tags=["Predictions"])
def predict_attrition_endpoint(request: AttritionPredictionRequest):
    """
    Predicts whether an employee is likely to leave.
    Any field left out uses the dataset's mode/median as a default,
    mirroring the Streamlit Attrition Predictor tabs.
    """
    return predict_attrition(request)


@app.post("/predict/salary", response_model=SalaryPredictionResponse, tags=["Predictions"])
def predict_salary_endpoint(request: SalaryPredictionRequest):
    """
    Predicts monthly income. Any field left out uses the dataset's
    mode/median as a default, mirroring the Streamlit Salary Predictor tab.
    """
    predicted = predict_salary(request)
    return {"predicted_monthly_income": predicted}


# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
@app.get("/model-performance", response_model=ModelPerformanceResponse, tags=["Model Performance"])
def model_performance_endpoint():
    """Accuracy/R² for every model, confusion matrices, and Random Forest
    feature importances — mirrors the Streamlit Model Performance tab."""
    return get_model_performance()


# ---------------------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------------------
@app.get("/visualizations/{graph_type}", tags=["Visualizations"])
def visualization_endpoint(graph_type: str):
    """
    Returns a PNG chart. Valid graph_type values:
    department_attrition, age_distribution, overtime_attrition,
    income_attrition, satisfaction_attrition, correlation_heatmap
    """
    if graph_type not in VALID_GRAPH_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid graph_type '{graph_type}'. Valid options: {VALID_GRAPH_TYPES}",
        )
    try:
        buf = generate_visualization(graph_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(buf, media_type="image/png")
