from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field

# Reusable Type Aliases to ensure consistency across inputs and predictions
DepartmentType = Literal["Sales", "Research & Development", "Human Resources"]
EducationFieldType = Literal["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
JobRoleType = Literal[
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
]


# ---------------------------------------------------------------------------
# EMPLOYEES
# ---------------------------------------------------------------------------
class EmployeeIn(BaseModel):
    Age: int = Field(..., ge=18, le=70)
    Attrition: Literal["Yes", "No"] = "No"
    BusinessTravel: Literal["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    DailyRate: int = Field(..., ge=1)
    Department: DepartmentType
    DistanceFromHome: int = Field(..., ge=0)
    Education: int = Field(..., ge=1, le=5)
    EducationField: EducationFieldType
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    Gender: Literal["Male", "Female"]
    HourlyRate: int = Field(..., ge=1)
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: JobRoleType
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    MonthlyIncome: int = Field(..., ge=1000)
    MonthlyRate: int = Field(..., ge=1)
    NumCompaniesWorked: int = Field(..., ge=0)
    OverTime: Literal["Yes", "No"]
    PercentSalaryHike: int = Field(..., ge=0)
    PerformanceRating: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0)
    TrainingTimesLastYear: int = Field(..., ge=0)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0)
    YearsInCurrentRole: int = Field(..., ge=0)
    YearsSinceLastPromotion: int = Field(..., ge=0)
    YearsWithCurrManager: int = Field(..., ge=0)


class EmployeeAddedResponse(BaseModel):
    message: str
    employee_number: int


class EmployeeListResponse(BaseModel):
    count: int
    total: int
    employees: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------------------------
class AttritionPredictionRequest(BaseModel):
    model: Literal["logistic", "knn", "decision_tree", "random_forest"] = "logistic"
    age: Optional[int] = Field(None, ge=18, le=70)  # FIX: Aligned upper boundary with max employee age
    overtime: Optional[Literal["Yes", "No"]] = None
    monthly_income: Optional[int] = Field(None, ge=1000, le=20000)
    job_satisfaction: Optional[int] = Field(None, ge=1, le=4)
    total_working_years: Optional[int] = Field(None, ge=0, le=40)
    years_at_company: Optional[int] = Field(None, ge=0, le=40)


class AttritionPredictionResponse(BaseModel):
    model: str
    prediction: str
    probability_of_leaving: float


class SalaryPredictionRequest(BaseModel):
    age: Optional[int] = Field(None, ge=18, le=70)  # FIX: Aligned upper boundary with max employee age
    education: Optional[int] = Field(None, ge=1, le=5)
    job_level: Optional[int] = Field(None, ge=1, le=5)
    job_role: Optional[JobRoleType] = None         # FIX: Enforced strict literal validation to prevent encoder crashes
    total_working_years: Optional[int] = Field(None, ge=0, le=40)
    years_at_company: Optional[int] = Field(None, ge=0, le=40)


class SalaryPredictionResponse(BaseModel):
    predicted_monthly_income: float


# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
class ModelPerformanceResponse(BaseModel):
    performance: Dict[str, float]
    confusion_matrices: Dict[str, List[List[int]]]
    top_feature_importances: Dict[str, float]