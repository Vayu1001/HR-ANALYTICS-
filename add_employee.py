"""
Command-line tool to add a new employee record to the hr_analytics database.

Usage:
    python add_employee.py
"""

from db_connection import insert_employee

VALID_BUSINESS_TRAVEL = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
VALID_DEPARTMENTS = ["Sales", "Research & Development", "Human Resources"]
VALID_EDUCATION_FIELDS = [
    "Life Sciences", "Medical", "Marketing",
    "Technical Degree", "Human Resources", "Other"
]
VALID_GENDERS = ["Male", "Female"]
VALID_JOB_ROLES = [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
]
VALID_MARITAL_STATUS = ["Single", "Married", "Divorced"]
VALID_YES_NO = ["Yes", "No"]


def prompt_choice(label, choices):
    choices_str = "/".join(choices)
    while True:
        val = input(f"{label} ({choices_str}): ").strip()
        for c in choices:
            if val.lower() == c.lower():
                return c
        print(f"  -> Please enter one of: {choices_str}")


def prompt_int(label, min_val=None, max_val=None):
    range_hint = ""
    if min_val is not None and max_val is not None:
        range_hint = f" ({min_val}-{max_val})"
    elif min_val is not None:
        range_hint = f" (>= {min_val})"

    while True:
        raw = input(f"{label}{range_hint}: ").strip()
        try:
            val = int(raw)
        except ValueError:
            print("  -> Please enter a whole number.")
            continue
        if min_val is not None and val < min_val:
            print(f"  -> Value must be >= {min_val}")
            continue
        if max_val is not None and val > max_val:
            print(f"  -> Value must be <= {max_val}")
            continue
        return val


def collect_employee_data():
    print("=== Add New Employee ===\n")

    # Initial standalone core field gathers
    age = prompt_int("Age", 18, 70)
    attrition = prompt_choice("Attrition", VALID_YES_NO)
    business_travel = prompt_choice("Business Travel", VALID_BUSINESS_TRAVEL)
    daily_rate = prompt_int("Daily Rate", 1)
    department = prompt_choice("Department", VALID_DEPARTMENTS)
    distance_from_home = prompt_int("Distance From Home", 0)
    education = prompt_int("Education", 1, 5)
    education_field = prompt_choice("Education Field", VALID_EDUCATION_FIELDS)
    env_satisfaction = prompt_int("Environment Satisfaction", 1, 4)
    gender = prompt_choice("Gender", VALID_GENDERS)
    hourly_rate = prompt_int("Hourly Rate", 1)
    job_involvement = prompt_int("Job Involvement", 1, 4)
    job_level = prompt_int("Job Level", 1, 5)
    job_role = prompt_choice("Job Role", VALID_JOB_ROLES)
    job_satisfaction = prompt_int("Job Satisfaction", 1, 4)
    marital_status = prompt_choice("Marital Status", VALID_MARITAL_STATUS)
    monthly_income = prompt_int("Monthly Income", 1000)
    monthly_rate = prompt_int("Monthly Rate", 1)
    num_companies = prompt_int("Number of Companies Worked", 0)
    overtime = prompt_choice("OverTime", VALID_YES_NO)
    percent_hike = prompt_int("Percent Salary Hike", 0)
    perf_rating = prompt_int("Performance Rating", 1, 4)
    rel_satisfaction = prompt_int("Relationship Satisfaction", 1, 4)
    stock_level = prompt_int("Stock Option Level", 0, 3)
    
    # FIX: Logical validation loop for dynamic tenure fields
    total_working_years = prompt_int("Total Working Years", 0)
    while True:
        years_at_company = prompt_int("Years At Company", 0, total_working_years)
        if years_at_company <= total_working_years:
            break
        print(f"  -> 'Years At Company' cannot exceed 'Total Working Years' ({total_working_years}).")

    while True:
        years_in_role = prompt_int("Years In Current Role", 0, years_at_company)
        if years_in_role <= years_at_company:
            break
        print(f"  -> 'Years In Current Role' cannot exceed 'Years At Company' ({years_at_company}).")

    while True:
        years_since_promo = prompt_int("Years Since Last Promotion", 0, years_at_company)
        if years_since_promo <= years_at_company:
            break
        print(f"  -> 'Years Since Last Promotion' cannot exceed 'Years At Company' ({years_at_company}).")

    while True:
        years_with_manager = prompt_int("Years With Current Manager", 0, years_at_company)
        if years_with_manager <= years_at_company:
            break
        print(f"  -> 'Years With Current Manager' cannot exceed 'Years At Company' ({years_at_company}).")

    training_times = prompt_int("Training Times Last Year", 0)
    work_life = prompt_int("Work Life Balance", 1, 4)

    employee = {
        "Age": age,
        "Attrition": attrition,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": env_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies,
        "OverTime": overtime,
        "PercentSalaryHike": percent_hike,
        "PerformanceRating": perf_rating,
        "RelationshipSatisfaction": rel_satisfaction,
        "StockOptionLevel": stock_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times,
        "WorkLifeBalance": work_life,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_since_promo,
        "YearsWithCurrManager": years_with_manager,
    }
    return employee


def main():
    try:
        employee = collect_employee_data()

        print("\nReview the entry above? Press Enter to confirm and save, or Ctrl+C to cancel.")
        input()

        new_id = insert_employee(employee)
        print(f"\n✅ Employee added successfully. Assigned EmployeeNumber = {new_id}")
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user. Exiting.")
    except Exception as e:
        print(f"\n❌ Failed to add employee: {e}")


if __name__ == "__main__":
    main()