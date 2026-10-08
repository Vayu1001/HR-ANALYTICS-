import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from db_connection import get_employee_data

def load_raw_data():
    d = get_employee_data()
    return d

def clean_data(df):
    df = df.copy()
    
    # FIX: Add errors="ignore" so live prediction payloads don't throw KeyErrors if these columns aren't passed
    df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], errors="ignore")
    df = df.rename(columns={"ï»¿Age": "Age"})

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    return df

# Your categorical columns are converted into numbers using LabelEncoder
def encode_categorical(df, encoders=None):
    df = df.copy()
    col = df.select_dtypes(include="object").columns.tolist()

    if encoders is None:
        encoders = {}
        for i in col:
            le = LabelEncoder()
            df[i] = le.fit_transform(df[i])
            encoders[i] = le
    else:
        for i in col:
            if i in df.columns:  # Safeguard for selective feature inference pipelines
                le = encoders[i]
                df[i] = le.transform(df[i])
    return df, encoders

def scale_numerical(df, target_cols, scaler=None):
    df = df.copy()
    col = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c not in target_cols]

    if scaler is None:    
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[col])
    else:
        df[col] = scaler.transform(df[col])

    return df, scaler

def get_attrition_data(test_size=0.2, random_state=42):
    # FIX: Run data cleaning first so column arrays remain perfectly uniform across scripts
    df = clean_data(load_raw_data())

    # Map Attrition safely after capturing categorical variables if needed, or explicitly isolate target
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    
    df_encoded, encoders = encode_categorical(df)
    df_scaled, scaler = scale_numerical(df_encoded, target_cols=["Attrition", "MonthlyIncome"])
    
    X = df_scaled.drop(columns=["Attrition"])
    y = df_scaled["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, encoders, scaler

def get_salary_data(test_size=0.2, random_state=42):
    df = clean_data(load_raw_data())
    
    # FIX: Aligned target conversion pipeline with training layout sequence
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0}).fillna(0)

    df_encoded, encoders = encode_categorical(df)
    df_scaled, scaler = scale_numerical(df_encoded, target_cols=["Attrition", "MonthlyIncome"])

    X = df_scaled.drop(columns=["MonthlyIncome"])
    y = df_scaled["MonthlyIncome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, encoders, scaler

if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, encoders, scaler = get_attrition_data()
        print("Attrition data ready:", X_train.shape, X_test.shape)

        X_train2, X_test2, y_train2, y_test2, encoders2, scaler2 = get_salary_data()
        print("Salary data ready:", X_train2.shape, X_test2.shape)
    except Exception as e:
        print(f"Data preprocessing verification failed: {e}")