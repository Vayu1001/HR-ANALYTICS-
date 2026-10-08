# HR Analytics: Employee Attrition & Salary Prediction

A machine learning project that analyzes employee data to predict **attrition** (will an employee leave?) and **salary**, served through a FastAPI backend and an interactive Streamlit dashboard.

## 📌 Project Overview

This project uses an IBM-style HR Analytics Employee Attrition dataset to:
- Explore HR data through visualizations (EDA)
- Predict employee attrition using classification models
- Predict employee salary using regression
- Store and retrieve data via a MySQL database
- Serve predictions and data through a FastAPI REST API
- Present everything through an interactive Streamlit web app

## 🎯 Features

- **Data Storage**: Employee data stored in MySQL, queried via `mysql-connector-python`
- **EDA**: Visualizations with Matplotlib & Seaborn (attrition trends, salary distribution, correlations)
- **Preprocessing**: Encoding categorical variables, scaling numerical features
- **Classification Models**: Logistic Regression, Decision Tree, Random Forest, and KNN to predict attrition
- **Regression Model**: Linear Regression to predict employee salary based on role, experience, and performance
- **Model Comparison**: Accuracy, confusion matrices, and feature importances across classifiers
- **REST API**: FastAPI backend exposing employee data, predictions, model performance, and visualizations
- **Interactive App**: Streamlit dashboard to explore data and get live predictions

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | Scikit-learn (Logistic Regression, Decision Tree, Random Forest, KNN, Linear Regression) |
| Database | MySQL, mysql-connector-python |
| API | FastAPI, Uvicorn, Pydantic |
| App/UI | Streamlit |

## 📂 Project Structure

```
HR Analytics with API/
│
├── Backend/
│   ├── main.py                  # FastAPI app and route definitions
│   ├── db_connection.py         # MySQL connection and insert/query helpers
│   ├── preprocessing.py         # Cleaning, encoding, and scaling utilities
│   ├── schemas.py                # Pydantic request/response models
│   ├── ml_utils.py              # Model loading, caching, and prediction logic
│   ├── train_classification.py  # Trains and saves attrition classifiers
│   └── train_regression.py      # Trains and saves the salary regressor
├── Frontend/
│   └── app.py                   # Streamlit dashboard
├── Database/
│   └── hr_data.csv.csv          # Source employee dataset
├── models/                      # Trained models, encoders, and scalers (.pkl)
├── requirement.txt
└── Readme.md
```

## 🚀 Getting Started

### 1. Set up the project folder
```bash
cd "HR Analytics with API"
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirement.txt
```

### 4. Set up MySQL
- Create a database named `hr_analytics`
- Create an `employees` table whose columns match `Database/hr_data.csv.csv`
- Load the CSV data into that table
- Update the credentials in `Backend/db_connection.py` (`DB_CONFIG`) to match your local MySQL setup — **do not commit real credentials to version control**

### 5. Train the models (optional — pretrained models are already included in `models/`)
```bash
cd Backend
python train_classification.py
python train_regression.py
```

### 6. Run the FastAPI backend
```bash
cd Backend
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

### 7. Run the Streamlit app
```bash
cd Frontend
streamlit run app.py
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/employees` | List employees (paginated) |
| GET | `/employees/{employee_number}` | Get a single employee |
| POST | `/employees` | Add a new employee record |
| POST | `/predict/attrition` | Predict attrition risk (choice of model) |
| POST | `/predict/salary` | Predict monthly income |
| GET | `/model-performance` | Accuracy/R² metrics, confusion matrices, feature importances |
| GET | `/visualizations/{graph_type}` | Returns a PNG chart (e.g. `age_distribution`, `correlation_heatmap`) |

## 📊 Dataset

Based on the [IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — contains employee demographic, job, and performance data with an attrition label and salary-related fields.

## ⚠️ Security Note

`Backend/db_connection.py` currently contains a hardcoded database password. Before sharing this repo or deploying it, move credentials into environment variables (e.g. via a `.env` file with `python-dotenv`) and add that file to `.gitignore`.

## 📈 Status

- [x] Load data into MySQL and connect via Python
- [x] Perform EDA with visualizations
- [x] Encode categorical features, scale numerical features
- [x] Train and compare Logistic Regression, Decision Tree, Random Forest, KNN for attrition
- [x] Train regression model for salary prediction
- [x] Build Streamlit app with Data Explorer, Attrition Predictor, and Salary Predictor tabs
- [x] Add model evaluation metrics and visualizations to the app
- [x] Expose all functionality through a FastAPI backend
- [ ] Move database credentials to environment variables
- [ ] Add automated tests for API endpoints

## 📝 License

This project is for learning purposes.