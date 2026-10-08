from sklearn.linear_model import LinearRegression
import joblib
import os
from preprocessing import get_salary_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

X_train, X_test, y_train, y_test, encoders, scaler = get_salary_data()

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)

print("Linear Regression:")
print("MAE :", mean_absolute_error(y_test, lin_pred))
print("MSE :", mean_squared_error(y_test, lin_pred))
print("R2 Score :", r2_score(y_test, lin_pred))

joblib.dump(lin_model, os.path.join(MODELS_DIR,"Linear_model.pkl"))
joblib.dump(encoders, os.path.join(MODELS_DIR,"encoder_lin.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR,"scaler_lin.pkl"))