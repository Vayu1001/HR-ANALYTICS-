from sklearn.linear_model import LogisticRegression
from preprocessing import get_attrition_data,get_salary_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

X_train,X_test,y_train,y_test,encoders,scaler = get_attrition_data()

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train,y_train)

log_pred = log_model.predict(X_test)

print("Accuracy :",accuracy_score(y_test,log_pred))
print("Confusion Matrix :\n",confusion_matrix(y_test,log_pred))
print("Classification Report :\n",classification_report(y_test,log_pred))


knn_model = KNeighborsClassifier()
knn_model.fit(X_train,y_train)

knn_pred = knn_model.predict(X_test)
print("Accuracy :",accuracy_score(y_test,knn_pred))
print("Confusion Matrix :\n",confusion_matrix(y_test,knn_pred))
print("Classification Report :\n",classification_report(y_test,knn_pred))

dt_model = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
dt_model.fit(X_train,y_train)

dt_pred = dt_model.predict(X_test)
print("Accuracy :",accuracy_score(y_test,dt_pred))
print("Confusion Matrix :\n",confusion_matrix(y_test,dt_pred))
print("Classification Report :\n",classification_report(y_test,dt_pred))

rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
rf_model.fit(X_train,y_train)

rf_pred = rf_model.predict(X_test)
print("Accuracy :",accuracy_score(y_test,rf_pred))
print("Confusion Matrix :\n",confusion_matrix(y_test,rf_pred))
print("Classification Report :\n",classification_report(y_test,rf_pred))



joblib.dump(log_model, os.path.join(MODELS_DIR, "logistic_model.pkl"))
joblib.dump(knn_model, os.path.join(MODELS_DIR,"knn_model.pkl"))
joblib.dump(encoders, os.path.join(MODELS_DIR,"encoder_log.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR,"scaler_log.pkl"))
joblib.dump(dt_model, os.path.join(MODELS_DIR,"decision_tree_model.pkl"))
joblib.dump(rf_model, os.path.join(MODELS_DIR,"random_forest_model.pkl"))
