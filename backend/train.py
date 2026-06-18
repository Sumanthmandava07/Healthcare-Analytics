import os
import json
import urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# Dataset URLs
HEART_DISEASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

# Local directories
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "model_data.json")

os.makedirs(DATA_DIR, exist_ok=True)

# Helper function to download file
def download_file(url, filepath):
    print(f"Downloading {url} to {filepath}...")
    try:
        # Avoid SSL issues by defining a user agent and disabling certificate verification if needed
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(filepath, 'wb') as out_file:
                out_file.write(response.read())
        print("Download successful.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False
    return True

def get_heart_disease_data():
    filepath = os.path.join(DATA_DIR, "heart.csv")
    if not os.path.exists(filepath):
        success = download_file(HEART_DISEASE_URL, filepath)
        if not success:
            print("Using mock/simulated heart disease dataset due to download failure.")
            return get_mock_heart_data()
            
    # Cleveland dataset column names
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    try:
        df = pd.read_csv(filepath, header=None, names=columns, na_values="?")
        # Impute missing values
        df["ca"] = df["ca"].fillna(df["ca"].median())
        df["thal"] = df["thal"].fillna(df["thal"].median())
        # Binarize target: 0 is healthy, 1+ is heart disease
        df["target"] = (df["target"] > 0).astype(int)
        return df
    except Exception as e:
        print(f"Error reading heart dataset: {e}. Generating mock data.")
        return get_mock_heart_data()

def get_diabetes_data():
    filepath = os.path.join(DATA_DIR, "diabetes.csv")
    if not os.path.exists(filepath):
        success = download_file(DIABETES_URL, filepath)
        if not success:
            print("Using mock/simulated diabetes dataset due to download failure.")
            return get_mock_diabetes_data()
            
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "target"
    ]
    try:
        df = pd.read_csv(filepath, header=None, names=columns)
        # In Pima dataset, 0 is missing for many physiological variables
        cols_with_zero_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in cols_with_zero_missing:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
        return df
    except Exception as e:
        print(f"Error reading diabetes dataset: {e}. Generating mock data.")
        return get_mock_diabetes_data()

def get_mock_heart_data(num_samples=300):
    np.random.seed(42)
    age = np.random.normal(54, 9, num_samples).clip(29, 77)
    sex = np.random.binomial(1, 0.67, num_samples)
    cp = np.random.choice([0, 1, 2, 3], num_samples, p=[0.47, 0.16, 0.28, 0.09])
    trestbps = np.random.normal(131, 17, num_samples).clip(94, 200)
    chol = np.random.normal(246, 51, num_samples).clip(126, 564)
    fbs = np.random.binomial(1, 0.15, num_samples)
    restecg = np.random.choice([0, 1, 2], num_samples, p=[0.49, 0.01, 0.50])
    thalach = np.random.normal(149, 23, num_samples).clip(71, 202)
    exang = np.random.binomial(1, 0.32, num_samples)
    oldpeak = np.random.exponential(1.0, num_samples).clip(0, 6.2)
    slope = np.random.choice([0, 1, 2], num_samples, p=[0.46, 0.46, 0.08])
    ca = np.random.choice([0, 1, 2, 3], num_samples, p=[0.58, 0.21, 0.12, 0.09])
    thal = np.random.choice([3, 6, 7], num_samples, p=[0.55, 0.06, 0.39])
    
    prob = (
        (age - 50)/50 + sex*0.5 + (cp == 0)*(-0.8) + (cp > 0)*0.5 + 
        (trestbps - 120)/100 + (chol - 200)/300 - (thalach - 150)/150 + 
        exang*0.8 + oldpeak*0.6 + ca*0.7 + (thal == 7)*0.8
    )
    prob_sig = 1 / (1 + np.exp(-prob))
    target = np.random.binomial(1, prob_sig)
    
    return pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": target
    })

def get_mock_diabetes_data(num_samples=768):
    np.random.seed(42)
    pregnancies = np.random.poisson(3.8, num_samples).clip(0, 17)
    glucose = np.random.normal(121, 30, num_samples).clip(44, 199)
    bp = np.random.normal(72, 12, num_samples).clip(24, 122)
    skin = np.random.normal(29, 8, num_samples).clip(7, 99)
    insulin = np.random.normal(155, 85, num_samples).clip(14, 846)
    bmi = np.random.normal(32.4, 6.9, num_samples).clip(18.2, 67.1)
    dpf = np.random.exponential(0.47, num_samples).clip(0.08, 2.42)
    age = np.random.normal(33.2, 11.7, num_samples).clip(21, 81)
    
    prob = (
        (glucose - 100)/50 + (bmi - 25)/15 + (age - 30)/40 + 
        (pregnancies - 3)*0.1 + (bp - 70)/100 + dpf*0.8 + (insulin - 100)/500
    )
    prob_sig = 1 / (1 + np.exp(-prob))
    target = np.random.binomial(1, prob_sig)
    
    return pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age, "target": target
    })

def train_and_eval(df, numeric_cols, dataset_name):
    X = df[numeric_cols]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    
    # Random Forest
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
    
    # Save model and scaler objects for the Flask backend
    clean_name = dataset_name.lower().replace(" ", "_")
    joblib.dump(scaler, os.path.join(DATA_DIR, f"{clean_name}_scaler.joblib"))
    joblib.dump(lr, os.path.join(DATA_DIR, f"{clean_name}_lr.joblib"))
    joblib.dump(rf, os.path.join(DATA_DIR, f"{clean_name}_rf.joblib"))
    
    # Metrics
    lr_metrics = {
        "accuracy": float(accuracy_score(y_test, lr_preds)),
        "precision": float(precision_score(y_test, lr_preds)),
        "recall": float(recall_score(y_test, lr_preds)),
        "f1": float(f1_score(y_test, lr_preds)),
        "auc": float(roc_auc_score(y_test, lr_probs))
    }
    
    rf_metrics = {
        "accuracy": float(accuracy_score(y_test, rf_preds)),
        "precision": float(precision_score(y_test, rf_preds)),
        "recall": float(recall_score(y_test, rf_preds)),
        "f1": float(f1_score(y_test, rf_preds)),
        "auc": float(roc_auc_score(y_test, rf_probs))
    }
    
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    
    step_lr = max(1, len(fpr_lr) // 20)
    step_rf = max(1, len(fpr_rf) // 20)
    
    lr_roc = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr_lr[::step_lr], tpr_lr[::step_lr])]
    rf_roc = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr_rf[::step_rf], tpr_rf[::step_rf])]
    lr_roc.append({"fpr": 1.0, "tpr": 1.0})
    rf_roc.append({"fpr": 1.0, "tpr": 1.0})
    
    cm_lr = confusion_matrix(y_test, lr_preds).tolist()
    cm_rf = confusion_matrix(y_test, rf_preds).tolist()
    
    lr_coefs = lr.coef_[0].tolist()
    rf_importances = rf.feature_importances_.tolist()
    
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "all_mean": float(df[col].mean()),
            "healthy_mean": float(df[df["target"] == 0][col].mean()),
            "sick_mean": float(df[df["target"] == 1][col].mean()),
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }
        
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    
    return {
        "scaler_params": scaler_params,
        "features": numeric_cols,
        "lr": {
            "metrics": lr_metrics,
            "coefficients": lr_coefs,
            "intercept": float(lr.intercept_[0]),
            "roc": lr_roc,
            "confusion_matrix": cm_lr
        },
        "rf": {
            "metrics": rf_metrics,
            "importances": rf_importances,
            "roc": rf_roc,
            "confusion_matrix": cm_rf
        },
        "stats": stats
    }

def main():
    print("Loading data...")
    heart_df = get_heart_disease_data()
    diabetes_df = get_diabetes_data()
    
    heart_features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    diabetes_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    
    print("Training heart disease models...")
    heart_results = train_and_eval(heart_df, heart_features, "Heart Disease")
    
    print("Training diabetes models...")
    diabetes_results = train_and_eval(diabetes_df, diabetes_features, "Diabetes")
    
    output_data = {
        "heart_disease": heart_results,
        "diabetes": diabetes_results
    }
    
    print(f"Saving model data to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
