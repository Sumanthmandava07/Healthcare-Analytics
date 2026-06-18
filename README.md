# 🏥 Healthcare Predictive Analytics Dashboard

A machine learning-powered healthcare analytics platform designed to predict the risk of **Heart Disease** and **Diabetes** using clinical patient data. The system provides real-time risk assessment, feature importance analysis, model performance evaluation, and privacy-focused data handling.

## 🚀 Features

### Heart Disease Risk Prediction

* Predicts heart disease risk using clinical parameters.
* Supports multiple machine learning models:

  * Logistic Regression
  * Random Forest Classifier
* Provides probability-based risk scoring.

### Diabetes Risk Prediction

* Estimates diabetes risk using patient health indicators.
* Handles missing and zero-valued medical records through preprocessing and imputation.
* Generates interpretable risk assessments.

### Model Analytics

* ROC Curve Visualization
* Confusion Matrix Analysis
* Accuracy, Precision, Recall, and F1-Score Metrics
* Feature Importance Comparison

### Privacy & Ethics Module

* Client-side anonymization simulation
* SHA-256 based pseudonymization
* Data minimization principles
* Healthcare AI ethics awareness
* HIPAA & GDPR-inspired privacy practices

---

## 🛠️ Tech Stack

### Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js

### Backend

* Python
* Flask

### Machine Learning

* Scikit-learn
* Pandas
* NumPy

### Datasets

* UCI Heart Disease Dataset
* Pima Indians Diabetes Dataset

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Missing Value Handling
4. Feature Engineering
5. Data Normalization
6. Model Training
7. Model Evaluation
8. Feature Importance Analysis
9. Prediction Deployment

---

## 📈 Models Implemented

### Logistic Regression

* High Interpretability
* Fast Training
* Clinical Explainability

### Random Forest Classifier

* High Predictive Accuracy
* Robust Against Overfitting
* Feature Importance Support

---

## 🔒 Privacy & Ethical Considerations

This project emphasizes responsible AI practices:

* No patient data stored permanently
* Client-side anonymization demonstrations
* Pseudonymization using SHA-256 hashing
* Transparent model decision support
* Bias-aware healthcare analytics
* Ethical handling of sensitive medical information

---

## 📂 Project Structure

```text
Healthcare-Analytics/
│
├── backend/
│   ├── app.py
│   ├── train.py
│   ├── test_api.py
│   └── data/
│       ├── heart.csv
│       └── diabetes.csv
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── main.js
│
└── README.md
```

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Sumanthmandava07/Healthcare-Analytics.git
cd Healthcare-Analytics
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Backend

```bash
python app.py
```

### Launch Frontend

Open:

```text
frontend/index.html
```

in your browser.

---

## 📸 Dashboard Highlights

* Heart Disease Risk Assessment
* Diabetes Risk Prediction
* Model Performance Dashboard
* ROC Curve Visualization
* Feature Importance Analysis
* Privacy & Ethics Module

---
## 📸 Dashboard Screenshots

### ❤️ Heart Disease Risk Assessment

<p align="center">
  <img src="screenshots/heartdashboard.png" width="900">
</p>

Provides real-time heart disease risk prediction using Logistic Regression and Random Forest models with explainable risk factors.

---

### 🩺 Diabetes Risk Assessment

<p align="center">
  <img src="screenshots/diabetesdashboard.png" width="900">
</p>

Predicts diabetes risk based on clinical parameters and displays probability-based health insights.

---

### 📊 Model Analytics Dashboard

<p align="center">
  <img src="screenshots/analyticsdashboard.png" width="900">
</p>

Visualizes model performance using ROC Curves, Confusion Matrices, Accuracy, Precision, Recall, F1-Score, and Feature Importance Analysis.

---

### 🔒 Privacy & Ethics Dashboard

<p align="center">
  <img src="screenshots/ethicsdashboard.png" width="900">
</p>

Demonstrates privacy-preserving healthcare analytics through data anonymization, SHA-256 hashing, and ethical AI practices inspired by HIPAA and GDPR standards.

## 🎯 Future Enhancements

* Deep Learning Models
* Real-Time EHR Integration
* Explainable AI (SHAP/LIME)
* Multi-Disease Prediction
* Cloud Deployment
* Healthcare Chatbot Integration

---

## 👨‍💻 Author

**Sumanth Mandava**

* GitHub: https://github.com/Sumanthmandava07
* LinkedIn: [www.linkedin.com/in/sumanth-mandava](http://www.linkedin.com/in/sumanth-mandava)

---

⭐ If you found this project useful, consider giving it a star!
