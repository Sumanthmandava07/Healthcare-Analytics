import os
import sys
import json
import unittest

# Ensure backend directory is in the path
sys.path.append(os.path.dirname(__file__))

from app import app

class TestHealthcareAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_model_info_endpoint(self):
        """Verify model-info endpoint returns valid metadata structures."""
        response = self.app.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn("heart_disease", data)
        self.assertIn("diabetes", data)
        
        # Test heart disease schema details
        heart = data["heart_disease"]
        self.assertIn("scaler_params", heart)
        self.assertIn("features", heart)
        self.assertIn("lr", heart)
        self.assertIn("rf", heart)
        self.assertIn("stats", heart)
        self.assertEqual(len(heart["features"]), 13)

    def test_predict_endpoint_heart_lr(self):
        """Verify heart disease prediction with Logistic Regression."""
        payload = {
            "disease": "heart",
            "model": "lr",
            "patient_name": "Test Patient Name",
            "features": {
                "age": 54.0, "sex": 1.0, "cp": 3.0, "trestbps": 130.0, "chol": 240.0,
                "fbs": 0.0, "restecg": 2.0, "thalach": 150.0, "exang": 0.0,
                "oldpeak": 1.0, "slope": 1.0, "ca": 0.0, "thal": 3.0
            }
        }
        response = self.app.post("/api/predict", 
                                 data=json.dumps(payload),
                                 content_type="application/json")
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("risk_probability", data)
        self.assertIn("anonymized_id", data)
        # Verify pseudonymization is active (should not equal patient_name)
        self.assertNotEqual(data["anonymized_id"], "Test Patient Name")
        self.assertEqual(len(data["anonymized_id"]), 64) # SHA-256 length
        
        # Verify explainability structure is returned
        self.assertIn("contributions", data)
        self.assertIn("age", data["contributions"])
        self.assertIn("coefficient", data["contributions"]["age"])

    def test_predict_endpoint_diabetes_rf(self):
        """Verify diabetes prediction with Random Forest."""
        payload = {
            "disease": "diabetes",
            "model": "rf",
            "patient_name": "Jane Doe",
            "features": {
                "Pregnancies": 3.0, "Glucose": 120.0, "BloodPressure": 70.0,
                "SkinThickness": 20.0, "Insulin": 80.0, "BMI": 32.0,
                "DiabetesPedigreeFunction": 0.45, "Age": 33.0
            }
        }
        response = self.app.post("/api/predict", 
                                 data=json.dumps(payload),
                                 content_type="application/json")
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("risk_probability", data)
        self.assertIn("contributions", data)
        self.assertIn("Glucose", data["contributions"])
        self.assertIn("gini_importance", data["contributions"]["Glucose"])

    def test_predict_validation_failure(self):
        """Verify endpoint returns 400 Bad Request if features are missing or invalid."""
        # Missing feature "Age"
        invalid_payload = {
            "disease": "diabetes",
            "model": "lr",
            "features": {
                "Pregnancies": 3.0, "Glucose": 120.0, "BloodPressure": 70.0,
                "SkinThickness": 20.0, "Insulin": 80.0, "BMI": 32.0,
                "DiabetesPedigreeFunction": 0.45
            }
        }
        response = self.app.post("/api/predict", 
                                 data=json.dumps(invalid_payload),
                                 content_type="application/json")
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn("error", data)
        self.assertIn("details", data)
        self.assertTrue(any("Age" in d for d in data["details"]))

if __name__ == "__main__":
    unittest.main()
