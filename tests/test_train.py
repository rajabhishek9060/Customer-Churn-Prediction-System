import os
import json
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.train import (
    load_data,
    split_data,
    train_model,
    evaluate_model,
    save_model,
    save_feature_importance,
    run_training_pipeline,
)

class TestTrain(unittest.TestCase):
    def setUp(self):
        # Create a mock cleaned dataset
        # In a real run, all columns are numeric
        self.mock_data = pd.DataFrame({
            "tenure": [12, 24, 6, 36, 48, 2, 60, 72, 18, 5],
            "MonthlyCharges": [29.85, 56.95, 18.85, 99.35, 104.9, 20.0, 85.0, 115.0, 45.0, 25.0],
            "Contract": [0, 1, 2, 0, 2, 0, 1, 2, 0, 0],
            "Churn": [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
        })

    def test_load_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "clean_temp.csv")
            self.mock_data.to_csv(csv_path, index=False)
            
            X, y = load_data(csv_path)
            self.assertEqual(X.shape, (10, 3))
            self.assertEqual(y.shape, (10,))
            self.assertNotIn("Churn", X.columns)
            self.assertEqual(y.name, "Churn")

    def test_split_data(self):
        X = self.mock_data.drop(columns=["Churn"])
        y = self.mock_data["Churn"]
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        self.assertEqual(len(X_train), 8)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(y_train), 8)
        self.assertEqual(len(y_test), 2)

    def test_train_model(self):
        X = self.mock_data.drop(columns=["Churn"])
        y = self.mock_data["Churn"]
        clf = train_model(X, y, n_estimators=10)
        self.assertIsInstance(clf, RandomForestClassifier)
        # Model should be fitted
        self.assertTrue(hasattr(clf, "classes_"))

    def test_evaluate_model(self):
        X = self.mock_data.drop(columns=["Churn"])
        y = self.mock_data["Churn"]
        clf = train_model(X, y, n_estimators=10)
        
        metrics = evaluate_model(clf, X, y)
        self.assertIn("accuracy", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        for key in ["accuracy", "recall", "f1"]:
            self.assertIsInstance(metrics[key], float)
            self.assertTrue(0.0 <= metrics[key] <= 1.0)

    def test_save_model(self):
        X = self.mock_data.drop(columns=["Churn"])
        y = self.mock_data["Churn"]
        clf = train_model(X, y, n_estimators=10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pkl")
            save_model(clf, model_path)
            self.assertTrue(os.path.exists(model_path))

    def test_save_feature_importance(self):
        X = self.mock_data.drop(columns=["Churn"])
        y = self.mock_data["Churn"]
        clf = train_model(X, y, n_estimators=10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            imp_path = os.path.join(tmpdir, "feature_importance.json")
            save_feature_importance(clf, X.columns.tolist(), imp_path)
            
            self.assertTrue(os.path.exists(imp_path))
            with open(imp_path, "r") as f:
                data = json.load(f)
            
            # Check keys match features and are sorted descending
            self.assertEqual(len(data), 3)
            self.assertIn("tenure", data)
            values = list(data.values())
            self.assertTrue(all(values[i] >= values[i+1] for i in range(len(values)-1)))

    def test_run_training_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_clean_csv = os.path.join(tmpdir, "clean_temp.csv")
            temp_model_path = os.path.join(tmpdir, "model_temp.pkl")
            temp_feature_imp = os.path.join(tmpdir, "feat_temp.json")
            
            self.mock_data.to_csv(temp_clean_csv, index=False)
            
            with patch("src.train.CLEAN_CSV", temp_clean_csv), \
                 patch("src.train.MODEL_PATH", temp_model_path), \
                 patch("src.train.FEATURE_IMP_PATH", temp_feature_imp):
                
                metrics = run_training_pipeline()
                
                self.assertTrue(os.path.exists(temp_model_path))
                self.assertTrue(os.path.exists(temp_feature_imp))
                self.assertIn("accuracy", metrics)
                self.assertIn("recall", metrics)
                self.assertIn("f1", metrics)

if __name__ == "__main__":
    unittest.main()
