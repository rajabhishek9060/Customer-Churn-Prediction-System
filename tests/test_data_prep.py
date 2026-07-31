import os
import tempfile
import unittest
import pandas as pd
import numpy as np
from src.data_prep import (
    load_raw_data,
    clean_total_charges,
    drop_non_predictive,
    encode_target,
    encode_categoricals,
    preprocess,
)

class TestDataPrep(unittest.TestCase):
    def setUp(self):
        # Create a mock dataframe representing the raw data
        self.mock_data = pd.DataFrame({
            "customerID": ["1234-ABCD", "5678-EFGH", "9012-IJKL"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "Yes", "No"],
            "tenure": [12, 24, 6],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["Yes", "No", "No internet service"],
            "TechSupport": ["No", "Yes", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
            "MonthlyCharges": [29.85, 56.95, 18.85],
            "TotalCharges": ["29.85", " ", "113.1"],  # includes a space/empty value
            "Churn": ["No", "Yes", "No"]
        })

    def test_clean_total_charges(self):
        # Test that blank charges are converted to numeric and replaced by median
        df = self.mock_data.copy()
        cleaned_df = clean_total_charges(df)
        
        # " " should be coerced to NaN and filled with the median of [29.85, 113.1] which is 71.475
        expected_median = (29.85 + 113.1) / 2
        self.assertAlmostEqual(cleaned_df.loc[1, "TotalCharges"], expected_median)
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"]))

    def test_drop_non_predictive(self):
        # Test that customerID is dropped if present
        df = self.mock_data.copy()
        self.assertIn("customerID", df.columns)
        
        df_dropped = drop_non_predictive(df)
        self.assertNotIn("customerID", df_dropped.columns)
        self.assertIn("gender", df_dropped.columns)

    def test_encode_target(self):
        # Test Churn mapping Yes->1, No->0
        df = self.mock_data.copy()
        encoded_df = encode_target(df)
        
        self.assertEqual(encoded_df.loc[0, "Churn"], 0)
        self.assertEqual(encoded_df.loc[1, "Churn"], 1)
        self.assertEqual(encoded_df.loc[2, "Churn"], 0)

    def test_encode_categoricals(self):
        # Test encoding object/string columns to label-encoded integers
        df = self.mock_data.copy()
        df = clean_total_charges(df)
        df = drop_non_predictive(df)
        df = encode_target(df)
        
        encoded_df = encode_categoricals(df)
        
        # Verify columns that were string/object are now numeric
        for col in ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
                    "PaperlessBilling", "PaymentMethod"]:
            self.assertTrue(pd.api.types.is_integer_dtype(encoded_df[col]) or pd.api.types.is_numeric_dtype(encoded_df[col]))

    def test_preprocess_pipeline(self):
        # Test preprocessing end to end with temporary CSV files
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw_temp.csv")
            clean_path = os.path.join(tmpdir, "clean_temp.csv")
            
            # Save mock raw data to temp path
            self.mock_data.to_csv(raw_path, index=False)
            
            # Preprocess
            processed_df = preprocess(raw_path=raw_path, clean_path=clean_path)
            
            # Checks
            self.assertTrue(os.path.exists(clean_path))
            self.assertEqual(processed_df.shape[0], 3)
            self.assertNotIn("customerID", processed_df.columns)
            self.assertEqual(processed_df.loc[1, "Churn"], 1)

if __name__ == "__main__":
    unittest.main()
