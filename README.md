# Customer Churn Prediction System

A machine learning-powered system designed to predict customer churn risk using a Random Forest Classifier trained on the Telco Customer Churn dataset. The system features a modular data preprocessing pipeline, automated model training, a robust unit test suite, and an interactive Streamlit web dashboard.

---

## 📁 Directory Structure

```text
Customer Churn Prediction System/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
│   └── cleaned_churn.csv                     # Processed dataset for model consumption
├── src/
│   ├── __init__.py
│   ├── data_prep.py                          # Data cleaning, imputation, and encoding
│   └── train.py                              # Classifier training, evaluation, and serialization
├── tests/
│   ├── __init__.py
│   ├── test_data_prep.py                     # Unit tests for data preprocessing
│   └── test_train.py                         # Unit tests for the training pipeline
├── app.py                                    # Streamlit web dashboard
├── model.pkl                                 # Serialized Random Forest model
├── requirements.txt                          # Python dependencies list
└── README.md                                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup
Create a Python virtual environment and install the required dependencies:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Preprocessing & Training
Run the data pipeline to prepare the dataset and train the machine learning model:
```powershell
# Run data preprocessing
python src/data_prep.py

# Train the Random Forest Classifier
python src/train.py
```
This generates the serialized model (`model.pkl`) and extracts model feature importances saved to `artifacts/feature_importance.json`.

---

## 🖥️ Running the Web App

Launch the interactive Streamlit dashboard to predict customer churn risks:
```powershell
python -m streamlit run app.py
```
Once started, open [http://localhost:8501](http://localhost:8501) in your browser. The app allows you to configure a customer's profile in the sidebar and dynamically predict their churn probability.

---

## 🧪 Running Unit Tests

A comprehensive unit test suite is included to verify all preprocessing and model training code without affecting the production datasets or saved model:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📊 Core Technologies
- **Python 3.x**
- **Scikit-Learn** (Random Forest Classifier)
- **Pandas & NumPy** (Data processing)
- **Streamlit** (Web dashboard interface)
- **Matplotlib** (Feature driver charting)
- **Unittest** (Testing framework)
