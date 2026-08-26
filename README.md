<div align="center">
  <h1>📉 Customer Churn Prediction System</h1>
  <p>
    <strong>End-to-end customer churn prediction system featuring data preprocessing, Random Forest model training, and a Streamlit dashboard.</strong>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version" />
    <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg" alt="Scikit-Learn" />
    <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg" alt="Streamlit" />
    <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg" alt="Pandas" />
  </p>
</div>

---

## 📖 Overview

**Customer Churn Prediction System** is a machine learning-powered application designed to predict the likelihood of customers leaving a service. By inputting customer data, the system deploys a trained Random Forest model to evaluate churn risk, enabling proactive retention strategies.

### 🌟 Key Features

- **End-to-End Pipeline:** Seamless integration of data preprocessing, model training, evaluation, and deployment.
- **Robust Machine Learning:** Utilizes a highly accurate Random Forest Classifier trained on comprehensive customer datasets.
- **Interactive Dashboard:** Built on Streamlit, offering a clean, user-friendly interface for dynamic predictions.
- **Data Insights:** Efficient data processing and feature engineering using Pandas.

## 🏗️ Architecture

The system is structured into key modular components:
1. `app.py`: The presentation layer providing an interactive Streamlit web dashboard for real-time churn predictions.
2. `src/train.py`: The model training module that orchestrates data splitting, Random Forest training, evaluation, and saving the serialized model.
3. `src/data_prep.py`: The data engineering module responsible for cleaning, imputing, and encoding the raw dataset.

---

## 🚀 Quickstart Guide

Want to run the Customer Churn Prediction System locally? Follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/amritrajrajput/Customer-Churn-Prediction-System.git
cd Customer-Churn-Prediction-System
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
The pre-trained model is included, but you can retrain it by running:
```bash
python src/train.py
```

### 4. Run the Streamlit App
Launch the interactive dashboard:
```bash
streamlit run app.py
```
*The application will automatically open in your browser at `http://localhost:8501/`.*

---

## 💡 Why This Project Stands Out

This project demonstrates proficiency in building **End-to-End Machine Learning** applications. It showcases the ability to take a raw dataset, engineer features, train a robust model (Random Forest), and deploy it within a **clean, user-facing full-stack Streamlit dashboard**. It bridges the gap between data science and functional software engineering.

---

<p align="center">
  <i>Built with ❤️. If you find this project interesting, feel free to check out the <a href="https://github.com/amritrajrajput/Customer-Churn-Prediction-System">GitHub Repository</a> and give it a ⭐!</i>
</p>
