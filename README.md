# IIOT4-MACHINE-LEARNING
# 🏙️ SMART CITY ANALYTIC SYSTEM 

> **Neural Network Powered Urban Intelligence System v2.0**  
> Real-time predictions for Traffic, Crime, House Prices & Citizen Segmentation

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Max%20Accuracy-96.89%25-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

Smart City Analytic System is a Machine Learning based system that analyzes urban data and provides intelligent predictions across multiple domains.

---

## 👨‍💻 Team — Neural Network Team

| Name | Roll Number | Role |
|------|-------------|------|
| Yash Sharma | BCS2024221 | 🏆 Team Leader |
| Manjesh Yadav | BCS2024213 | Member |
| Mayuri Singh | BCS2024220 | Member |
| Mohammad Anas Usmani | BCS2024224 | Member |
| Lucky Gangwar | BCS2024233 | Member |
| Abhinav Tiwari | BCS2024243 | Member |
| Mohd Uzair | BCS2024255 | Member |
| Vansh Saxena | BCS2024256 | Member |
| Aanya Srivastava | BCS2024257 | Member |
| Manan Vashishtha | BCS2024263 | Member |

---

## 🤖 ML Models

### 1. 🏠 House Price Prediction
- **Algorithm:** Linear Regression
- **Features:** Square Footage, Bedrooms, Bathrooms, Lot Size, Neighborhood Quality
- **Output:** Predicted House Price (USD)

### 2. 🚦 Traffic Analyzer
- **Algorithm:** SVM (Support Vector Machine)
- **Accuracy:** 96.89%
- **Features:** City, Area Code, Population Density, Vehicles/Hr, Traffic Signals, Weather, Time of Day, Day Type, Road Type
- **Output:** HIGH TRAFFIC / LOW TRAFFIC

### 3. 🚨 Crime Risk Analyzer
- **Algorithm:** Logistic Regression
- **Features:** City, Area Code, Population, Unemployment Rate, Police Stations, Time, Victim Age, Victim Gender, Weapon Used, Past Crime Rate
- **Output:** HIGH RISK / LOW RISK

### 4. 👥 Citizen Segmentation
- **Algorithm:** SVM
- **Accuracy:** 85%+
- **Features:** City, Age, Monthly Income, Education Level, Employment Type, Digital Usage Hours, Household Size, Area Type, Govt Scheme Usage
- **Output:** High Value Citizen / Low Value Citizen

---

## 📁 Project Structure

```
IIOT4-MACHINE-LEARNING/
│
├── Backend/
│   ├── app.py                  # Flask REST API
│   ├── requirement.txt         # Python dependencies
│   └── models/
│       ├── citizen_segment_model.pkl
│       ├── crime_model.pkl
│       ├── crime_scaler.pkl
│       ├── crime_label_encoders.pkl
│       ├── house_price_model.pkl
│       ├── house_price_features.pkl
│       ├── traffic_model.pkl
│       ├── traffic_scaler.pkl
│       ├── traffic_features.pkl
│       └── label_encoders.pkl
│
├── Frontend/
│   └── index.html (Main Dashboard UI)
│
├── CITIZEN SEGMENTATION/
│   ├── Citizen_Segmentation.ipynb
│   └── citizen.csv
│
├── CRIME RISK/
│   ├── CRIME.ipynb
│   └── crime dataset.csv
│
├── HOUSE PRICE PREDICTION/
│   ├── HousePricePredictionModel.ipynb
│   └── house_price_regression_dataset.csv
│
├── TRAFFIC PREDICTION/
│   ├── TRAFFIC.ipynb
│   └── traffic.CSV
│
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Install Dependencies
```bash
cd Backend
pip install -r requirement.txt
```

### Step 2 — Start Backend Server
```bash
python app.py
```
Server will start at: `http://localhost:5000`

### Step 3 — Open Frontend
Open `Frontend/2model_workinghtml.html` in your browser.

> ⚠️ Make sure backend is running before opening the frontend!

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check & model status |
| `/predict/house_price` | POST | Predict house price |
| `/predict/traffic` | POST | Predict traffic status |
| `/predict/crime` | POST | Predict crime risk |
| `/predict/citizen` | POST | Predict citizen segment |

---

## 📦 Requirements

```
flask
flask-cors
joblib
numpy
pandas
scikit-learn
```

---

## 📊 Dataset Info

| Module | Dataset | Records |
|--------|---------|---------|
| Traffic | traffic.CSV | Indian city traffic data |
| Crime | crime dataset.csv | Urban crime statistics |
| House Price | house_price_regression_dataset.csv | Property data |
| Citizen | citizen.csv | Urban citizen profiles |

---

## 🏫 Institution

**IIoT 4th Semester Project**  
B.Tech Computer Science Engineering  

---

> *"Intelligent cities start with intelligent data."* 🌆
