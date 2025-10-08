# 🏭 Milling Machine Predictive Maintenance System

A comprehensive machine learning solution with a modern, industrial-style Streamlit dashboard for real-time predictive maintenance monitoring of milling machines.

## 📊 Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from Kaggle:

**Dataset Link:** [Predictive Maintenance Dataset (AI4I 2020)](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)

The dataset contains synthetic data that reflects real predictive maintenance scenarios encountered in industry. It consists of 10,000 data points with features including:
- Machine type (L/M/H quality variants)
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Machine failure indicators

## 🎯 Project Overview

This project implements an end-to-end machine learning solution for predictive maintenance:

1. **Data Analysis & Preprocessing** (Jupyter Notebook)
   - Exploratory Data Analysis (EDA)
   - Feature Engineering
   - Model Training & Evaluation
   - Model Selection & Optimization

2. **Interactive Dashboard** (Streamlit Application)
   - Real-time failure prediction
   - Interactive parameter adjustment
   - Real-time simulation mode
   - Visual analytics and reporting

## 🤖 Machine Learning Models

The project evaluates multiple classification models:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier (Best Performance)
- Gradient Boosting
- AdaBoost

**Best Model:** XGBoost with **99%+ accuracy** on the test set.

## 🎨 Design Theme

The interface features a professional industrial design with:
- **Dark Gray (#2C3E50)** - Primary background
- **Steel Blue (#34495E)** - Secondary elements
- **Safety Yellow (#F39C12)** - Highlights and alerts
- Clean, modern layout suitable for engineers and maintenance teams

## 📋 Features

### Input Parameters
- **Machine Type**: Quality variant (L/M/H - Low/Medium/High)
- **Torque (Nm)**: Operating torque of the milling machine
- **Rotational Speed (rpm)**: Spindle rotation speed
- **Process Temperature (K)**: Temperature during machining
- **Air Temperature (K)**: Ambient air temperature
- **Tool Wear (min)**: Tool wear time in minutes

### Derived Features (Automatically Calculated)
- **Temperature Difference**: Process temp - Air temp
- **Mechanical Power (W)**: (Torque × RPM × 2π) / 60

### Dashboard Components
1. **Real-time Parameter Monitoring** - Visual charts showing current operating conditions
2. **Health Status Indicators** - Color-coded status (Healthy/Caution/Failure Risk)
3. **Predictive Analytics** - ML-powered failure prediction with confidence levels
4. **Interactive Gauges** - Visual representation of key metrics
5. **Actionable Recommendations** - Context-aware maintenance suggestions
6. **Detailed Reports** - Comprehensive parameter analysis

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```powershell
pip install -r requirements.txt
```

### Run the Application

```powershell
streamlit run app.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

## 📊 How to Use

1. **Adjust Parameters**: Use the sliders in the left sidebar to input machine operating conditions
2. **View Derived Features**: Automatically calculated metrics are displayed in the sidebar
3. **Analyze Health**: Click the "ANALYZE MACHINE HEALTH" button to run the prediction
4. **Review Results**: 
   - View the machine status (Healthy/Caution/Failure Risk)
   - Check confidence levels
   - Review probability distribution
   - Read actionable recommendations
5. **Generate Reports**: View detailed parameter reports in the table format

## 🎯 Status Indicators

- **🟢 Healthy (Green)**: Machine operating optimally - continue normal operations
- **🟡 Caution (Yellow)**: Monitor closely - machine approaching operational limits
- **🔴 Failure Risk (Red)**: Immediate action required - high risk of equipment failure

## 🔧 Model Information

The system uses an XGBoost machine learning model (`xgboost_model.pkl`) trained on historical milling machine data to predict potential failures before they occur.

### Model Input Features:
1. Type (Machine quality variant)
2. Air temperature (K)
3. Process temperature (K)
4. Rotational speed (rpm)
5. Torque (Nm)
6. Tool wear (min)
7. Temperature difference (K) - Derived
8. Mechanical Power (W) - Derived

## 📁 Project Structure

```
Predicitive mantainence/
│
├── app.py                 # Main Streamlit application
├── xgboost_model.pkl      # Trained XGBoost model
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🛠️ Troubleshooting

### Model Not Found Error
Ensure `xgboost_model.pkl` is in the same directory as `app.py`

### Package Import Errors
Run: `pip install -r requirements.txt`

### Port Already in Use
Specify a different port: `streamlit run app.py --server.port 8502`

## 💡 Tips for Best Results

- Ensure input values are within realistic operating ranges
- Higher confidence levels (>85%) indicate more reliable predictions
- Monitor trends over time by testing different parameter combinations
- Use the recommendations section for maintenance planning

## 🔒 Safety Notes

⚠️ This system is designed as a monitoring tool and should not be the sole basis for critical safety decisions. Always follow your organization's safety protocols and maintenance procedures.

## 📞 Support

For issues or questions about the dashboard, please refer to the Streamlit documentation at https://docs.streamlit.io

---

**Developed for Industrial Equipment Monitoring | Powered by XGBoost & Streamlit**
