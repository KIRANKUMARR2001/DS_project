# Predictive Maintenance for Industrial Machinery | Python, RandomForest,XGBoost, Pandas, Scikit-Learn
<img width="857" height="760" alt="image" src="https://github.com/user-attachments/assets/fbefc7a8-38af-479f-909a-45dc7fa48373" />
[App link]([https://dsproject-gtqjspyaxhsevqbvceopxf.streamlit.app/])

### **Overview**
This project is an end-to-end Machine Learning pipeline that predicts catastrophic machinery failures based on real-time sensor telemetry. It uses an **XGBoost** classification model deployed as a live **Streamlit** web application.

### **The Business Problem**
In industrial manufacturing, unexpected equipment breakdowns cause massive financial losses. The goal of this project is to catch impending failures *before* they happen. 

Dealing with predictive maintenance means handling highly imbalanced data (equipment operates normally 97% of the time and fails only 3% of the time). This project specifically addresses the **Precision-Recall Tradeoff**. 

We evaluated both Random Forest and XGBoost. Because the business cost of a missed machine failure (expensive factory downtime) is significantly higher than the cost of a false alarm (a quick routine inspection), the final model utilizes **XGBoost with hyperparameter weighting (`scale_pos_weight`) to optimize for Recall (80%)**, ensuring the maximum number of true failures are caught.

### **Features Engineered**
To improve the model's predictive power, raw sensor data was transformed into physical mechanical indicators. 

*   **Power Stress:** `Torque [Nm]` × `Rotational speed [rpm]`
*   **Temperature Differential:** `Process temperature [K]` - `Air temperature [K]`

### **Tech Stack**
*   **Language:** Python
*   **Machine Learning:** XGBoost, Scikit-Learn, Pandas
*   **Frontend Deployment:** Streamlit
*   **Model Serialization:** Joblib


