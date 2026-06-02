# Cold Case Risk Analytics Framework & Predictive AI Web Application

## 📌 Project Overview

This repository hosts an end-to-end Machine Learning, Explainable AI (XAI), Business Intelligence, and Streamlit deployment solution designed to analyze historical homicide patterns, profile unresolved case characteristics, and generate real-time cold case risk predictions.

The framework transitions seamlessly from exploratory notebook modeling to visualization and production-grade deployment:

1. **The Model Layer:** A Jupyter Notebook implementing both unsupervised and supervised machine learning workflows.
2. **The XAI Layer:** Global and local model explanations generated using SHAP (SHapley Additive exPlanations).
3. **The BI Layer:** A multi-page Power BI dashboard suite for interactive spatial and temporal crime analysis.
4. **The Web Layer:** A Streamlit application enabling real-time cold case risk prediction through an interactive interface.

---

## 🔑 Key Findings & Core Analytics Insights

Analysis of historical homicide records revealed several significant patterns and predictive indicators.

### 🚨 Macroscopic Realities

* 🔴 Nearly **1 in 3 homicide cases** remain unsolved.
* 🔴 The **victim-offender relationship** emerged as the strongest predictor of case resolution.
* 🔴 **Hawaii** demonstrated the highest baseline cold-case risk among all states analyzed.
* 🔴 Cases occurring in **December** exhibited the lowest overall clearance rates.

### 👥 Demographic Vulnerabilities & Solve Factors

* 🔴 Senior citizens recorded the lowest clearance rates, approximately **65%**.
* 🔴 Child victims showed the highest probability of case resolution, reaching nearly **90%**.

### 🛠️ Method Mechanics & Predictive Bounds

* 🔴 Cases involving an **Unknown Weapon** displayed the lowest likelihood of resolution.
* 🔴 The optimized **Random Forest Classifier** achieved an overall classification accuracy of **83.4%**.

---

## 🖥️ Interactive Analytics Dashboard

|                        Executive Overview                       |                       ML & SHAP Insights                      |
| :-------------------------------------------------------------: | :-----------------------------------------------------------: |
| ![Executive Overview](./dashboard/page1_executive_overview.png) | ![ML & SHAP Insights](./dashboard/page2_ml_shap_insights.png) |

|                          Geographic Intelligence                          |                          Pattern & Trend Analysis                         |
| :-----------------------------------------------------------------------: | :-----------------------------------------------------------------------: |
| ![Geographic Intelligence](./dashboard/page3_geographic_intelligence.png) | ![Pattern & Trend Analysis](./dashboard/page4_pattern_trend_analysis.png) |

---

## 🛠️ System Architecture
## 🛠️ Full-Stack System Architecture & Tech Stack

```text
[Raw Dataset] ➔ [Pandas Preprocessing & Engineering] ➔ [SMOTE Balancing] ➔ [Random Forest Classification]
                                                                                      │
  ┌─────────────────────────────────┬─────────────────────────────────────────────────┴────────────────────────────────────────┐
  ▼                                 ▼                                                                                          ▼
[SHAP Tree Inferences]     [Data Engine Output Exports]                                                               [Joblib Model Serialization]
  │                                 │                                                                                          │
  ▼                                 ▼                                                                                          ▼
[Global/Local Insights]    [Power BI Interactive Analytics]                                                           [Django Production Deployment]
                           - Executive Overview                                                                       - Manual Pipeline Feature Map
                           - ML & SHAP Explanations                                                                   - Real-Time Risk Diagnostics UI
                           - Geographic Hotspots
                           - Temporal Trend Slices

```

### Technology Stack

| Category            | Technologies             |
| ------------------- | ------------------------ |
| Programming         | Python                   |
| Data Processing     | Pandas, NumPy            |
| Machine Learning    | Scikit-Learn             |
| Imbalanced Learning | SMOTE (Imbalanced-Learn) |
| Explainable AI      | SHAP                     |
| Visualization       | Power BI                 |
| Deployment          | Streamlit                |
| Model Storage       | Joblib                   |

---

## 🤖 Machine Learning Pipeline

The project follows a structured five-stage workflow:

### Phase 1: Data Cleaning & Preparation

* Missing value treatment
* Feature engineering
* Target encoding
* Data transformation

### Phase 2: Feature Encoding

* One-Hot Encoding of categorical variables
* Creation of **91 model-ready features**

### Phase 3: K-Modes Clustering

* Identification of distinct cold-case profiles
* Segmentation of unsolved cases into operational clusters

### Phase 4: Supervised Learning

* SMOTE balancing of minority classes
* Random Forest model training
* Hyperparameter optimization
* Performance evaluation

### Phase 5: Explainable AI (XAI)

* Global feature importance analysis
* Local prediction explanations
* Top contributing risk-factor extraction using SHAP

---

## 📈 Model Performance

| Metric                   | Value                        |
| ------------------------ | ---------------------------- |
| Classification Accuracy  | **83.4%**                    |
| Number of Features       | **91**                       |
| Model Type               | **Random Forest Classifier** |
| Explainability Framework | **SHAP TreeExplainer**       |

### Most Influential Predictor

SHAP analysis identified **Unknown Relationship** as the strongest contributor to unresolved homicide cases, producing the largest average impact on model predictions.

---



* The final feature vector perfectly matches the model's training schema.

---

## 📁 Repository Structure

```text
├── app.py                              # Main Streamlit application

├── dashboard/
│   ├── cold_case_pattern_analysis_dashboard.pbix
│   ├── page1_executive_overview.png
│   ├── page2_ml_shap_insights.png
│   ├── page3_geographic_intelligence.png
│   └── page4_pattern_trend_analysis.png

├── models/
│   ├── cold_case_rf_model.pkl
│   └── feature_columns.pkl

├── notebooks/
│   └── cold_case_analysis.ipynb

├── outputs/
│   └── final_portfolio_export.csv

├── requirements.txt

└── README.md
```

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cold-case-risk-analytics.git
cd cold-case-risk-analytics
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas numpy scikit-learn joblib shap imbalanced-learn openpyxl
```

### 3. Launch the Streamlit Application

```bash
streamlit run app.py
```

### 4. Open in Browser

```text
http://localhost:8501
```

---

## 📊 Business Intelligence Dashboard

The Power BI dashboard suite provides:

* Executive-level KPI monitoring
* SHAP-based feature importance visualization
* Geographic hotspot identification
* State-wise clearance analysis
* Seasonal and temporal trend exploration
* Victim demographic breakdowns
* Resolution-rate benchmarking

---

## 🎯 Project Objectives

* Identify characteristics associated with unresolved homicide cases.
* Build predictive models capable of estimating case-resolution likelihood.
* Improve transparency through explainable AI techniques.
* Deliver actionable insights through interactive dashboards.
* Deploy a user-friendly prediction interface for real-time analysis.

---

## 📄 License

This project is intended for educational, research, and portfolio demonstration purposes. Please review the original dataset licensing terms before any commercial use.
