# 🩺 Data Doctor AI

### Universal Data Analytics + Machine Learning Platform

**Upload your data → understand it → analyze it → predict what may happen → get practical business actions.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://data-doctor-ai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit--learn](https://img.shields.io/badge/ML-Scikit--learn-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 📌 Overview

Data Doctor AI is an end-to-end **Data Analytics + Machine Learning** platform designed for business datasets in CSV and Excel format.

Instead of building a separate dashboard for every dataset, the system automatically:

1. Understands the uploaded schema
2. Detects data types and business fields
3. Profiles data quality
4. Identifies useful relationships
5. Generates analytical views and business KPIs
6. Creates interactive visualizations
7. Identifies suitable machine-learning problems
8. Trains and compares multiple ML models where appropriate
9. Detects unusual patterns
10. Produces business-focused insights and recommendations

The same platform can adapt to structured datasets such as:

- Retail / Grocery
- Restaurant
- Healthcare
- Finance
- HR
- E-commerce
- Operations
- Other structured business datasets

---

# 🎯 Problem Statement

Traditional analytics workflows often require:

**Raw file → manual cleaning → manual profiling → manual dashboard → manual ML → manual interpretation**

Data Doctor AI combines these steps into one workflow.

The goal is to reduce the gap between:

**Raw Business Data**

and

**Actionable Business Decisions**

---

# 🏗️ End-to-End Architecture

```mermaid
flowchart TD

A[CSV / Excel Dataset] --> B[Schema & Field Discovery]

B --> C[Data Cleaning & Quality Check]

C --> D[Business Meaning Detection]

D --> E[Data Analysis & BI]

E --> E1[KPIs]
E --> E2[Trends]
E --> E3[Segmentation]
E --> E4[Relationships]
E --> E5[Visual Analysis]

D --> F[ML Problem Discovery]

F --> G{Suitable ML Problem?}

G -->|Yes| H[Multiple Candidate Models]
G -->|No| I[Explain Why ML Is Not Appropriate]

H --> J[Validation & Model Comparison]

J --> K[Predictions / Forecasts / Anomalies]

E --> L[Business Insights]

K --> L

L --> M[Business Recommendations]

M --> N[Business Owner Dashboard]
M --> O[Data / Power BI Analyst Workspace]

