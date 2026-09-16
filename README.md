🩺 Data Doctor AI

Universal Data Analytics + Machine Learning Platform

Upload business data → understand it automatically → analyze it deeply → build dashboards → discover relationships → train suitable ML models → get practical business actions.



Live Demo: https://data-doctor-ai.streamlit.app/

GitHub: https://github.com/Ujjwal-005/data-doctor-ai

📌 Project Overview

Data Doctor AI is a universal tabular analytics and machine-learning workspace designed to work with structured business datasets without requiring a fixed schema.

Instead of building a separate dashboard for every dataset, the application first discovers what the uploaded data contains and then adapts the analysis around the available fields.

It supports datasets from areas such as:

Grocery / Retail

Restaurants

Healthcare

Finance

HR

E-commerce

Logistics

Other structured business datasets

The same uploaded data can be viewed in two ways:

👤 Business Owner

A simple decision dashboard focused on revenue, profit, trends, important items, unusual activity, predictions and recommended actions.

📊 Data / Power BI Analyst

A deeper analytical workspace covering profiling, cleaning, visual analysis, machine learning, relational modeling, SQL suggestions, business storytelling and export.

🧠 How Data Doctor AI Works

flowchart TD
    A[Upload CSV / Excel] --> B[Read & Clean Data]
    B --> C[Automatic Schema & Field Discovery]
    C --> D[Domain Detection]
    D --> E[Business Metrics & Data Profiling]
    E --> F[Interactive Visual Analysis]
    E --> G[ML Problem Detection]
    E --> H[Relationship Detection]
    G --> I[Train Suitable ML Models]
    I --> J[Compare Models with Validation]
    J --> K[Forecast / Classification / Regression / Anomalies / Clusters]
    H --> L[SQL & Relational Model]
    K --> M[Business Insights]
    L --> M
    F --> M
    M --> N[Business Owner Dashboard]
    M --> O[Analyst Workspace]
    M --> P[Recommendations & Next Actions]

🔍 Core Pipeline

Raw Data
   ↓
Data Cleaning
   ↓
Automatic Field Discovery
   ↓
Domain & Dataset Understanding
   ↓
Data Analysis + Business Metrics
   ↓
Interactive Visualizations
   ↓
Relationship / SQL Modeling
   ↓
Machine Learning
   ↓
Validation & Model Comparison
   ↓
Insights + Recommendations

📊 Data Analysis Capabilities

Data Doctor AI performs analysis before applying machine learning.

1. Data Quality & Profiling

The Analyst workspace reports:

Number of rows and columns

Column names

Data types

Missing values

Missing percentage

Duplicate rows

Unique-value counts

Basic field statistics

Dataset date range when available

2. Automatic Field Discovery

The system maps different business naming styles to common analytical roles.

For example:

Dataset column

Detected role

Invoice ID, Order ID, Transaction ID

ID / Record Identifier

Date, Order Date, Sale Date

Date / Time

Product, Product Name, Item Name

Item / Product / Service

Qty, Quantity, Units Sold

Quantity / Count

Sales, Revenue, Total

Revenue / Sales

Profit, Gross Income, Net Profit

Profit / Income

Cost, COGS, Unit Cost

Cost

Branch, Store, Location

Location / Branch

The original fields are preserved so the analyst can still inspect the source dataset.

3. Business Analysis

Depending on the fields available, the application can generate:

Revenue / sales metrics

Profit and margin metrics

Transaction counts

Volume / quantity metrics

Top revenue drivers

Profit by item/category

Revenue trends over time

Peak sales periods

Category/location comparisons

Numeric relationship maps

Business observations

Recommended next actions

The dashboard does not invent a metric when the required field is missing.

🤖 Machine Learning Layer

Machine learning is an integrated part of the project, not a separate demo.

The ML engine first checks which prediction problems are actually supported by the uploaded data.

Supervised Learning

When a suitable target is available, multiple candidate models can be trained and compared.

Regression

Used when the target is numeric.

Linear Regression

Ridge Regression

Random Forest Regressor

Extra Trees Regressor

Typical examples:

Revenue prediction

Profit prediction

Quantity prediction

Numeric business outcomes

Classification

Used when the target is categorical with enough usable observations.

Logistic Regression

Random Forest Classifier

Extra Trees Classifier

Typical examples:

Outcome / status prediction

Customer or operational classification tasks

Other categorical targets discovered from the data

Unsupervised Learning

Anomaly Detection

Isolation Forest identifies unusually different observations across available numeric signals.

An anomaly is presented as unusual activity, not automatically as fraud.

Clustering

K-Means groups observations into similar segments when multiple numeric features are available.

This can support use cases such as:

Customer segmentation

Product grouping

Operational pattern discovery

Time-Based Forecasting

When the dataset contains a usable date/time field and enough history, the platform can build a lag-feature forecasting model.

The workflow uses:

Time-ordered observations

Lag features

Rolling features

Chronological train/validation split

MAE-based validation

Future-period forecasts

This avoids using future observations to train on earlier periods.

🧪 ML Validation Strategy

A major design goal is to avoid misleading model results.

Dataset
   ↓
Choose a valid target
   ↓
Remove obvious leakage fields
   ↓
Prepare numeric + categorical predictors
   ↓
Use chronological holdout when time is available
   ↓
Train multiple candidate models
   ↓
Compare validation metrics
   ↓
Select the best validated candidate

For regression, the application reports MAE and R².

For classification, the application evaluates candidate models using classification performance metrics where supported.

When the dataset cannot support a supervised problem, the application does not force a meaningless model.

🔗 Multi-Dataset Relationships & SQL

Data Doctor AI can accept multiple CSV/Excel files together.

Example:

customers.csv
    Customer ID
          │
          │ relationship
          ▼
orders.csv
    Customer ID

The relationship engine looks for likely identifier relationships using:

Identifier-like field names

Uniqueness

Shared values across datasets

Basic structural evidence

It then produces a suggested relational model and SQL relationship statements.

Example SQL

ALTER TABLE orders
ADD FOREIGN KEY (customer_id)
REFERENCES customers(customer_id);

Primary-key candidates are treated as suggestions, not automatically declared facts.

👤 Business Owner Dashboard

The Business Owner view is designed for a non-technical user.

It turns analytical output into simple business language.

Example sections

Business snapshot

Revenue / profit / margin

Transactions and volume

Top revenue drivers

Profit by product/category

Revenue trend

Peak time / peak day

Unusual activity

Demand / activity forecast

What the owner should know

Recommended next actions

The dashboard adapts to what is actually present in the uploaded dataset.

For example:

Retail dataset

Focus may include:

Products

Sales

Profit

Categories

Store/branch

Demand

Unusual activity

Healthcare dataset

Focus can shift toward available fields such as:

Patients / records

Departments

Charges or revenue

Outcomes/status

Time trends

Relevant numerical and categorical patterns

Restaurant dataset

Focus may include:

Menu items

Quantity sold

Revenue

Profit/cost when available

Peak hours

Item performance

The application does not hard-code a single business dataset.

📊 Data / Power BI Analyst Workspace

The Analyst workspace is intentionally deeper than the Business Owner experience.

Overview

Dataset size

Number of fields

Missing values

Duplicates

Domain estimate

Confidence

Column inventory

Data Check

Field-by-field profiling

Data type audit

Missing-value analysis

Uniqueness

Automatic semantic mapping

Cleaning / preparation findings

Visual Analysis

Interactive Plotly visualizations including:

Time trends

Top items

Aggregations by business dimensions

Profit/revenue comparisons

Numeric relationship maps

ML Lab

Candidate target selection

Multiple model training

Model comparison

Validation metrics

Leakage-aware predictors

Anomaly detection

Clustering

Time-based forecasting

SQL & Data Model

Identifier candidates

Cross-file relationships

Suggested relational structure

SQL DDL / relationship statements

Business Story

A concise narrative explaining:

What happened?

What stands out?

What may need attention?

What should be checked next?

Export

Download analytical outputs where supported.

🏗️ Project Architecture

                    ┌─────────────────────┐
                    │   CSV / Excel Data  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Schema Engine     │
                    │ field + domain      │
                    │ discovery           │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌────────────┐ ┌────────────┐ ┌──────────────┐
        │ Data       │ │ Business   │ │ Relationship │
        │ Analysis   │ │ Insights   │ │ + SQL Model  │
        └─────┬──────┘ └─────┬──────┘ └──────┬───────┘
              │              │                │
              └──────────────┼────────────────┘
                             ▼
                    ┌─────────────────────┐
                    │    ML Engine        │
                    │ Regression          │
                    │ Classification      │
                    │ Forecasting         │
                    │ Anomaly Detection   │
                    │ Clustering          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Business Decision   │
                    │ Dashboard           │
                    │ + Recommendations   │
                    └─────────────────────┘

🧰 Tech Stack

Area

Technology

UI

Streamlit

Data analysis

Pandas, NumPy

Visualization

Plotly

Machine learning

Scikit-learn

File handling

CSV, Excel, OpenPyXL, xlrd

Testing

Pytest

Deployment

Streamlit Community Cloud

📂 Project Structure

Data_Doctor_AI/
│
├── app.py                 # Main Streamlit application
├── schema_engine.py       # Field discovery, canonicalization, domain detection
├── ml_engine.py           # Supervised ML, forecasting, anomaly detection, clustering
├── insights_engine.py     # Business insights and recommendations
├── relationships.py       # Cross-dataset relationship + SQL model logic
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit UI configuration
└── tests/
    ├── conftest.py
    └── test_project.py

🚀 Run Locally

git clone https://github.com/Ujjwal-005/data-doctor-ai.git
cd data-doctor-ai

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python -m streamlit run app.py

Then open:

http://localhost:8501

✅ Testing

Run:

python -m pytest -q

The project includes tests for the core schema, analytics and ML behavior.

🎯 What Problem Does This Project Solve?

Traditional dashboards usually assume a fixed schema and require a developer or analyst to build a new dashboard for every business dataset.

Data Doctor AI explores a different workflow:

The dataset becomes the starting point. The system discovers its structure first, then chooses the analysis and ML capabilities that the data can actually support.

This makes the project useful as a portfolio demonstration of:

Data analysis

Business intelligence

Automated data understanding

Machine learning

Model validation

Forecasting

Anomaly detection

Clustering

SQL / relational modeling

Decision-oriented communication

📌 ML + Data Analysis Requirement in This Project

This project is intentionally a combined Data Analytics + Machine Learning application.

Data Analysis is responsible for:

Understand → Clean → Profile → Measure → Visualize → Explain

Machine Learning is responsible for:

Predict → Forecast → Compare Models → Detect Anomalies → Segment

Together:

Data Analysis
     ↓
Find the business problem
     ↓
ML decides whether the problem is predictable
     ↓
Train suitable models
     ↓
Validate the result
     ↓
Turn the result into a business action

This separation is intentional: ML is not forced onto every dataset, while the analytics layer remains useful even when the data is not suitable for predictive modeling.

🔮 Future Improvements

Potential production upgrades include:

Managed PostgreSQL storage

Authentication and role-based access

More specialized forecasting models

Automatic model monitoring and retraining

More advanced natural-language questions over data

Additional domain-specific analytics packs

Scheduled reporting

👨‍💻 Author

Ujjwal Singh

Data Analytics · Business Intelligence · Machine Learning

GitHub: https://github.com/Ujjwal-005

⭐ Portfolio Note

This project combines Data Analysis + Business Intelligence + Machine Learning in one deploy
