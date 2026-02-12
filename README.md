Predictive Maintenance for Industrial Equipment
Project Overview

This project implements a predictive maintenance framework for industrial equipment using machine learning on synthetic sensor data. The objective is to anticipate equipment failures, enable condition-based maintenance, minimize unplanned downtime, and optimize maintenance scheduling.

We simulate 10 industrial machines over 30 days, including realistic sensor readings, maintenance events, and failure occurrences, to create a controlled environment for predictive modeling and analysis.

Datasets (Synthetic)

sensor_readings.csv – Daily operational parameters for each machine (300 records)

Vibration (mm/s), temperature (°C), pressure (bar), rotational speed (RPM), power consumption (kW)

Cumulative runtime, days since last maintenance, computed health score

Failure label (1 if failure occurred on that day)

maintenance_history.csv – Preventive and corrective maintenance events (~30 records)

failure_events.csv – Failure details including type, downtime, and repair cost (~20 records)

Analytical Techniques

Feature Engineering: Creation of rolling statistics, lag features, rate-of-change indicators, and domain-specific health scores

Exploratory Data Analysis: Distribution analysis, time-series visualization, correlation heatmaps

Imbalanced Learning: SMOTE oversampling to address rare failure events (~5% occurrence)

Time-Series Cross-Validation: Temporal train/test splits to prevent data leakage

Modeling: Random Forest and XGBoost with hyperparameter tuning using GridSearchCV

Evaluation Metrics: ROC-AUC, Precision, Recall, F1-Score, confusion matrix

Interpretability: SHAP (SHapley Additive exPlanations) for global and individual feature importance

Installation

Install required dependencies with:

pip install -r requirements.txt
