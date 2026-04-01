"""
Shared utilities: data generation, preprocessing, model training
Cached with st.cache_data / st.cache_resource
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, roc_auc_score, mean_absolute_error,
                              r2_score, confusion_matrix, classification_report)

FEATURE_COLS = [
    'Age', 'Gender', 'Department', 'Role', 'Education', 'Tenure', 'Salary',
    'JobSatisfaction', 'Workload', 'ManagerScore', 'NumPromotions',
    'TrainingHours', 'Overtime', 'DistanceFromOffice', 'NumProjects'
]

TARGETS = {
    'Attrition':           'clf',
    'PerformanceRating':   'clf',
    'AbsentDays':          'reg',
    'PromotionLikelihood': 'clf',
}

TARGET_META = {
    'Attrition':           {'icon': '🔴', 'color': '#f87171', 'label': 'Attrition Risk'},
    'PerformanceRating':   {'icon': '⭐', 'color': '#fbbf24', 'label': 'Performance'},
    'AbsentDays':          {'icon': '📅', 'color': '#60a5fa', 'label': 'Absent Days/yr'},
    'PromotionLikelihood': {'icon': '🚀', 'color': '#34d399', 'label': 'Promotion Chance'},
}


@st.cache_data
def generate_data(n=1500, seed=42):
    np.random.seed(seed)
    dept   = np.random.choice(['Engineering','Sales','HR','Finance','Marketing','Operations'],
                               n, p=[0.25,0.20,0.10,0.15,0.15,0.15])
    role   = np.random.choice(['Analyst','Manager','Director','IC','Lead'], n)
    edu    = np.random.choice(['High School','Bachelor','Master','PhD'],
                               n, p=[0.10,0.50,0.30,0.10])
    gender = np.random.choice(['Male','Female','Other'], n, p=[0.48,0.48,0.04])

    age           = np.random.randint(22, 60, n)
    tenure        = np.clip(np.random.exponential(4, n).astype(int), 0, 35)
    salary        = np.random.normal(70000, 20000, n).clip(30000, 200000)
    satisfaction  = np.random.randint(1, 6, n)
    workload      = np.random.randint(1, 6, n)
    manager_score = np.random.randint(1, 6, n)
    promotions    = np.random.poisson(1.2, n).clip(0, 6)
    training_hrs  = np.random.randint(0, 80, n)
    overtime      = np.random.choice([0, 1], n, p=[0.65, 0.35])
    distance      = np.random.randint(1, 60, n)
    projects      = np.random.randint(1, 8, n)

    attrition_prob = (
        0.05 + 0.15*(satisfaction <= 2) + 0.12*(workload >= 4)
        + 0.10*overtime + 0.08*(distance > 40)
        - 0.06*(tenure > 5) - 0.05*(salary > 90000)
        + np.random.normal(0, 0.05, n)
    ).clip(0, 1)
    attrition = (np.random.rand(n) < attrition_prob).astype(int)

    perf_score = (
        2.0 + 0.3*(satisfaction - 3) + 0.25*(manager_score - 3)
        + 0.2*(training_hrs/40) + 0.15*(projects - 3)
        - 0.1*overtime + np.random.normal(0, 0.4, n)
    ).clip(1, 4)
    performance = np.round(perf_score).astype(int).clip(1, 4)

    absent_days = (
        5 + 3*(satisfaction <= 2) + 2*overtime + 1.5*(workload >= 4)
        - 1.5*(manager_score >= 4) + np.random.normal(0, 3, n)
    ).clip(0, 30).round().astype(int)

    promo_prob = (
        0.10 + 0.20*(performance >= 3) + 0.15*(tenure > 3)
        + 0.12*(training_hrs > 40) + 0.10*(projects > 4)
        - 0.08*attrition + np.random.normal(0, 0.05, n)
    ).clip(0, 1)
    promotion = (np.random.rand(n) < promo_prob).astype(int)

    return pd.DataFrame({
        'Age': age, 'Gender': gender, 'Department': dept,
        'Role': role, 'Education': edu, 'Tenure': tenure,
        'Salary': salary.round(0), 'JobSatisfaction': satisfaction,
        'Workload': workload, 'ManagerScore': manager_score,
        'NumPromotions': promotions, 'TrainingHours': training_hrs,
        'Overtime': overtime, 'DistanceFromOffice': distance,
        'NumProjects': projects, 'Attrition': attrition,
        'PerformanceRating': performance, 'AbsentDays': absent_days,
        'PromotionLikelihood': promotion
    })


def encode_df(df):
    df = df.copy()
    cat_cols = ['Gender', 'Department', 'Role', 'Education']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


@st.cache_resource
def train_all_models(n_samples=1500, seed=42):
    df = generate_data(n_samples, seed)
    df_enc, encoders = encode_df(df)
    X = df_enc[FEATURE_COLS]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    results = {}
    for target, task in TARGETS.items():
        y = df_enc[target]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=0.2, random_state=seed,
            stratify=y if task == 'clf' else None
        )
        if task == 'clf':
            models = {
                'Random Forest':     RandomForestClassifier(n_estimators=150, max_depth=8, random_state=seed),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=seed),
                'Logistic Reg':      LogisticRegression(max_iter=500, random_state=seed),
            }
        else:
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=8, random_state=seed),
                'Ridge':         Ridge(alpha=1.0),
            }

        target_results = {}
        for mname, model in models.items():
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            if task == 'clf':
                acc = accuracy_score(y_te, y_pred)
                cv  = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy').mean()
                try:
                    if len(np.unique(y)) == 2:
                        proba = model.predict_proba(X_te)[:, 1]
                        auc   = roc_auc_score(y_te, proba)
                    else:
                        auc = roc_auc_score(y_te, model.predict_proba(X_te),
                                            multi_class='ovr', average='macro')
                except Exception:
                    auc = None
                target_results[mname] = {
                    'model': model, 'task': task,
                    'X_te': X_te, 'y_te': y_te, 'y_pred': y_pred,
                    'accuracy': acc, 'cv_accuracy': cv, 'auc': auc,
                    'cm': confusion_matrix(y_te, y_pred),
                    'report': classification_report(y_te, y_pred, output_dict=True),
                }
            else:
                mae = mean_absolute_error(y_te, y_pred)
                r2  = r2_score(y_te, y_pred)
                cv  = cross_val_score(model, X_scaled, y, cv=5, scoring='r2').mean()
                target_results[mname] = {
                    'model': model, 'task': task,
                    'X_te': X_te, 'y_te': y_te, 'y_pred': y_pred,
                    'mae': mae, 'r2': r2, 'cv_r2': cv,
                }
        results[target] = target_results

    return results, scaler, encoders, df


def predict_employee(employee_dict, results, scaler, encoders):
    """Score a single employee dict across all targets."""
    row = pd.DataFrame([employee_dict])
    for col, le in encoders.items():
        if col in row.columns:
            try:
                row[col] = le.transform(row[col].astype(str))
            except Exception:
                row[col] = 0
    row_scaled = pd.DataFrame(scaler.transform(row[FEATURE_COLS]), columns=FEATURE_COLS)

    preds = {}
    for target, task in TARGETS.items():
        tres = results[target]
        if task == 'clf':
            best_m = max(tres, key=lambda m: tres[m]['cv_accuracy'])
            model  = tres[best_m]['model']
            pred   = model.predict(row_scaled)[0]
            proba  = model.predict_proba(row_scaled)[0]
            preds[target] = {'value': pred, 'proba': proba, 'task': task, 'model': best_m}
        else:
            best_m = max(tres, key=lambda m: tres[m]['cv_r2'])
            model  = tres[best_m]['model']
            pred   = model.predict(row_scaled)[0]
            preds[target] = {'value': pred, 'task': task, 'model': best_m}
    return preds
