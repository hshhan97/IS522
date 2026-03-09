import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
from tensorflow import keras
from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title='COVID-19 Mortality Prediction', layout='wide')

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'covid.csv'
SAVED_MODELS = ROOT / 'saved_models'
ARTIFACTS = ROOT / 'artifacts'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c != 'Unnamed: 0')
    df.columns = df.columns.str.strip().str.upper()

    # Balanced sample to match the notebook workflow
    death_1_sample = df[df['DEATH'] == 1].sample(n=min(5000, (df['DEATH'] == 1).sum()), random_state=42)
    death_0_sample = df[df['DEATH'] == 0].sample(n=min(5000, (df['DEATH'] == 0).sum()), random_state=42)
    model_df = pd.concat([death_1_sample, death_0_sample], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    return df, model_df

@st.cache_resource
def load_models():
    models = {}
    missing = []
    for name, filename in {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'Random Forest': 'random_forest.pkl',
        'LightGBM': 'lightgbm.pkl',
        'Neural Network': 'neural_network.keras',
        'Scaler': 'nn_scaler.pkl',
    }.items():
        path = SAVED_MODELS / filename
        if not path.exists():
            missing.append(filename)
            continue
        if filename.endswith('.keras'):
            models[name] = keras.models.load_model(path)
        else:
            models[name] = joblib.load(path)
    return models, missing


def metric_card(label, value):
    st.metric(label, value)


def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='DEATH', ax=ax)
    ax.set_title('Target Distribution (Balanced Modeling Sample)')
    ax.set_xlabel('DEATH (0 = Lived, 1 = Died)')
    ax.set_ylabel('Count')
    fig.tight_layout()
    return fig


def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['AGE'], bins=30, kde=True, ax=ax)
    ax.set_title('Age Distribution')
    ax.set_xlabel('Age')
    fig.tight_layout()
    return fig


def plot_age_boxplot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='DEATH', y='AGE', ax=ax)
    ax.set_title('Age Distribution by Mortality Outcome')
    ax.set_xlabel('DEATH')
    fig.tight_layout()
    return fig


def plot_comorbidity_chart(df):
    cols = [c for c in ['DIABETES', 'HYPERTENSION', 'OBESITY', 'PNEUMONIA'] if c in df.columns]
    summary = []
    for col in cols:
        summary.append({'Feature': col, 'Mortality Rate': df.loc[df[col] == 1, 'DEATH'].mean()})
    summary = pd.DataFrame(summary).sort_values('Mortality Rate', ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=summary, x='Feature', y='Mortality Rate', ax=ax)
    ax.set_title('Mortality Rate With Selected Conditions')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_heatmap(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap')
    fig.tight_layout()
    return fig


def build_input_df(model_df):
    # Use a handful of meaningful fields for interactive prediction; use dataset mean/mode for the rest.
    feature_defaults = model_df.drop(columns=['DEATH']).median(numeric_only=True).to_dict()
    input_data = feature_defaults.copy()

    cols = st.columns(2)
    input_data['AGE'] = cols[0].slider('Age', 0, 100, int(feature_defaults.get('AGE', 50)))

    for i, col in enumerate(['SEX', 'PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HYPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'INTUBED', 'PATIENT_TYPE']):
        if col in model_df.columns:
            target_col = cols[i % 2]
            input_data[col] = target_col.selectbox(col, [0, 1], index=int(round(feature_defaults.get(col, 0))))

    input_df = pd.DataFrame([input_data])
    # Ensure all training columns exist and order matches
    X = model_df.drop(columns=['DEATH'])
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else X[col].mode().iloc[0]
    input_df = input_df[X.columns]
    return input_df


def predict_with_model(model_name, models, input_df):
    if model_name == 'Neural Network':
        scaler = models.get('Scaler')
        model = models.get('Neural Network')
        if scaler is None or model is None:
            return None, None
        proba = float(model.predict(scaler.transform(input_df), verbose=0).ravel()[0])
    else:
        model = models.get(model_name)
        if model is None:
            return None, None
        proba = float(model.predict_proba(input_df)[:, 1][0])
    pred = int(proba >= 0.5)
    return pred, proba


st.title('COVID-19 Mortality Prediction Dashboard')
st.caption('MSIS 522 — Homework 1: The Complete Data Science Workflow')

if not DATA_PATH.exists():
    st.error("`covid.csv` was not found in this folder. Add the dataset to the repo root before deployment.")
    st.stop()

raw_df, model_df = load_data()
models, missing_models = load_models()

if missing_models:
    st.warning('Some saved model files are missing: ' + ', '.join(missing_models) + '. Run the notebook cells that save models before deployment.')

tab1, tab2, tab3, tab4 = st.tabs([
    'Executive Summary',
    'Descriptive Analytics',
    'Model Performance',
    'Explainability & Interactive Prediction'
])

with tab1:
    st.header('Executive Summary')
    st.write(
        "This project predicts COVID-19 mortality risk from a tabular patient dataset containing demographics, comorbidities, and clinical indicators. "
        "The target variable is `DEATH`, a binary outcome that indicates whether a patient died or survived. "
        "The dataset is valuable for a healthcare analytics workflow because it supports a high-impact classification task in which predictions can inform prioritization and triage."
    )
    st.write(
        "Several supervised learning models were trained and compared, including Logistic Regression, Decision Tree, Random Forest, LightGBM, and a Neural Network. "
        "Across the models, performance was consistently strong, with F1 scores around 0.90 and AUC values around 0.95. "
        "Random Forest delivered the best overall F1 score in the notebook results, but the margin over the other models was small, suggesting that the dataset contains strong predictive signal."
    )
    c1, c2, c3 = st.columns(3)
    metric_card('Rows in modeling sample', f"{len(model_df):,}")
    c2.metric('Features', model_df.shape[1] - 1)
    c3.metric('Target', 'DEATH')

with tab2:
    st.header('Descriptive Analytics')
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_target_distribution(model_df))
        st.caption('The modeling dataset was balanced to keep the classes comparable during training. Because of that, evaluation focuses on F1 and AUC rather than accuracy alone.')
        st.pyplot(plot_age_distribution(model_df))
        st.caption('Age is concentrated in adult and older-adult ranges, which suggests it is likely to be a major mortality predictor.')
    with col2:
        st.pyplot(plot_age_boxplot(model_df))
        st.caption('Patients who died tend to be older than those who survived, which makes age one of the clearest risk signals in the dataset.')
        st.pyplot(plot_comorbidity_chart(model_df))
        st.caption('Pneumonia and major comorbidities are associated with higher mortality rates, which is clinically intuitive and useful for triage.')
    st.pyplot(plot_heatmap(model_df))
    st.caption('The correlation heatmap shows several moderate associations with mortality, but no single feature completely dominates the dataset.')

with tab3:
    st.header('Model Performance')
    comparison_path = ARTIFACTS / 'model_comparison.csv'
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        st.dataframe(comparison_df, use_container_width=True)
    else:
        comparison_df = pd.DataFrame({
            'Model': ['Random Forest', 'Logistic Regression', 'Decision Tree', 'Neural Network', 'LightGBM'],
            'Accuracy': [0.9013, 0.8990, 0.8987, 0.9007, 0.8997],
            'AUC': [0.9505, 0.9496, 0.9456, 0.9466, 0.9501],
            'Precision': [0.8697, 0.8638, 0.8627, 0.8671, 0.8672],
            'Recall': [0.9440, 0.9490, 0.9490, 0.9420, 0.9410],
            'F1': [0.9053, 0.9039, 0.9036, 0.9029, 0.9024],
        })
        st.dataframe(comparison_df, use_container_width=True)

    img1 = ARTIFACTS / 'model_comparison_f1.png'
    img2 = ARTIFACTS / 'model_comparison_auc.png'
    c1, c2 = st.columns(2)
    if img1.exists():
        c1.image(str(img1), use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=comparison_df, x='Model', y='F1', ax=ax)
        ax.set_ylim(0.89, 0.907)
        ax.tick_params(axis='x', rotation=20)
        c1.pyplot(fig)
    if img2.exists():
        c2.image(str(img2), use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=comparison_df, x='Model', y='AUC', ax=ax)
        ax.set_ylim(0.944, 0.9515)
        ax.tick_params(axis='x', rotation=20)
        c2.pyplot(fig)

    st.markdown('**Best hyperparameters from the notebook**')
    st.write('- Decision Tree: `max_depth=5`, `min_samples_leaf=40`')
    st.write('- Random Forest: `n_estimators=100`, `max_depth=8`')
    st.write('- LightGBM: best values selected through GridSearchCV over `n_estimators`, `max_depth`, and `learning_rate`')

with tab4:
    st.header('Explainability & Interactive Prediction')
    st.subheader('Custom Input Prediction')
    model_name = st.selectbox('Choose a model', ['Logistic Regression', 'Decision Tree', 'Random Forest', 'LightGBM', 'Neural Network'])
    input_df = build_input_df(model_df)
    st.dataframe(input_df, use_container_width=True)

    pred, proba = predict_with_model(model_name, models, input_df)
    if pred is not None:
        st.success(f'Predicted class: {pred} ({"High risk" if pred == 1 else "Lower risk"})')
        st.metric('Predicted probability of death', f'{proba:.3f}')
    else:
        st.info('Prediction unavailable until saved models are generated and placed in `saved_models/`.')

    st.subheader('SHAP Explainability')
    if 'LightGBM' in models:
        X = model_df.drop(columns=['DEATH'])
        sample_X = X.sample(n=min(300, len(X)), random_state=42)
        explainer = shap.TreeExplainer(models['LightGBM'])
        shap_values = explainer.shap_values(sample_X)
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1]
            expected_value = explainer.expected_value[1]
        else:
            shap_values_plot = shap_values
            expected_value = explainer.expected_value

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_plot, sample_X, show=False)
        st.pyplot(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_plot, sample_X, plot_type='bar', show=False)
        st.pyplot(fig)
        plt.close(fig)

        waterfall_idx = 0
        shap_explanation = shap.Explanation(
            values=shap_values_plot[waterfall_idx],
            base_values=expected_value,
            data=sample_X.iloc[waterfall_idx],
            feature_names=sample_X.columns.tolist(),
        )
        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_explanation, max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info('SHAP plots will appear here after the saved LightGBM model is available.')
