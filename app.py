import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="COVID-19 Mortality Prediction", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "covid.csv"
MODELS_DIR = ROOT / "saved_models"
ARTIFACTS_DIR = ROOT / "artifacts"

st.title("COVID-19 Mortality Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().upper() for c in df.columns]
    return df

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
        "Decision Tree": MODELS_DIR / "decision_tree.pkl",
        "Random Forest": MODELS_DIR / "random_forest.pkl",
        "LightGBM": MODELS_DIR / "lightgbm.pkl",
    }

    for name, path in model_files.items():
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")
        else:
            st.warning(f"Missing model file: {path.name}")

    return models

def infer_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c != "DEATH"]

def prepare_input_row(df: pd.DataFrame, feature_cols):
    st.subheader("Interactive Prediction")

    default_row = df[feature_cols].median(numeric_only=True).to_dict()
    values = {}

    cols = st.columns(3)
    for i, col in enumerate(feature_cols):
        container = cols[i % 3]
        series = df[col].dropna()

        unique_vals = sorted(series.unique().tolist()) if not series.empty else []
        if len(unique_vals) > 0 and len(unique_vals) <= 6:
            cleaned = []
            for v in unique_vals:
                try:
                    cleaned.append(int(v) if float(v).is_integer() else v)
                except Exception:
                    cleaned.append(v)
            options = cleaned
            default_value = options[0]
            if col in default_row and pd.notna(default_row.get(col)):
                try:
                    candidate = int(round(default_row[col]))
                    if candidate in options:
                        default_value = candidate
                except Exception:
                    pass
            idx = options.index(default_value) if default_value in options else 0
            values[col] = container.selectbox(col, options=options, index=idx)
        else:
            col_default = float(default_row.get(col, 0))
            if pd.api.types.is_integer_dtype(series):
                values[col] = int(container.number_input(col, value=int(round(col_default)), step=1))
            else:
                values[col] = float(container.number_input(col, value=float(col_default)))
    return pd.DataFrame([values])

def predict_with_model(model, input_df):
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(input_df)[:, 1][0])
        pred = int(proba >= 0.5)
        return pred, proba
    pred = int(model.predict(input_df)[0])
    return pred, float(pred)

def show_artifact_image(filename, caption):
    path = ARTIFACTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{filename} not found in artifacts/.")

def show_shap_waterfall(model_name, model, background_X, input_df):
    st.subheader("Local Explainability (SHAP Waterfall)")
    try:
        background = background_X.sample(min(200, len(background_X)), random_state=42)
        explainer = shap.Explainer(model, background)
        shap_values = explainer(input_df)

        fig = plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.info(f"Could not generate SHAP waterfall for {model_name}: {e}")

df = load_data()
models = load_models()
feature_cols = infer_feature_columns(df)

if not models:
    st.error("No saved models were found in the saved_models/ folder.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Prediction",
])

with tab1:
    st.header("Executive Summary")
    st.write(
        """
        This project predicts COVID-19 mortality risk using a tabular patient dataset with
        demographic variables, comorbidities, and clinical indicators. The target variable is
        **DEATH**, which represents whether a patient died or survived.

        This problem matters because accurate risk stratification can support triage, resource
        allocation, and earlier intervention for higher-risk patients. Multiple machine learning
        models were trained and compared, including Logistic Regression, Decision Tree, Random
        Forest, LightGBM, and a Neural Network in the notebook analysis.

        The deployed app focuses on the non-neural-network models so it can run reliably in
        Streamlit Cloud without TensorFlow. In the full notebook analysis, model performance
        across approaches was strong and relatively close, with tree-based ensemble methods
        slightly outperforming the simpler baselines.
        """
    )

    st.subheader("Dataset Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Features", f"{len(feature_cols):,}")
    c3.metric("Target", "DEATH")

with tab2:
    st.header("Descriptive Analytics")
    show_artifact_image("target_distribution.png", "Target distribution")
    show_artifact_image("correlation_heatmap.png", "Correlation heatmap")
    st.markdown(
        """
        The descriptive analytics section highlights the target distribution and the strongest
        feature relationships in the dataset. These visuals help identify the prevalence of the
        mortality outcome and reveal which clinical variables may be most informative for
        predictive modeling.
        """
    )

with tab3:
    st.header("Model Performance")

    comparison_csv = ARTIFACTS_DIR / "model_comparison.csv"
    if comparison_csv.exists():
        comparison_df = pd.read_csv(comparison_csv)
        st.subheader("Model Comparison Table")
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("model_comparison.csv not found in artifacts/.")

    col1, col2 = st.columns(2)
    with col1:
        show_artifact_image("model_comparison_f1.png", "Model comparison by F1")
    with col2:
        show_artifact_image("model_comparison_auc.png", "Model comparison by AUC")

    st.markdown(
        """
        These results compare the classification models on the test set. In the notebook,
        Random Forest performed slightly better than the other models by F1 score, although
        the differences were relatively small overall.
        """
    )

with tab4:
    st.header("Explainability & Interactive Prediction")

    model_name = st.selectbox("Choose a model", list(models.keys()))
    model = models[model_name]

    input_df = prepare_input_row(df, feature_cols)

    pred, proba = predict_with_model(model, input_df)

    st.subheader("Prediction Result")
    label = "High Risk (Death = 1)" if pred == 1 else "Lower Risk (Death = 0)"
    c1, c2 = st.columns(2)
    c1.metric("Predicted Class", label)
    c2.metric("Predicted Probability of Death", f"{proba:.3f}")

    st.markdown("### Input Summary")
    st.dataframe(input_df, use_container_width=True)

    st.markdown("### SHAP Explainability")
    st.markdown("### SHAP Explainability")
    show_shap_waterfall(model_name, model, df[feature_cols], input_df)
    show_shap_waterfall(model_name, model, df[feature_cols], input_df)
