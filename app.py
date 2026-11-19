from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from advanced_future_yield_models import make_models, make_preprocessor
from ensemble_models import ENSEMBLE_NAMES, make_base_models, make_ensemble


DATA_PATH = Path(__file__).parent / "dataset" / "yield_df.csv"
TARGET = "hg/ha_yield"
CATEGORICAL_FEATURES = ["Area", "Item"]
NUMERIC_FEATURES = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
SELECTED_MODEL = "Extra Trees"
MODEL_OPTIONS = [SELECTED_MODEL, "Neural Network (MLP)",
                 "Histogram Gradient Boosting", "Random Forest (reference)"] + ENSEMBLE_NAMES
TREE_PIPELINES = ["Extra Trees", "Histogram Gradient Boosting", "Random Forest (reference)"]
SHAP_BACKGROUND_ROWS = 50
SHAP_IMPORTANCE_ROWS = 100

# Recorded from ensemble_models.py (weights from 2006–2009 validation RMSE, proportional
# to 1 / RMSE²). Hard-coded so the app never reruns the full experiment.
ENSEMBLE_WEIGHTS = {
    "Extra Trees": 0.4279,
    "Histogram Gradient Boosting": 0.2535,
    "Neural Network (MLP)": 0.3186,
}
STACKING_COEFFICIENTS = {
    "Extra Trees": 0.8156,
    "Histogram Gradient Boosting": 0.0000,
    "Neural Network (MLP)": 0.2081,
}
STACKING_INTERCEPT = 618.15


def advanced_results():
    """Recorded, rounded results from the executed advanced notebook."""
    return pd.DataFrame([
        ["Extra Trees", 16102.478, 10152.975, 19942.622, 0.956, 0.997],
        ["Neural Network (MLP)", 18661.863, 11221.705, 19397.042, 0.959, 0.991],
        ["Histogram Gradient Boosting", 20919.691, 13459.170, 23347.448, 0.940, 0.977],
        ["Random Forest (reference)", 16732.869, 10442.003, 20533.341, 0.954, 0.997],
    ], columns=["Model", "Validation RMSE", "Test MAE", "Test RMSE", "Test R²", "Train R²"])


def ensemble_results():
    """Recorded results from ensemble_models.py, sorted by validation RMSE."""
    return pd.DataFrame([
        ["Stacking", "Ensemble", 15616.21, 9273.16, 18270.33, 0.9634],
        ["Extra Trees", "Base", 16102.48, 10152.98, 19942.62, 0.9564],
        ["Weighted Average", "Ensemble", 16687.65, 10049.13, 18655.47, 0.9619],
        ["Simple Average", "Ensemble", 17023.91, 10326.79, 18818.90, 0.9612],
        ["Neural Network (MLP)", "Base", 18661.86, 11221.70, 19397.04, 0.9588],
        ["Histogram Gradient Boosting", "Base", 20919.69, 13459.17, 23347.45, 0.9403],
    ], columns=["Model", "Type", "Validation RMSE", "Test MAE", "Test RMSE", "Test R²"])


st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --teal: #0b4650;
            --teal-light: #165b65;
            --gold: #d2a33a;
            --ink: #17383d;
            --muted: #66777a;
            --canvas: #edf2f0;
            --card: #ffffff;
        }

        .stApp {background: var(--canvas);}
        .main .block-container {
            max-width: 1180px;
            padding: 2rem 2.25rem 3rem;
        }

        /* Sidebar navigation */
        [data-testid="stSidebar"] {
            background: var(--teal);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 1.25rem 0.85rem;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f5f8f7 !important;
        }
        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.2rem 0.35rem 1.4rem;
        }
        .brand-mark {
            width: 2.25rem;
            height: 2.25rem;
            display: grid;
            place-items: center;
            border: 1px solid var(--gold);
            border-radius: 50%;
            color: var(--gold);
            font-family: Georgia, serif;
            font-weight: 700;
        }
        .brand-name {
            color: white;
            font-family: Georgia, serif;
            font-size: 1.08rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .brand-tagline {
            color: #b9cdcf;
            font-size: 0.7rem;
            margin-top: 0.15rem;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 7px;
            margin: 0.2rem 0;
            padding: 0.55rem 0.65rem;
            transition: background 0.15s ease;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: var(--teal-light);
            box-shadow: inset 3px 0 0 var(--gold);
        }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div > div:first-child {
            display: none !important;
        }
        .team-card {
            border-top: 1px solid rgba(255, 255, 255, 0.14);
            margin-top: 1.1rem;
            padding: 1.2rem 0.4rem 0;
        }
        .team-label {
            color: #9fb9bc;
            font-size: 0.67rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        [data-testid="stSidebar"] .team-link {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            width: fit-content;
            color: white !important;
            font-size: 0.84rem;
            font-weight: 600;
            margin: 0.42rem 0;
            text-decoration: none;
            opacity: 0.94;
            transition: color 0.15s ease, opacity 0.15s ease;
        }
        [data-testid="stSidebar"] .team-link:hover {
            color: var(--gold) !important;
            opacity: 1;
        }
        .github-mark {
            width: 1rem;
            height: 1rem;
            flex: 0 0 1rem;
        }

        /* Page typography */
        .page-kicker {
            color: #a47717;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .main-title {
            color: var(--ink);
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 2.25rem;
            font-weight: 700;
            line-height: 1.15;
        }
        .subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin: 0.45rem 0 1.65rem;
        }
        h1, h2, h3 {
            color: var(--ink) !important;
            font-family: Georgia, 'Times New Roman', serif !important;
        }
        h2 {font-size: 1.35rem !important;}
        h3 {font-size: 1.08rem !important;}

        /* Dashboard cards */
        div[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid #dce5e2;
            border-top: 3px solid var(--gold);
            border-radius: 9px;
            padding: 1rem 1.05rem;
            box-shadow: 0 4px 14px rgba(24, 57, 61, 0.07);
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.78rem;
        }
        div[data-testid="stMetricValue"] {
            color: var(--teal);
            font-family: Georgia, 'Times New Roman', serif;
        }
        div[data-testid="stMetricValue"] p {
            font-size: 1.9rem !important;
            white-space: nowrap;
            overflow: visible;
            text-overflow: clip;
        }
        [data-testid="stForm"] {
            background: var(--card);
            border: 1px solid #dce5e2;
            border-top: 4px solid var(--gold);
            border-radius: 10px;
            padding: 1.25rem 1.4rem 1.35rem;
            box-shadow: 0 6px 18px rgba(24, 57, 61, 0.07);
        }
        [data-testid="stImage"] {
            background: var(--card);
            border: 1px solid #dce5e2;
            border-radius: 10px;
            padding: 0.6rem;
            box-shadow: 0 5px 16px rgba(24, 57, 61, 0.06);
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid #dce5e2;
            border-radius: 9px;
            overflow: hidden;
            box-shadow: 0 4px 14px rgba(24, 57, 61, 0.05);
        }
        [data-testid="stExpander"] {
            background: white;
            border: 1px solid #dce5e2;
            border-radius: 9px;
        }
        .stButton > button,
        [data-testid="stFormSubmitButton"] > button {
            background: var(--teal);
            border: 1px solid var(--teal);
            color: white;
            border-radius: 7px;
            font-weight: 700;
            min-height: 2.7rem;
        }
        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            background: var(--teal-light);
            border-color: var(--gold);
            color: white;
        }
        [data-testid="stAlert"] {
            border-radius: 8px;
            border-left-width: 4px;
        }
        hr {border-color: #d8e2df !important;}
        a {color: var(--teal-light) !important;}

        @media (max-width: 768px) {
            .main .block-container {padding: 1.25rem 1rem 2rem;}
            .main-title {font-size: 1.8rem;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)

sns.set_theme(style="whitegrid", palette=["#0b4650", "#d2a33a"])
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#dce5e2",
        "axes.titlecolor": "#17383d",
        "axes.labelcolor": "#52666a",
        "xtick.color": "#65777a",
        "ytick.color": "#65777a",
        "grid.color": "#e8eeec",
    }
)


@st.cache_data
def load_data():
    """Load the merged dataset once and remove its saved index column."""
    data = pd.read_csv(DATA_PATH)
    return data.drop(columns=["Unnamed: 0"], errors="ignore")


def training_message(model_name):
    if model_name == "Stacking":
        return ("Training Stacking: 3 base models are fitted on 4 time folds to train the "
                "meta-model, then refitted on all data (15 fits). This can take a few minutes...")
    if model_name in ENSEMBLE_NAMES:
        return (f"Training {model_name}: fitting Extra Trees, Histogram Gradient Boosting "
                "and the neural network on historical data...")
    return f"Training {model_name} on historical data..."


@st.cache_resource(show_spinner=False, max_entries=1)
def train_model(data, model_name):
    """Fit only the requested model; keep one fitted model cached to limit memory."""
    if model_name in ENSEMBLE_NAMES:
        base_models = make_base_models()
        base_models["Extra Trees"].set_params(model__n_jobs=1)
        weights = ENSEMBLE_WEIGHTS if model_name == "Weighted Average" else None
        ensemble = make_ensemble(model_name, weights).set_params(base_models=base_models)
        with threadpool_limits(limits=2):
            ensemble.fit(data[FEATURES], data[TARGET])
        return ensemble

    models = make_models(make_preprocessor())
    models["Random Forest (reference)"] = Pipeline([
        ("preprocessing", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=50, min_samples_split=5, random_state=42, n_jobs=1))
    ])
    pipeline = models[model_name]
    if model_name == "Extra Trees":
        pipeline.set_params(model__n_jobs=1)
    with threadpool_limits(limits=2):
        pipeline.fit(data[FEATURES], data[TARGET])
    return pipeline


@st.cache_data(show_spinner=False)
def shap_background(data):
    """Small fixed sample of historical rows used as the SHAP reference."""
    return data[FEATURES].sample(n=SHAP_BACKGROUND_ROWS, random_state=42).reset_index(drop=True)


def explain_scenario(model, model_name, input_data, background):
    """SHAP values for one scenario on the 6 original features."""
    # Imported lazily: shap is only needed once a prediction is explained.
    from explainability import explain_model, explain_tree_pipeline

    with threadpool_limits(limits=2):
        if model_name in TREE_PIPELINES:
            return explain_tree_pipeline(model, input_data, X_background=background,
                                         n_background=SHAP_BACKGROUND_ROWS)
        return explain_model(model, background, input_data, max_background=SHAP_BACKGROUND_ROWS)


@st.cache_data(show_spinner="Computing SHAP values for 100 historical rows...", max_entries=1)
def extra_trees_importance(data):
    """Global mean |SHAP| for the default Extra Trees model (100 sampled rows)."""
    from explainability import explain_tree_pipeline, shap_importance_table

    model = train_model(data, SELECTED_MODEL)
    rows = data[FEATURES].sample(n=SHAP_IMPORTANCE_ROWS, random_state=7)
    with threadpool_limits(limits=2):
        explanation = explain_tree_pipeline(model, rows, X_background=shap_background(data),
                                            n_background=SHAP_BACKGROUND_ROWS)
    return shap_importance_table(explanation)


def show_header(title, subtitle):
    st.markdown(
        '<div class="page-kicker">Agricultural intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


data = load_data()

st.sidebar.markdown(
    """
    <div class="brand-wrap">
        <div class="brand-mark">CY</div>
        <div>
            <div class="brand-name">CropYield</div>
            <div class="brand-tagline">Prediction dashboard</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Explore data", "Predict yield", "Model performance"],
    label_visibility="collapsed",
)
st.sidebar.markdown(
    """
    <div class="team-card">
        <div class="team-label">Team members</div>
        <a class="team-link" href="https://github.com/Nidhish01" target="_blank" rel="noopener noreferrer">
            <svg class="github-mark" viewBox="0 0 24 24" aria-hidden="true">
                <path fill="currentColor" d="M12 2C6.48 2 2 6.58 2 12.26c0 4.54 2.87 8.39 6.84 9.75.5.1.68-.22.68-.49 0-.24-.01-1.04-.01-1.89-2.78.62-3.37-1.21-3.37-1.21-.45-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.63.07-.63 1 .08 1.53 1.06 1.53 1.06.89 1.57 2.34 1.12 2.91.85.09-.67.35-1.12.63-1.38-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.04 1.03-2.76-.1-.26-.45-1.31.1-2.72 0 0 .84-.28 2.75 1.05A9.3 9.3 0 0 1 12 6.18a9.3 9.3 0 0 1 2.5.35c1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.46.1 2.72.64.72 1.03 1.64 1.03 2.76 0 3.94-2.34 4.8-4.57 5.06.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.81 0 .27.18.59.69.49A10.3 10.3 0 0 0 22 12.26C22 6.58 17.52 2 12 2Z"/>
            </svg>
            <span>Nidhish Bansal</span>
        </a>
        <a class="team-link" href="https://github.com/yvishi" target="_blank" rel="noopener noreferrer">
            <svg class="github-mark" viewBox="0 0 24 24" aria-hidden="true">
                <path fill="currentColor" d="M12 2C6.48 2 2 6.58 2 12.26c0 4.54 2.87 8.39 6.84 9.75.5.1.68-.22.68-.49 0-.24-.01-1.04-.01-1.89-2.78.62-3.37-1.21-3.37-1.21-.45-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.63.07-.63 1 .08 1.53 1.06 1.53 1.06.89 1.57 2.34 1.12 2.91.85.09-.67.35-1.12.63-1.38-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.04 1.03-2.76-.1-.26-.45-1.31.1-2.72 0 0 .84-.28 2.75 1.05A9.3 9.3 0 0 1 12 6.18a9.3 9.3 0 0 1 2.5.35c1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.46.1 2.72.64.72 1.03 1.64 1.03 2.76 0 3.94-2.34 4.8-4.57 5.06.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.81 0 .27.18.59.69.49A10.3 10.3 0 0 0 22 12.26C22 6.58 17.52 2 12 2Z"/>
            </svg>
            <span>Yash Vishnoi</span>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)


if page == "Overview":
    show_header(
        "Crop Yield Prediction Dashboard",
        "Explore historical agricultural data and estimate crop yield with machine learning.",
    )

    col1, col2 = st.columns(2)
    col1.metric("Observations", f"{len(data):,}")
    col2.metric("Countries", data["Area"].nunique())
    col3, col4 = st.columns(2)
    col3.metric("Crops", data["Item"].nunique())
    col4.metric("Year Range", f"{data['Year'].min()}–{data['Year'].max()}")

    st.subheader("About the project")
    st.write(
        "Crop yield is influenced by location, crop type, rainfall, temperature, "
        "pesticide use, and time. Alongside the original regression models, we now "
        "compare Extra Trees, Histogram Gradient Boosting, a neural network, and three "
        "ensembles that combine them. Extra Trees is the default prediction model because "
        "it had the lowest 2006–2009 validation RMSE among single models; a stacking "
        "ensemble improves on it, and SHAP explains each prediction."
    )
    st.caption("Final evaluation: train on 1990–2009, test on 2010–2013. "
               "The neural network had the lowest test RMSE; model selection used earlier validation years.")

    left, right = st.columns(2)
    with left:
        crop_yield = (
            data.groupby("Item", as_index=False)[TARGET]
            .median()
            .sort_values(TARGET, ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=crop_yield, x=TARGET, y="Item", color="#4f8a54", ax=ax)
        ax.set_title("Median yield by crop")
        ax.set_xlabel("Median yield (hg/ha)")
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        yearly_yield = data.groupby("Year", as_index=False)[TARGET].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=yearly_yield,
            x="Year",
            y=TARGET,
            marker="o",
            color="#d18445",
            ax=ax,
        )
        ax.set_title("Average recorded yield over time")
        ax.set_ylabel("Mean yield (hg/ha)")
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "This dashboard is an educational scenario tool. It does not forecast weather "
        "and should not be used as professional agricultural advice."
    )


elif page == "Explore data":
    show_header(
        "Explore the Dataset",
        "Filter the historical records to understand differences between countries and crops.",
    )

    filter1, filter2 = st.columns(2)
    countries = sorted(data["Area"].unique())
    crops = sorted(data["Item"].unique())
    default_country = countries.index("India") if "India" in countries else 0

    selected_country = filter1.selectbox(
        "Country",
        countries,
        index=default_country,
    )
    available_crops = sorted(
        data.loc[data["Area"] == selected_country, "Item"].unique()
    )
    selected_crop = filter2.selectbox("Crop", available_crops)

    filtered = data[
        (data["Area"] == selected_country) & (data["Item"] == selected_crop)
    ].sort_values("Year")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Available records", len(filtered))
    metric2.metric("Average yield", f"{filtered[TARGET].mean():,.0f} hg/ha")
    metric3.metric("Latest year", int(filtered["Year"].max()))

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.lineplot(
            data=filtered,
            x="Year",
            y=TARGET,
            marker="o",
            color="#2f6f3e",
            ax=ax,
        )
        ax.set_title(f"{selected_crop} yield in {selected_country}")
        ax.set_ylabel("Yield (hg/ha)")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        chart_data = filtered.set_index("Year")[[
            "average_rain_fall_mm_per_year",
            "avg_temp",
        ]]
        normalized = (chart_data - chart_data.min()) / (
            chart_data.max() - chart_data.min()
        ).replace(0, 1)
        normalized.columns = ["Rainfall", "Temperature"]
        st.markdown("#### Relative weather trend")
        st.caption("Values are scaled from 0 to 1 so both variables can share one chart.")
        st.line_chart(normalized)

    with st.expander("View filtered data"):
        st.dataframe(filtered, width="stretch", hide_index=True)


elif page == "Predict yield":
    show_header(
        "Predict Crop Yield",
        "Choose a model and supply future weather assumptions to estimate crop yield.",
    )

    model_name = st.selectbox(
        "Prediction model",
        MODEL_OPTIONS,
        help="Stacking has the lowest 2006–2009 validation RMSE of all models (15,616 hg/ha). "
             "Simple Average, Weighted Average and Stacking combine Extra Trees, Histogram "
             "Gradient Boosting and the neural network.",
    )
    st.caption("Default: Extra Trees, the best single model on 2006–2009 validation. "
               "Best overall by validation RMSE: Stacking, which takes longest to train. "
               "Models are fitted on all 1990–2013 records for scenario predictions. "
               "The first prediction trains the selected model and may take a minute "
               "(several minutes for Stacking).")

    countries = sorted(data["Area"].unique())
    crops = sorted(data["Item"].unique())
    default_country = countries.index("India") if "India" in countries else 0

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        country = col1.selectbox("Country", countries, index=default_country)
        crop = col2.selectbox("Crop", crops)

        col3, col4 = st.columns(2)
        year = col3.slider("Year", 1990, 2030, 2014)
        rainfall = col4.number_input(
            "Average rainfall (mm/year)",
            min_value=0.0,
            max_value=5000.0,
            value=float(data["average_rain_fall_mm_per_year"].median()),
            step=10.0,
        )

        col5, col6 = st.columns(2)
        pesticides = col5.number_input(
            "Pesticide use (tonnes)",
            min_value=0.0,
            max_value=500000.0,
            value=float(data["pesticides_tonnes"].median()),
            step=100.0,
        )
        temperature = col6.number_input(
            "Average temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=float(data["avg_temp"].median()),
            step=0.1,
        )

        submitted = st.form_submit_button("Predict yield", type="primary")

    if submitted:
        input_data = pd.DataFrame(
            [
                {
                    "Area": country,
                    "Item": crop,
                    "Year": year,
                    "average_rain_fall_mm_per_year": rainfall,
                    "pesticides_tonnes": pesticides,
                    "avg_temp": temperature,
                }
            ]
        )

        with st.spinner(training_message(model_name)):
            model = train_model(data, model_name)
        predicted_yield = float(model.predict(input_data)[0])
        if not np.isfinite(predicted_yield) or predicted_yield < 0:
            st.error("This model produced an invalid yield for this scenario. Review the inputs or try another model.")
            st.stop()
        tonnes_per_hectare = predicted_yield / 10_000

        st.success(f"Prediction completed using {model_name}")
        result1, result2 = st.columns(2)
        result1.metric("Predicted yield", f"{predicted_yield:,.0f} hg/ha")
        result2.metric("Converted yield", f"{tonnes_per_hectare:,.2f} tonnes/ha")

        st.markdown("#### Scenario used")
        st.dataframe(input_data, width="stretch", hide_index=True)

        if year > int(data["Year"].max()):
            st.warning(
                "This year is beyond the observed history ending in 2013. "
                "Tree models do not extend time trends beyond learned splits, and neural-network "
                "extrapolation is also uncertain. Distant-year predictions need recent data and validation."
            )

        for feature in NUMERIC_FEATURES[1:]:
            if not data[feature].min() <= input_data.iloc[0][feature] <= data[feature].max():
                st.warning(f"{feature} is outside the observed historical range.")
        if data[(data.Area == country) & (data.Item == crop)].empty:
            st.warning("This country/crop combination has no observed records in the dataset.")

        st.caption(
            "Yield is conditional on the supplied weather and pesticide assumptions. "
            "This does not forecast those inputs or provide a prediction interval."
        )

        st.markdown("#### Why this prediction? (SHAP)")
        try:
            with st.spinner("Explaining the prediction with SHAP..."):
                explanation = explain_scenario(model, model_name, input_data, shap_background(data))
                from explainability import explain_single_prediction, plot_waterfall

                fig = plot_waterfall(explanation, 0, title=f"Why {model_name} predicts "
                                     f"{predicted_yield:,.0f} hg/ha")
        except Exception as error:  # keep the prediction visible if SHAP fails
            st.warning(f"The SHAP explanation could not be computed: {error}")
        else:
            st.write(explain_single_prediction(explanation, 0))
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                f"Start: the average prediction over {SHAP_BACKGROUND_ROWS} sampled historical rows. "
                "Each bar shows how many hg/ha one input pushed this prediction up (red) or "
                "down (blue); together they add up exactly to the prediction. "
                "SHAP describes how the model uses the inputs, not what causes yield in the field."
            )


else:
    show_header(
        "Model Performance",
        "Compare the original models, the advanced chronological experiment, and the ensembles.",
    )

    experiment = st.radio("Experiment", ["Advanced models", "Ensembles", "Original models"],
                          horizontal=True)
    if experiment == "Ensembles":
        performance = ensemble_results()
        st.success("Best by validation RMSE: Stacking (15,616.21 hg/ha), which also has the "
                   "lowest test MAE and RMSE.")
        st.write("Three ensembles combine Extra Trees, Histogram Gradient Boosting and the "
                 "neural network. Same protocol as the advanced models: weights and the "
                 "stacking meta-model are chosen with 1990–2005 training and 2006–2009 "
                 "validation; the final refit on 1990–2009 is tested on 2010–2013, which is "
                 "never used for choosing weights or models.")
        st.caption("Saved results from ensemble_models.py; this page does not retrain the "
                   "evaluation models. Errors are in hg/ha. The Weighted Average validation RMSE "
                   "is slightly optimistic because its weights were computed from those years.")
        st.dataframe(
            performance.style.format({"Validation RMSE": "{:,.2f}", "Test MAE": "{:,.2f}",
                                      "Test RMSE": "{:,.2f}", "Test R²": "{:.4f}"}),
            width="stretch",
            hide_index=True,
        )

        rmse_long = performance.melt(
            id_vars="Model", value_vars=["Validation RMSE", "Test RMSE"],
            var_name="Period", value_name="RMSE",
        )
        fig, ax = plt.subplots(figsize=(10, 4.8))
        sns.barplot(data=rmse_long, y="Model", x="RMSE", hue="Period",
                    palette=["#0b4650", "#d2a33a"], ax=ax)
        ax.set_title("Validation (2006–2009) vs test (2010–2013) RMSE — lower is better")
        ax.set_xlabel("RMSE (hg/ha)")
        ax.set_ylabel("")
        ax.legend(title="")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        left, right = st.columns(2)
        with left:
            st.markdown("#### Weighted Average weights")
            st.dataframe(
                pd.DataFrame({"Base model": list(ENSEMBLE_WEIGHTS),
                              "Weight": list(ENSEMBLE_WEIGHTS.values())})
                .style.format({"Weight": "{:.4f}"}),
                width="stretch",
                hide_index=True,
            )
            st.caption("Weight ∝ 1 / validation RMSE², so a model with half the error gets "
                       "four times the weight. Weights sum to 1.")
        with right:
            st.markdown("#### Stacking meta-model")
            st.dataframe(
                pd.DataFrame({"Term": list(STACKING_COEFFICIENTS) + ["Intercept"],
                              "Value": list(STACKING_COEFFICIENTS.values()) + [STACKING_INTERCEPT]})
                .style.format({"Value": "{:,.4f}"}),
                width="stretch",
                hide_index=True,
            )
            st.caption("Linear regression with non-negative coefficients, trained on "
                       "out-of-fold predictions from expanding-window two-year time folds.")

        st.write("Stacking learned to rely mostly on Extra Trees with a smaller neural-network "
                 "share and gave Gradient Boosting zero weight. Combining models whose errors "
                 "are only partly correlated lowered test RMSE by 8.4% against Extra Trees alone. "
                 "This is one historical time window, not a guarantee of future performance.")
        with st.expander("How the ensembles work"):
            st.markdown("- **Simple Average:** each base model gets one third of the vote.\n"
                        "- **Weighted Average:** better validation models get larger fixed weights.\n"
                        "- **Stacking:** for each two-year block, base models trained only on earlier "
                        "years predict that block; a linear meta-model learns how to combine these "
                        "out-of-fold predictions, then the base models are refitted on all training data.")

        with st.expander("Global SHAP importance (Extra Trees)"):
            st.write("Mean absolute SHAP value per input for 100 sampled historical rows: on "
                     "average, how many hg/ha each input moves the default model's prediction.")
            if st.button("Compute SHAP importance"):
                importance = extra_trees_importance(data)
                value_column = importance.columns[1]
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=importance, y="Feature", x=value_column, color="#0b4650", ax=ax)
                ax.set_title("Feature importance — mean |SHAP|")
                ax.set_xlabel("Mean |SHAP value| (hg/ha)")
                ax.set_ylabel("")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Uses the Extra Trees model fitted for the prediction page (trained "
                           "first if needed). SHAP shows model behaviour, not causation.")
        st.stop()

    if experiment == "Advanced models":
        performance = advanced_results()
        st.success("Default model: Extra Trees — lowest validation RMSE (16,102.478 hg/ha).")
        st.write("Model selection: fit 1990–2005, validate 2006–2009. "
                 "Final evaluation: refit 1990–2009, test 2010–2013. "
                 "The neural network has the lowest final-test RMSE, while Extra Trees has the "
                 "lowest final-test MAE. We keep the model chosen before viewing the test results.")
        st.caption("Saved results from Advanced_Crop_Yield_Prediction.ipynb; "
                   "this page does not retrain the evaluation models. Errors are in hg/ha; R² is not percentage accuracy.")
        st.dataframe(performance, width="stretch", hide_index=True)
        left, right = st.columns(2)
        for column, metric in [(left, "Test RMSE"), (right, "Test R²")]:
            with column:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=performance, y="Model", x=metric, color="#0b4650", ax=ax)
                ax.set_title(metric + (" — lower is better" if metric == "Test RMSE" else " — higher is better"))
                ax.set_ylabel("")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        st.write("Extra Trees reduced test RMSE by 2.88% against the freshly rerun Random Forest "
                 "reference. This is one historical time window, not a guarantee of future performance.")
        with st.expander("How the advanced models work"):
            st.markdown("- **Extra Trees:** averages trees with randomized split thresholds.\n"
                        "- **Gradient Boosting:** adds trees to correct earlier prediction errors.\n"
                        "- **Neural network:** three hidden layers (128, 64, 32 neurons) learn nonlinear feature combinations.\n"
                        "- **Random Forest reference:** checks whether the advanced models improve on the earlier approach.")
        st.stop()

    performance = pd.DataFrame(
        [
            ["Random 80/20", "Random Forest (tuned)", 3452.628, 9419.223, 0.988, 0.998],
            ["Random 80/20", "Decision Tree", 3705.677, 12325.318, 0.979, 1.000],
            ["Random 80/20", "Linear Regression", 29582.557, 42144.051, 0.755, 0.756],
            ["Time 2010–2013", "Random Forest (tuned)", 10321.933, 20371.908, 0.955, 0.999],
            ["Time 2010–2013", "Decision Tree", 11631.345, 23551.293, 0.939, 1.000],
            ["Time 2010–2013", "Linear Regression", 33462.733, 50112.199, 0.725, 0.763],
        ],
        columns=["Split", "Model", "MAE", "RMSE", "Test R²", "Train R²"],
    )

    st.dataframe(
        performance.style.format(
            {"MAE": "{:,.3f}", "RMSE": "{:,.3f}", "Test R²": "{:.3f}", "Train R²": "{:.3f}"}
        ),
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=performance,
            x="Model",
            y="RMSE",
            hue="Split",
            palette=["#0b4650", "#d2a33a"],
            ax=ax,
        )
        ax.set_title("RMSE comparison")
        ax.tick_params(axis="x", rotation=15)
        ax.set_xlabel("")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=performance,
            x="Model",
            y="Test R²",
            hue="Split",
            palette=["#0b4650", "#d2a33a"],
            ax=ax,
        )
        ax.set_title("Test R² comparison")
        ax.tick_params(axis="x", rotation=15)
        ax.set_xlabel("")
        st.pyplot(fig)
        plt.close(fig)

    st.success(
        "The tuned Random Forest is the strongest model on both evaluation splits."
    )
    st.write(
        "The Decision Tree reaches a training R² of 1.000, which is a clear sign of "
        "overfitting. Random Forest reduces this problem by averaging many trees. The "
        "time-based result is lower because predicting later years is more difficult "
        "and is a more realistic test of future performance."
    )

