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
                 "Histogram Gradient Boosting", "Random Forest (reference)"]


def advanced_results():
    """Recorded, rounded results from the executed advanced notebook."""
    return pd.DataFrame([
        ["Extra Trees", 16102.478, 10152.975, 19942.622, 0.956, 0.997],
        ["Neural Network (MLP)", 18661.863, 11221.705, 19397.042, 0.959, 0.991],
        ["Histogram Gradient Boosting", 20919.691, 13459.170, 23347.448, 0.940, 0.977],
        ["Random Forest (reference)", 16732.869, 10442.003, 20533.341, 0.954, 0.997],
    ], columns=["Model", "Validation RMSE", "Test MAE", "Test RMSE", "Test R²", "Train R²"])


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


@st.cache_resource(show_spinner="Training the selected model on historical data...", max_entries=1)
def train_model(data, model_name):
    """Fit only the requested model; keep one fitted model cached to limit memory."""
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
        "compare Extra Trees, Histogram Gradient Boosting, and a neural network. "
        "Extra Trees is the default prediction model because it had the lowest "
        "RMSE on the 2006–2009 validation period."
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

    model_name = st.selectbox("Prediction model", MODEL_OPTIONS)
    st.caption("Default: Extra Trees, selected using 2006–2009 validation. "
               "Models are fitted on all 1990–2013 records for scenario predictions. "
               "The first prediction trains the selected model and may take a minute.")

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


else:
    show_header(
        "Model Performance",
        "Compare the original models and the advanced chronological experiment.",
    )

    experiment = st.radio("Experiment", ["Advanced models", "Original models"], horizontal=True)
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

