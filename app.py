from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
        .team-name {color: white; font-size: 0.86rem; margin: 0.25rem 0;}

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


@st.cache_resource(show_spinner="Training the Random Forest model...")
def train_model(data):
    """Train the same tuned Random Forest used in the notebook."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numbers", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
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
        <div class="team-name">Nidhish Bansal</div>
        <div class="team-name">Yash Vishnoi</div>
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
        "pesticide use, and time. This project compares Linear Regression, a Decision "
        "Tree, and a Random Forest. The tuned Random Forest produced the best results."
    )

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
        "Enter an agricultural scenario and let the trained Random Forest estimate its yield.",
    )

    countries = sorted(data["Area"].unique())
    crops = sorted(data["Item"].unique())
    default_country = countries.index("India") if "India" in countries else 0

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        country = col1.selectbox("Country", countries, index=default_country)
        crop = col2.selectbox("Crop", crops)

        col3, col4 = st.columns(2)
        year = col3.slider("Year", 1990, 2030, 2013)
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

        model = train_model(data)
        predicted_yield = max(0.0, float(model.predict(input_data)[0]))
        tonnes_per_hectare = predicted_yield / 10_000

        st.success("Prediction completed")
        result1, result2 = st.columns(2)
        result1.metric("Predicted yield", f"{predicted_yield:,.0f} hg/ha")
        result2.metric("Converted yield", f"{tonnes_per_hectare:,.2f} tonnes/ha")

        st.markdown("#### Scenario used")
        st.dataframe(input_data, width="stretch", hide_index=True)

        if year > int(data["Year"].max()):
            st.warning(
                "This year is outside the 1990–2013 training period. Tree models cannot "
                "reliably extrapolate long-term trends, so interpret the result carefully."
            )

        st.caption(
            "The estimate comes from the tuned Random Forest trained on the complete "
            "historical dataset. It is intended for demonstration and learning only."
        )


else:
    show_header(
        "Model Performance",
        "Compare the three regression approaches used in the project notebook.",
    )

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

