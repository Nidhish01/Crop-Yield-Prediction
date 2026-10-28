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
        .stApp {background-color: #f6f8f3;}
        [data-testid="stSidebar"] {background-color: #eaf2e3;}
        .main-title {color: #245c32; font-size: 2.5rem; font-weight: 750;}
        .subtitle {color: #52634f; font-size: 1.05rem; margin-bottom: 1.5rem;}
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #dce7d7;
            border-radius: 12px;
            padding: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
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
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


data = load_data()

st.sidebar.title("🌾 Crop Yield ML")
st.sidebar.caption("BTech Machine Learning Lab Project")
page = st.sidebar.radio(
    "Navigate",
    ["Project Overview", "Explore Data", "Predict Yield", "Model Performance"],
)
st.sidebar.divider()
st.sidebar.markdown("**Team Members**")
st.sidebar.write("Nidhish Bansal")
st.sidebar.write("Yash Vishnoi")


if page == "Project Overview":
    show_header(
        "Crop Yield Prediction Dashboard",
        "Explore historical agricultural data and estimate crop yield with machine learning.",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", f"{len(data):,}")
    col2.metric("Countries", data["Area"].nunique())
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


elif page == "Explore Data":
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


elif page == "Predict Yield":
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
        sns.barplot(data=performance, x="Model", y="RMSE", hue="Split", ax=ax)
        ax.set_title("RMSE comparison")
        ax.tick_params(axis="x", rotation=15)
        ax.set_xlabel("")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=performance, x="Model", y="Test R²", hue="Split", ax=ax)
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

