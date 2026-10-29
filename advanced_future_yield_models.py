"""Compare advanced models for future crop-yield prediction.

The important part of this experiment is the time-based split.  Every model is
trained on 1990-2009 and tested on the later years 2010-2013.  This gives a more
honest estimate of future performance than randomly mixing old and new rows.

Run:
    python advanced_future_yield_models.py

Edit FUTURE_SCENARIO below to try a different country, crop, year, rainfall,
pesticide use, or temperature.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path(__file__).parent / "dataset" / "yield_df.csv"
TARGET = "hg/ha_yield"
TRAIN_END_YEAR = 2009

# Change these values to make another future prediction.  Weather and pesticide
# values for a future year are scenarios/assumptions; they are not forecasts.
FUTURE_SCENARIO = {
    "Area": "India",
    "Item": "Wheat",
    "Year": 2014,
    "average_rain_fall_mm_per_year": 1083.0,
    "pesticides_tonnes": 45620.0,
    "avg_temp": 27.0,
}


def load_data():
    """Load the dataset and perform small, transparent checks."""
    data = pd.read_csv(DATA_FILE)
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")

    required_columns = {
        "Area",
        "Item",
        "Year",
        TARGET,
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if data[list(required_columns)].isna().any().any():
        raise ValueError("The required columns contain missing values.")

    return data


def make_preprocessor():
    """Create leak-safe preprocessing for categories and numeric values."""
    categorical_features = ["Area", "Item"]
    numeric_features = [
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    ]

    return ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("numbers", StandardScaler(), numeric_features),
        ]
    )


def make_models(preprocessor):
    """Return three nonlinear models that learn different kinds of patterns."""
    model_definitions = {
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250,
            max_features=0.8,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "Neural Network (MLP)": TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                batch_size=256,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                max_iter=300,
                random_state=42,
            ),
            transformer=StandardScaler(),
        ),
    }

    return {
        name: Pipeline(
            steps=[
                ("preprocessing", clone(preprocessor)),
                ("model", model),
            ]
        )
        for name, model in model_definitions.items()
    }


def evaluate_models(models, x_train, y_train, x_test, y_test):
    """Fit each model and return both the metrics and fitted pipelines."""
    rows = []
    fitted_models = {}

    for name, pipeline in models.items():
        print(f"Training {name} ...")
        start = perf_counter()
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                "R2": r2_score(y_test, predictions),
                "Training seconds": perf_counter() - start,
            }
        )
        fitted_models[name] = pipeline

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results, fitted_models


def check_future_scenario(scenario, data):
    """Warn when the requested scenario is outside the known data coverage."""
    warnings = []

    if scenario["Area"] not in set(data["Area"]):
        warnings.append(f"country '{scenario['Area']}' was not present in training data")
    if scenario["Item"] not in set(data["Item"]):
        warnings.append(f"crop '{scenario['Item']}' was not present in training data")

    for column in [
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    ]:
        value = scenario[column]
        if not data[column].min() <= value <= data[column].max():
            warnings.append(f"{column}={value} is outside the historical range")

    return warnings


def main():
    data = load_data()

    feature_columns = [
        "Area",
        "Item",
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    ]
    x = data[feature_columns]
    y = data[TARGET]

    # Earlier years teach the models; later years imitate unseen future data.
    train_mask = data["Year"] <= TRAIN_END_YEAR
    test_mask = data["Year"] > TRAIN_END_YEAR
    x_train, y_train = x.loc[train_mask], y.loc[train_mask]
    x_test, y_test = x.loc[test_mask], y.loc[test_mask]

    print(f"Rows: {len(data):,}")
    print(f"Training: {x_train['Year'].min()}-{x_train['Year'].max()} ({len(x_train):,} rows)")
    print(f"Future test: {x_test['Year'].min()}-{x_test['Year'].max()} ({len(x_test):,} rows)\n")

    models = make_models(make_preprocessor())
    results, fitted_models = evaluate_models(
        models, x_train, y_train, x_test, y_test
    )

    printable_results = results.copy()
    for column in ["MAE", "RMSE", "Training seconds"]:
        printable_results[column] = printable_results[column].round(2)
    printable_results["R2"] = printable_results["R2"].round(4)

    print("\nPerformance on unseen future years (lower MAE/RMSE is better):")
    print(printable_results.to_string(index=False))

    best_name = results.iloc[0]["Model"]
    print(f"\nBest model by future-test RMSE: {best_name}")

    # Once model selection is complete, learn from every available year before
    # estimating the requested future scenario.
    final_model = clone(fitted_models[best_name])
    final_model.fit(x, y)

    warnings = check_future_scenario(FUTURE_SCENARIO, data)
    scenario_frame = pd.DataFrame([FUTURE_SCENARIO], columns=feature_columns)
    predicted_yield = max(0.0, float(final_model.predict(scenario_frame)[0]))

    print("\nFuture scenario:")
    for feature, value in FUTURE_SCENARIO.items():
        print(f"  {feature}: {value}")
    print(f"Predicted yield: {predicted_yield:,.0f} hg/ha")
    print(f"Predicted yield: {predicted_yield / 10_000:,.2f} tonnes/ha")

    if warnings:
        print("\nCaution:")
        for warning in warnings:
            print(f"  - {warning}")

    print(
        "\nThis is a scenario estimate, not a weather forecast. Its reliability "
        "depends on the future input assumptions and on patterns remaining similar "
        "to the historical data."
    )


if __name__ == "__main__":
    main()
