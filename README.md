# Crop Yield Prediction

[**Open the live CropCast ML dashboard →**](https://cropcast-ml.streamlit.app/)

This BTech Machine Learning lab project uses historical agricultural and weather data to predict crop yield. It demonstrates the complete machine learning workflow, including data cleaning, exploratory data analysis, preprocessing, model training, hyperparameter tuning, evaluation, interpretation, and an interactive Streamlit dashboard.

## Team Members

- Nidhish Bansal
- Yash Vishnoi

## Dashboard Preview

These screenshots show the original dashboard. The current app also includes the advanced-model selector and comparison described below.

<table>
  <tr>
    <td width="50%">
      <strong>Project Overview</strong><br>
      <img src="screenshots/overview.png" alt="Crop yield dashboard overview">
    </td>
    <td width="50%">
      <strong>Explore Data</strong><br>
      <img src="screenshots/explore-data.png" alt="Crop yield data exploration dashboard">
    </td>
  </tr>
  <tr>
    <td width="50%">
      <strong>Predict Yield</strong><br>
      <img src="screenshots/predict-yield.png" alt="Interactive crop yield prediction form">
    </td>
    <td width="50%">
      <strong>Model Performance</strong><br>
      <img src="screenshots/model-performance.png" alt="Machine learning model comparison dashboard">
    </td>
  </tr>
</table>

## Project Objective

Crop yield depends on several factors, such as rainfall, temperature, pesticide usage, crop type, location, and year. The objective of this project is to compare multiple regression algorithms and identify the model that predicts crop yield most accurately.

The prediction target is `hg/ha_yield`, measured in hectograms per hectare.

> 10,000 hg/ha = 1 tonne/hectare

## Dataset

The merged dataset contains 28,242 observations covering:

- 101 countries
- 10 crops
- Years from 1990 to 2013
- Average annual rainfall
- Average temperature
- Pesticide usage
- Recorded crop yield

The `dataset` folder also contains the original yield, rainfall, temperature, and pesticide CSV files.

## Machine Learning Models

The original notebook compares three regression models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The Random Forest is tuned using a small grid search with expanding-year validation.

### Advanced Models and Future Scenarios

[`Advanced_Crop_Yield_Prediction.ipynb`](Advanced_Crop_Yield_Prediction.ipynb) adds Histogram Gradient Boosting, Extra Trees, and a neural network with three hidden layers (128, 64, 32 neurons), alongside a freshly rerun Random Forest reference. It includes executed results, eight figures, simple explanations, and viva questions.

Model selection uses **1990–2005 training and 2006–2009 validation**. Final evaluation refits on **1990–2009** and tests on **2010–2013**.

| Model | Validation RMSE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Extra Trees | 16,102.478 | 10,152.975 | 19,942.622 | 0.956 |
| Neural Network | 18,661.863 | 11,221.705 | 19,397.042 | 0.959 |
| Histogram Gradient Boosting | 20,919.691 | 13,459.170 | 23,347.448 | 0.940 |
| Random Forest reference | 16,732.869 | 10,442.003 | 20,533.341 | 0.954 |

Errors are in hg/ha. **Extra Trees is the dashboard default** because it won on validation RMSE. The neural network had the lowest final-test RMSE, but the test period is not used to change the selected model. Extra Trees reduced test RMSE by 2.88% against this rerun reference. R² is not percentage accuracy.

The prediction page lets you choose any of these four models. It fits the requested model on all available 1990–2013 data and caches one fitted model at a time. Supply a year and weather/pesticide assumptions to obtain a conditional yield estimate. First-time training may take a minute. All models learn from history; a neural network does not guarantee reliable extrapolation beyond 2013.

[`advanced_future_yield_models.py`](advanced_future_yield_models.py) contains the standalone initial advanced experiment and reusable model definitions used by the dashboard. Its standalone winner is selected using test RMSE; use the separate notebook's earlier-validation procedure for the more defensible lab comparison.

## Evaluation

The models are tested using two approaches:

- **Random 80/20 split:** Measures performance on randomly selected historical records.
- **Time-based split:** Trains on 1990–2009 and tests on 2010–2013 to better represent future prediction.

The evaluation metrics are MAE, RMSE, and R².

### Original Notebook Results

| Evaluation split | Best model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Random 80/20 | Tuned Random Forest | 3,452.628 | 9,419.223 | 0.988 |
| Time 2010–2013 | Tuned Random Forest | 10,321.933 | 20,371.908 | 0.955 |

The Random Forest performed best because it captured nonlinear relationships while reducing the overfitting seen in a single Decision Tree. The time-based split produced a larger error because predicting future years is more difficult and realistic than predicting randomly selected historical records.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Nidhish01/Crop-Yield-Prediction.git
   cd Crop-Yield-Prediction
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `Crop_Yield_Prediction.ipynb` or `Advanced_Crop_Yield_Prediction.ipynb` and select **Run All**.

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard provides four sections:

- Project summary and dataset statistics
- Country- and crop-based data exploration
- Interactive crop-yield prediction with an advanced-model selector
- Original and advanced model-performance comparisons

## Repository Structure

```text
Crop-Yield-Prediction/
|-- .streamlit/
|   `-- config.toml
|-- Crop_Yield_Prediction.ipynb
|-- Advanced_Crop_Yield_Prediction.ipynb
|-- advanced_future_yield_models.py
|-- app.py
|-- README.md
|-- requirements.txt
|-- screenshots/
|   |-- overview.png
|   |-- explore-data.png
|   |-- predict-yield.png
|   `-- model-performance.png
`-- dataset/
    |-- yield_df.csv
    |-- yield.csv
    |-- rainfall.csv
    |-- temp.csv
    `-- pesticides.csv
```

## Limitations

- The weather values are annual averages and do not capture seasonal conditions or extreme events.
- Important factors such as soil quality, irrigation, fertilizer use, and crop variety are unavailable.
- This is an educational prediction project, not a production agricultural advisory system.
