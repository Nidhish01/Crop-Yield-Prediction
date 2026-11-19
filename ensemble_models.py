"""Ensemble models for future crop-yield prediction.

An ensemble combines the predictions of several base models.  The three base
models (Extra Trees, Histogram Gradient Boosting and a Neural Network) are
reused from ``advanced_future_yield_models.py``.  Because they learn different
kinds of patterns, their errors are only partly correlated, and combining them
can reduce the overall error.

Three ensembles are built:

1. Simple Average   - every base model gets the same weight (1/3 each).
2. Weighted Average - better models (lower validation RMSE) get larger weights.
3. Stacking         - a small "meta-model" (a linear regression) LEARNS how to
                      combine the base predictions, using out-of-fold
                      predictions made with time-based folds.

Evaluation protocol (same as the Advanced Models notebook):
    * Model selection / weights : train 1990-2005, validate on 2006-2009.
    * Final evaluation          : refit on 1990-2009, test on 2010-2013.
The test years 2010-2013 are NEVER used to choose weights, fit the meta-model
or select a model.  An assertion in ``run_experiment`` checks this.

Run:
    python ensemble_models.py
"""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from advanced_future_yield_models import load_data, make_models, make_preprocessor


FEATURES = [
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]
TARGET = "hg/ha_yield"
BASE_MODEL_NAMES = ["Extra Trees", "Histogram Gradient Boosting", "Neural Network (MLP)"]
ENSEMBLE_NAMES = ["Simple Average", "Weighted Average", "Stacking"]

# Time-based protocol.
TRAIN_END_YEAR = 2005        # selection phase: train 1990-2005
VALIDATION_END_YEAR = 2009   # selection phase: validate 2006-2009
# Final phase: refit on 1990-2009 and test on 2010-2013.

# Two-year blocks used by stacking to create out-of-fold predictions.
DEFAULT_FOLD_YEARS = ((2002, 2003), (2004, 2005), (2006, 2007), (2008, 2009))
# During the validation phase only data up to 2005 is available.
VALIDATION_FOLD_YEARS = ((2000, 2001), (2002, 2003), (2004, 2005))


# ---------------------------------------------------------------------------
# Base models and weights
# ---------------------------------------------------------------------------
def make_base_models():
    """Return fresh, unfitted pipelines for the three base models.

    Each pipeline contains its own preprocessing (one-hot encoding + scaling),
    so every model can be given the raw feature DataFrame directly.
    """
    all_models = make_models(make_preprocessor())
    return {name: all_models[name] for name in BASE_MODEL_NAMES}


def inverse_rmse_weights(val_rmse):
    """Turn validation RMSE values into weights that sum to 1.

    Choice: weight is proportional to 1 / RMSE**2 (= 1 / MSE).  The MSE is an
    estimate of a model's error variance, and "inverse-variance weighting" is
    the classic way to combine independent estimates: a model with half the
    RMSE receives four times the weight.  Using 1/RMSE instead would give
    gentler weights; 1/RMSE**2 rewards the better models more clearly.
    """
    inverse = {name: 1.0 / (rmse ** 2) for name, rmse in val_rmse.items()}
    total = sum(inverse.values())
    return {name: value / total for name, value in inverse.items()}


def _normalise_weights(names, weights):
    """Return a weight dict over ``names`` that sums to 1 (equal if None)."""
    if weights is None:
        return {name: 1.0 / len(names) for name in names}
    missing = set(names).difference(weights)
    if missing:
        raise ValueError(f"Weights missing for: {sorted(missing)}")
    total = sum(weights[name] for name in names)
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return {name: weights[name] / total for name in names}


def _predict_all(fitted_models, X):
    """Predict with every fitted model; one column per model."""
    return pd.DataFrame(
        {name: model.predict(X) for name, model in fitted_models.items()},
        index=X.index,
    )


# ---------------------------------------------------------------------------
# Ensemble 1 and 2: (weighted) averaging
# ---------------------------------------------------------------------------
class AveragingEnsemble(BaseEstimator, RegressorMixin):
    """Average the predictions of several base models.

    prediction = sum_i  weight_i * prediction_i     (weights sum to 1)

    Parameters
    ----------
    base_models : dict name -> unfitted estimator, or None
        None uses ``make_base_models()``.
    weights : dict name -> weight, or None
        None gives every model the same weight (Simple Average).
    """

    def __init__(self, base_models=None, weights=None):
        self.base_models = base_models
        self.weights = weights

    def fit(self, X, y):
        """Clone and fit every base model, then store normalised weights."""
        base_models = self.base_models if self.base_models is not None else make_base_models()
        self.fitted_models_ = {
            name: clone(model).fit(X, y) for name, model in base_models.items()
        }
        self.weights_ = _normalise_weights(list(self.fitted_models_), self.weights)
        return self

    def base_predictions(self, X):
        """Return a DataFrame with one prediction column per base model."""
        return _predict_all(self.fitted_models_, X)

    def predict(self, X):
        """Weighted average of the base-model predictions."""
        base = self.base_predictions(X)
        weights = np.array([self.weights_[name] for name in base.columns])
        return base.to_numpy() @ weights


def averaging_from_fitted(fitted_models, weights=None):
    """Build an AveragingEnsemble from base models that are ALREADY fitted.

    This only saves time: the result is identical to calling ``fit`` on the
    same training data, because ``fit`` would train exactly the same models
    (same data, same random_state).
    """
    ensemble = AveragingEnsemble(base_models=dict(fitted_models), weights=weights)
    ensemble.fitted_models_ = dict(fitted_models)
    ensemble.weights_ = _normalise_weights(list(fitted_models), weights)
    return ensemble


# ---------------------------------------------------------------------------
# Ensemble 3: stacking
# ---------------------------------------------------------------------------
class StackingEnsemble(BaseEstimator, RegressorMixin):
    """Stacking: a meta-model learns how to combine the base predictions.

    Why out-of-fold (OOF) predictions?  If the meta-model were trained on
    predictions the base models made for their OWN training rows, it would see
    over-optimistic (almost perfect) predictions and trust the most overfitted
    model.  Instead, each base model predicts rows it has never seen.

    Why EXPANDING-WINDOW time folds?  Our task is predicting the future, so
    for every block of years (e.g. 2006-2007) the base models are trained only
    on EARLIER years (Year < 2006) and then predict that block.  This imitates
    the real situation and avoids using the future to predict the past.

    Parameters
    ----------
    base_models : dict name -> unfitted estimator, or None (make_base_models()).
    meta_model : estimator or None
        None uses LinearRegression(positive=True): non-negative combination
        weights plus an intercept, which is easy to interpret.
    fold_years : tuple of (start, end) year blocks used to build OOF predictions.
    """

    def __init__(self, base_models=None, meta_model=None, fold_years=DEFAULT_FOLD_YEARS):
        self.base_models = base_models
        self.meta_model = meta_model
        self.fold_years = fold_years

    def fit(self, X, y, fitted_base_models=None):
        """Fit the meta-model on OOF predictions, then refit bases on all of X.

        ``fitted_base_models`` is an optional time-saver: base models already
        fitted on exactly this (X, y).  If given, they are used as the final
        base models instead of refitting them.
        """
        if "Year" not in X.columns:
            raise ValueError("StackingEnsemble needs a 'Year' column to build time folds.")

        base_models = self.base_models if self.base_models is not None else make_base_models()
        years = X["Year"]

        # 1. Out-of-fold predictions with expanding-window time folds.
        oof_parts, target_parts, used_blocks = [], [], []
        for start, end in self.fold_years:
            block_mask = (years >= start) & (years <= end)
            history_mask = years < start
            if not block_mask.any() or not history_mask.any():
                continue  # block outside X's years, or nothing earlier to learn from

            x_hist, y_hist = X.loc[history_mask], y.loc[history_mask]
            x_block = X.loc[block_mask]
            block_preds = pd.DataFrame(
                {
                    name: clone(model).fit(x_hist, y_hist).predict(x_block)
                    for name, model in base_models.items()
                },
                index=x_block.index,
            )
            oof_parts.append(block_preds)
            target_parts.append(y.loc[block_mask])
            used_blocks.append((start, end))

        if not oof_parts:
            raise ValueError("No usable fold blocks: check fold_years against X's years.")

        self.oof_predictions_ = pd.concat(oof_parts)
        self.oof_target_ = pd.concat(target_parts)
        self.used_fold_years_ = tuple(used_blocks)

        # 2. Fit the meta-model on the OOF predictions only.
        meta = self.meta_model if self.meta_model is not None else LinearRegression(positive=True)
        self.meta_model_ = clone(meta).fit(self.oof_predictions_, self.oof_target_)
        if hasattr(self.meta_model_, "coef_"):
            self.coef_ = dict(zip(self.oof_predictions_.columns, np.ravel(self.meta_model_.coef_)))
            self.intercept_ = float(np.ravel(self.meta_model_.intercept_)[0])

        # 3. Refit every base model on ALL of X for making final predictions.
        if fitted_base_models is not None:
            self.fitted_models_ = {name: fitted_base_models[name] for name in base_models}
        else:
            self.fitted_models_ = {
                name: clone(model).fit(X, y) for name, model in base_models.items()
            }
        return self

    def base_predictions(self, X):
        """Return a DataFrame with one prediction column per base model."""
        return _predict_all(self.fitted_models_, X)

    def predict(self, X):
        """Feed the base predictions into the meta-model."""
        return self.meta_model_.predict(self.base_predictions(X))


def make_ensemble(kind, weights=None):
    """Return an unfitted ensemble of the requested kind (see ENSEMBLE_NAMES)."""
    if kind == "Simple Average":
        return AveragingEnsemble(weights=None)
    if kind == "Weighted Average":
        if weights is None:
            raise ValueError("'Weighted Average' needs weights (see inverse_rmse_weights).")
        return AveragingEnsemble(weights=weights)
    if kind == "Stacking":
        return StackingEnsemble()
    raise ValueError(f"Unknown ensemble kind '{kind}'. Choose from {ENSEMBLE_NAMES}.")


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_experiment(data=None, verbose=True):
    """Run the validation phase and the final test phase for all models.

    Returns a dict with: results (DataFrame), weights, stacking_coef, fitted
    (name -> fitted estimator from the test phase), x_train, y_train,
    x_test, y_test.
    """
    log = print if verbose else (lambda *args, **kwargs: None)
    start_time = perf_counter()

    if data is None:
        data = load_data()
    x_all, y_all = data[FEATURES], data[TARGET]

    # Everything before the test phase uses ONLY years <= 2009.
    development_mask = data["Year"] <= VALIDATION_END_YEAR
    x_dev, y_dev = x_all.loc[development_mask], y_all.loc[development_mask]
    assert x_dev["Year"].max() <= VALIDATION_END_YEAR, "Test years leaked into development data!"

    # ---------------- Phase 1: validation (train 1990-2005, validate 2006-2009)
    fit_mask = x_dev["Year"] <= TRAIN_END_YEAR
    x_fit, y_fit = x_dev.loc[fit_mask], y_dev.loc[fit_mask]
    x_val, y_val = x_dev.loc[~fit_mask], y_dev.loc[~fit_mask]
    assert x_fit["Year"].max() <= TRAIN_END_YEAR
    assert x_val["Year"].between(TRAIN_END_YEAR + 1, VALIDATION_END_YEAR).all()

    log(f"Validation phase: train {x_fit['Year'].min()}-{x_fit['Year'].max()} "
        f"({len(x_fit):,} rows), validate {x_val['Year'].min()}-{x_val['Year'].max()} "
        f"({len(x_val):,} rows)")

    val_rmse = {}
    val_fitted = {}
    for name, model in make_base_models().items():
        log(f"  fitting {name} ...")
        val_fitted[name] = model.fit(x_fit, y_fit)
        val_rmse[name] = _rmse(y_val, val_fitted[name].predict(x_val))

    weights = inverse_rmse_weights(val_rmse)

    # Note: the weighted average's validation RMSE is slightly optimistic,
    # because its weights were computed from this same validation period.
    val_rmse["Simple Average"] = _rmse(y_val, averaging_from_fitted(val_fitted).predict(x_val))
    val_rmse["Weighted Average"] = _rmse(
        y_val, averaging_from_fitted(val_fitted, weights).predict(x_val)
    )

    log("  fitting Stacking (validation folds) ...")
    val_stack = StackingEnsemble(fold_years=VALIDATION_FOLD_YEARS)
    val_stack.fit(x_fit, y_fit, fitted_base_models=val_fitted)
    val_rmse["Stacking"] = _rmse(y_val, val_stack.predict(x_val))

    # ---------------- Phase 2: final test (refit 1990-2009, test 2010-2013)
    x_train, y_train = x_dev, y_dev
    test_mask = data["Year"] > VALIDATION_END_YEAR
    x_test, y_test = x_all.loc[test_mask], y_all.loc[test_mask]

    log(f"Test phase: train {x_train['Year'].min()}-{x_train['Year'].max()} "
        f"({len(x_train):,} rows), test {x_test['Year'].min()}-{x_test['Year'].max()} "
        f"({len(x_test):,} rows)")

    fitted = {}
    for name, model in make_base_models().items():
        log(f"  fitting {name} ...")
        fitted[name] = model.fit(x_train, y_train)
    base_fitted = {name: fitted[name] for name in BASE_MODEL_NAMES}

    fitted["Simple Average"] = averaging_from_fitted(base_fitted)
    fitted["Weighted Average"] = averaging_from_fitted(base_fitted, weights)
    log("  fitting Stacking (default folds) ...")
    stacking = StackingEnsemble()
    stacking.fit(x_train, y_train, fitted_base_models=base_fitted)
    fitted["Stacking"] = stacking

    rows = []
    for name in BASE_MODEL_NAMES + ENSEMBLE_NAMES:
        predictions = fitted[name].predict(x_test)
        rows.append(
            {
                "Model": name,
                "Type": "Base" if name in BASE_MODEL_NAMES else "Ensemble",
                "Validation RMSE": val_rmse[name],
                "Test MAE": mean_absolute_error(y_test, predictions),
                "Test RMSE": _rmse(y_test, predictions),
                "Test R2": r2_score(y_test, predictions),
            }
        )
    results = pd.DataFrame(rows).sort_values("Validation RMSE").reset_index(drop=True)

    stacking_coef = dict(stacking.coef_)
    stacking_coef["intercept"] = stacking.intercept_

    log(f"Total runtime: {perf_counter() - start_time:.1f} s")
    return {
        "results": results,
        "weights": weights,
        "stacking_coef": stacking_coef,
        "fitted": fitted,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }


def main():
    output = run_experiment()

    printable = output["results"].copy()
    for column in ["Validation RMSE", "Test MAE", "Test RMSE"]:
        printable[column] = printable[column].round(2)
    printable["Test R2"] = printable["Test R2"].round(4)

    print("\nResults (errors in hg/ha; sorted by validation RMSE):")
    print(printable.to_string(index=False))

    print("\nWeighted-average weights (proportional to 1 / validation RMSE^2):")
    for name, weight in output["weights"].items():
        print(f"  {name}: {weight:.4f}")

    print("\nStacking meta-model (LinearRegression, positive=True):")
    for name, value in output["stacking_coef"].items():
        print(f"  {name}: {value:.4f}")

    best = output["results"].iloc[0]["Model"]
    print(f"\nBest model by validation RMSE: {best}")


if __name__ == "__main__":
    main()
