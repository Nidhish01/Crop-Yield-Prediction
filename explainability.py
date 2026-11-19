"""SHAP explainability helpers for the crop-yield models.

What is SHAP?
-------------
SHAP (SHapley Additive exPlanations) explains ONE prediction by splitting it
into a contribution per input feature.  It comes from cooperative game theory:
the features are "players" and the prediction is the "payout".  A feature's
Shapley value is its average extra effect on the prediction over every order in
which the features could be added to the model.

The key property (called *additivity* or *local accuracy*) is:

    base_value + sum(SHAP values of the row) == model prediction for that row

* base_value  = the average prediction (what the model says when it knows
                nothing specific about the row), in hg/ha.
* SHAP value  = how many hg/ha this feature pushed THIS prediction up (+) or
                down (-) compared with the base value.

Two ways of computing SHAP values are provided:

1. ``explain_tree_pipeline`` - for tree models (Extra Trees, Histogram Gradient
   Boosting).  ``shap.TreeExplainer`` walks the trees directly, so the values are
   exact and fast (no sampling).  It works on the *transformed* (one-hot
   encoded + scaled) columns, which we then add back up into the 6 original
   features.

2. ``explain_model`` - model-agnostic, for anything with ``.predict(df)``
   (the averaging/stacking ensembles, the MLP).  It only calls ``predict`` on the
   6 ORIGINAL features, so it does not care what is inside the model.  With just
   6 features there are only 2**6 = 64 feature subsets, so we can compute the
   EXACT Shapley values (``shap.explainers.Exact``) instead of estimating them.

Every function returns / accepts a ``shap.Explanation`` with
values (n, 6), base_values (n,), data = raw feature values (country and crop
names are kept as text), feature_names = readable labels.

All plot functions return a matplotlib Figure (they never call plt.show()), so
use ``st.pyplot(fig)`` in Streamlit or just display ``fig`` in a notebook.
"""

from __future__ import annotations

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

FEATURES = [
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]

FEATURE_LABELS = {
    "Area": "Country",
    "Item": "Crop",
    "Year": "Year",
    "average_rain_fall_mm_per_year": "Rainfall (mm/yr)",
    "pesticides_tonnes": "Pesticides (tonnes)",
    "avg_temp": "Avg temperature (°C)",
}

CATEGORICAL_FEATURES = ["Area", "Item"]
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
UNIT = "hg/ha"

_LABELS = [FEATURE_LABELS[f] for f in FEATURES]
_LABEL_TO_RAW = {label: raw for raw, label in FEATURE_LABELS.items()}


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------
def _thousands(value, _pos=None) -> str:
    return f"{value:,.0f}"


def _thousands_k(value, _pos=None) -> str:
    return "0" if value == 0 else f"{value / 1000:+,.0f}k"


_NUMBER = re.compile(r"[+\-−]?\d+(?:\.\d+)?")


def _reformat_number_text(text: str) -> str:
    """'−72536.03' -> '-72,536' and '$f(x)$ = 9905.792' -> '$f(x)$ = 9,906'."""
    def repl(match):
        raw = match.group(0).replace("−", "-")
        signed = raw[0] in "+-"
        number = float(raw)
        return f"{number:+,.0f}" if signed else f"{number:,.0f}"

    result = _NUMBER.sub(repl, text)
    if "$" in text:  # inside math text a bare "," gets a space after it
        result = result.replace(",", "{,}")
    return result


def _resolve_feature(feature: str) -> str:
    """Accept either the raw column name or its readable label."""
    if feature in FEATURES:
        return feature
    if feature in _LABEL_TO_RAW:
        return _LABEL_TO_RAW[feature]
    raise KeyError(
        f"Unknown feature '{feature}'. Use one of {FEATURES} or {_LABELS}."
    )


def _make_explanation(values, base_values, X_explain: pd.DataFrame) -> shap.Explanation:
    """Build the common Explanation format used by every function here."""
    base_values = np.broadcast_to(
        np.asarray(base_values, dtype=float).reshape(-1), (len(X_explain),)
    ).copy()
    return shap.Explanation(
        values=np.asarray(values, dtype=float),
        base_values=base_values,
        data=X_explain[FEATURES].to_numpy(dtype=object),
        feature_names=list(_LABELS),
    )


def _check_columns(X: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    return X[FEATURES].reset_index(drop=True)


def _transformed_column_owner(preprocessor) -> np.ndarray:
    """For every column the ColumnTransformer outputs, find the original feature.

    Example: "categories__Area_India" -> "Area", "numbers__avg_temp" -> "avg_temp".
    We use the transformer's own column lists (longest matching prefix), so a
    country name containing "_" cannot confuse the mapping.
    """
    owners = []
    for out_name in preprocessor.get_feature_names_out():
        transformer_name, _, rest = out_name.partition("__")
        input_cols = None
        for name, _trans, cols in preprocessor.transformers_:
            if name == transformer_name:
                input_cols = list(cols)
                break
        if input_cols is None:
            raise ValueError(f"Cannot map transformed column '{out_name}'.")
        matches = [c for c in input_cols if rest == c or rest.startswith(c + "_")]
        if not matches:
            raise ValueError(f"Cannot map transformed column '{out_name}'.")
        owners.append(max(matches, key=len))
    return np.array(owners, dtype=object)


# ---------------------------------------------------------------------------
# 1. Exact, fast path for tree pipelines
# ---------------------------------------------------------------------------
MAX_PATH_DEPENDENT_DEPTH = 30  # deeper trees -> use the "interventional" algorithm


def _max_tree_depth(model) -> int:
    """Deepest tree in a scikit-learn forest (0 if it cannot be inspected)."""
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return 0
    depths = [
        est.tree_.max_depth for est in np.ravel(estimators) if hasattr(est, "tree_")
    ]
    return max(depths, default=0)


def _to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def explain_tree_pipeline(
    pipeline,
    X_explain: pd.DataFrame,
    *,
    X_background: pd.DataFrame | None = None,
    n_background: int = 100,
    seed: int = 42,
    check_additivity: bool = False,
) -> shap.Explanation:
    """Exact SHAP values for a fitted tree Pipeline ("preprocessing" -> "model").

    Works for Extra Trees and Histogram Gradient Boosting from
    ``advanced_future_yield_models.make_models``.

    How it works
    ~~~~~~~~~~~~
    1. Apply the pipeline's preprocessing (one-hot encoding of Country/Crop,
       scaling of the numeric columns) to ``X_explain``.
    2. ``shap.TreeExplainer`` computes exact SHAP values for every transformed
       column by walking the trees themselves, so there is no random sampling.
    3. One-hot encoding turned "Country" into ~101 columns (Area_India,
       Area_Brazil, ...).  A single row has exactly one of them equal to 1, but
       the model may also have learned from the 0s ("NOT Brazil").  The effect of
       "the country" is therefore the SUM of the SHAP values of all its one-hot
       columns.  Because SHAP values are additive, summing keeps the total
       exactly the same: base + sum(values) still equals the prediction.

    Two TreeExplainer algorithms, picked automatically
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * "tree_path_dependent" (no background data): uses the training-sample
      counts stored in each node; base value = average training prediction.
      Used for shallow trees such as Histogram Gradient Boosting.
    * "interventional" (needs a small background sample): "removing" a feature
      means taking its value from background rows; base value = average
      prediction on the background.  Used for forests deeper than
      ``MAX_PATH_DEPENDENT_DEPTH``.  Our Extra Trees grow to depth 65-80 (no
      max_depth), and we measured that the path-dependent algorithm breaks down
      numerically at that depth (additivity errors of 1e3 to 1e9 hg/ha), while
      interventional stays exact (< 0.01 hg/ha) and is ~30x faster here.
      Background = ``X_background`` if given, otherwise a ``seed``-ed sample of
      ``n_background`` rows of ``X_explain``.

      Tip: pass ``X_background=x_train`` to compare against the training years.
      If the background is drawn from the 2010-2013 test rows, "Year" gets a
      SHAP value of 0: the trees only learned splits inside 1990-2009, so every
      year from 2010 on lands in the same branch and cannot change anything.

    As a safety net the function always checks base + sum(values) against
    ``pipeline.predict``; if path-dependent ever fails that check it is redone
    with the interventional algorithm.  The final error is stored in
    ``explanation.additivity_error`` (hg/ha).

    Why ``check_additivity=False`` by default?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    shap's built-in check raises an exception (instead of fixing anything) and
    its tolerance is tight for yields of ~100,000+ hg/ha.  We run our own,
    more useful check (above) instead.

    Speed / recommended sample sizes (measured on a laptop)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * Histogram Gradient Boosting: ~200 rows in well under a second - the whole
      test set is fine.
    * Extra Trees (interventional, 100 background rows): ~0.1 s per row, so use a
      random sample of 100-300 rows for summary plots; a single row for a
      waterfall takes a fraction of a second.

    Returns a ``shap.Explanation``: values (n, 6) in hg/ha, base_values (n,),
    data = raw feature values (object array so names are shown), feature_names =
    readable labels in ``FEATURES`` order.
    """
    if not hasattr(pipeline, "named_steps") or "model" not in pipeline.named_steps:
        raise TypeError("Expected a fitted sklearn Pipeline with a 'model' step.")
    X = _check_columns(X_explain, "X_explain")
    preprocessor = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]
    X_transformed = _to_dense(preprocessor.transform(X))
    predictions = np.asarray(pipeline.predict(X), dtype=float)

    def background_matrix():
        source = X if X_background is None else _check_columns(X_background, "X_background")
        if len(source) > n_background:
            source = source.sample(n=n_background, random_state=seed)
        return _to_dense(preprocessor.transform(source))

    def run(interventional: bool):
        try:
            if interventional:
                explainer = shap.TreeExplainer(
                    model, data=background_matrix(), feature_perturbation="interventional"
                )
            else:
                explainer = shap.TreeExplainer(model)
        except Exception as exc:  # e.g. the MLP or a TransformedTargetRegressor
            raise TypeError(
                f"{type(model).__name__} is not a tree model supported by "
                "shap.TreeExplainer; use explain_model() instead."
            ) from exc
        raw = np.asarray(
            explainer.shap_values(X_transformed, check_additivity=check_additivity),
            dtype=float,
        )
        base = float(np.ravel(explainer.expected_value)[0])
        error = float(np.max(np.abs(base + raw.sum(axis=1) - predictions)))
        return raw, base, error

    interventional = (
        X_background is not None or _max_tree_depth(model) > MAX_PATH_DEPENDENT_DEPTH
    )
    raw_values, base_value, error = run(interventional)
    tolerance = 1e-6 * max(1.0, float(np.max(np.abs(predictions))))
    if error > tolerance and not interventional:
        raw_values, base_value, error = run(True)
    if error > tolerance:
        warnings.warn(
            f"SHAP additivity error is {error:,.3f} hg/ha; treat values with care."
        )

    # Sum the transformed columns back into the 6 original features.
    owners = _transformed_column_owner(preprocessor)
    values = np.column_stack(
        [raw_values[:, owners == feature].sum(axis=1) for feature in FEATURES]
    )
    explanation = _make_explanation(values, base_value, X)
    explanation.additivity_error = error
    return explanation


# ---------------------------------------------------------------------------
# 2. Model-agnostic path (ensembles, MLP, anything with .predict(df))
# ---------------------------------------------------------------------------
class _CategoryCodec:
    """Turns Country/Crop text into integer codes and back again.

    SHAP explainers work on numeric arrays, so we temporarily replace each
    category with its position in a sorted list (e.g. "India" -> 41) and decode
    the codes back to text right before calling the real model.
    """

    def __init__(self, *frames: pd.DataFrame):
        joined = pd.concat([f[CATEGORICAL_FEATURES] for f in frames], ignore_index=True)
        self.categories = {
            col: np.array(sorted(joined[col].astype(str).unique()), dtype=object)
            for col in CATEGORICAL_FEATURES
        }
        self.lookup = {
            col: {value: code for code, value in enumerate(cats)}
            for col, cats in self.categories.items()
        }

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(X), len(FEATURES)), dtype=float)
        for j, col in enumerate(FEATURES):
            if col in CATEGORICAL_FEATURES:
                out[:, j] = X[col].astype(str).map(self.lookup[col]).to_numpy()
            else:
                out[:, j] = X[col].to_numpy(dtype=float)
        return out

    def decode(self, Z: np.ndarray) -> pd.DataFrame:
        Z = np.asarray(Z, dtype=float)
        columns = {}
        for j, col in enumerate(FEATURES):
            if col in CATEGORICAL_FEATURES:
                columns[col] = self.categories[col][np.rint(Z[:, j]).astype(int)]
            else:
                columns[col] = Z[:, j]
        frame = pd.DataFrame(columns, columns=FEATURES)
        frame["Year"] = np.rint(frame["Year"]).astype(int)
        return frame


def explain_model(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    max_evals: int | None = None,
    seed: int = 42,
    *,
    max_background: int = 50,
) -> shap.Explanation:
    """Model-agnostic SHAP values on the 6 ORIGINAL features.

    Use this for the averaging / stacking ensembles and the MLP - any object
    with ``model.predict(dataframe)``.

    How it works
    ~~~~~~~~~~~~
    * "Removing" a feature is simulated by replacing its value with values from
      a small background sample of real rows (``max_background`` rows, default
      50, drawn with ``seed``) and averaging the predictions.  So the base value
      is the average prediction over the background rows.
    * Country and Crop are encoded to integer codes so SHAP can mix rows; our
      wrapper decodes the codes back to the original text and builds the
      DataFrame the model expects.  The model never sees the codes.

    Which explainer and why
    ~~~~~~~~~~~~~~~~~~~~~~~
    * ``max_evals=None`` (default): ``shap.explainers.Exact``.  With only 6
      features there are 2**6 = 64 coalitions, so we evaluate all of them and get
      the EXACT Shapley values for the chosen background - no randomness and
      additivity holds to floating-point precision.  Cost per explained row is
      64 x background predictions (50 rows -> 3,200 predictions).
    * ``max_evals=<int>``: ``shap.PermutationExplainer`` with that budget
      (must be at least 2 * 6 + 1 = 13).  It samples feature orderings, so it is
      an approximation, but because every ordering goes from "all removed" to
      "all present" the values still add up exactly to the prediction.  Useful
      only if you ever add many more features.

    Suggested sizes: 50-100 background rows and 50-200 explained rows (each
    explained row costs ~3,200 ensemble predictions, done in one batch).

    Returns a ``shap.Explanation`` in the same format as
    ``explain_tree_pipeline``.
    """
    X_bg = _check_columns(X_background, "X_background")
    X = _check_columns(X_explain, "X_explain")
    if len(X_bg) > max_background:
        X_bg = X_bg.sample(n=max_background, random_state=seed).reset_index(drop=True)

    codec = _CategoryCodec(X_bg, X)

    def predict_from_codes(Z):
        return np.asarray(model.predict(codec.decode(Z)), dtype=float).reshape(-1)

    background_codes = codec.encode(X_bg)
    explain_codes = codec.encode(X)
    masker = shap.maskers.Independent(background_codes, max_samples=len(background_codes))

    if max_evals is None:
        explainer = shap.explainers.Exact(predict_from_codes, masker)
        result = explainer(explain_codes, silent=True)
    else:
        explainer = shap.explainers.Permutation(predict_from_codes, masker, seed=seed)
        result = explainer(explain_codes, max_evals=max_evals, silent=True)

    return _make_explanation(result.values, result.base_values, X)


# ---------------------------------------------------------------------------
# Checks and tables
# ---------------------------------------------------------------------------
def additivity_error(explanation, predictions) -> float:
    """Max |base_value + sum(SHAP values) - prediction| over all rows (hg/ha).

    Extra helper (not required by the app): proves the explanation is faithful.
    """
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    return float(np.max(np.abs(reconstructed - np.asarray(predictions, dtype=float))))


def shap_importance_table(explanation) -> pd.DataFrame:
    """Global importance: mean absolute SHAP value of each feature.

    "Mean |SHAP|" = on average, how many hg/ha this feature moves a prediction
    (in either direction).  "Share (%)" = its share of the total.
    """
    mean_abs = np.abs(np.asarray(explanation.values, dtype=float)).mean(axis=0)
    total = mean_abs.sum()
    table = pd.DataFrame(
        {
            "Feature": list(explanation.feature_names),
            f"Mean |SHAP| ({UNIT})": mean_abs,
            "Share (%)": 100 * mean_abs / total if total > 0 else 0.0,
        }
    )
    return table.sort_values(f"Mean |SHAP| ({UNIT})", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots (all return a Figure)
# ---------------------------------------------------------------------------
def plot_importance(explanation, title=None) -> Figure:
    """Horizontal bar chart of mean |SHAP| per feature (in hg/ha)."""
    table = shap_importance_table(explanation).iloc[::-1]  # largest at the top
    value_col = f"Mean |SHAP| ({UNIT})"

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(table["Feature"], table[value_col], color="#2b7bba")
    for bar, share in zip(bars, table["Share (%)"]):
        ax.text(
            bar.get_width(), bar.get_y() + bar.get_height() / 2,
            f"  {bar.get_width():,.0f}  ({share:.1f}%)",
            va="center", fontsize=9,
        )
    ax.set_xlabel(f"Mean |SHAP value| ({UNIT}) - average impact on predicted yield")
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlim(0, table[value_col].max() * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title or "Feature importance (mean |SHAP|)")
    fig.tight_layout()
    return fig


def plot_beeswarm(explanation, title=None) -> Figure:
    """Beeswarm of the NUMERIC features, each dot coloured by its feature value.

    Colour needs a numeric scale ("high" vs "low"), which Country and Crop do
    not have, so they are left out here.  Their overall importance is shown in
    ``plot_importance``; a note under the plot says so.
    """
    idx = [FEATURES.index(f) for f in NUMERIC_FEATURES]
    numeric = shap.Explanation(
        values=np.asarray(explanation.values, dtype=float)[:, idx],
        base_values=explanation.base_values,
        data=np.asarray(explanation.data)[:, idx].astype(float),
        feature_names=[explanation.feature_names[i] for i in idx],
    )
    fig = plt.figure(figsize=(8, 4.5))
    shap.plots.beeswarm(numeric, show=False, plot_size=None, color_bar=True)
    ax = fig.axes[0]
    ax.set_xlabel(f"SHAP value ({UNIT}) - impact on predicted yield")
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands_k))
    ax.set_title(title or "How numeric features push predictions up or down")
    fig.text(
        0.01, 0.01,
        "Country and Crop are categorical (no high/low colour); see the importance chart.",
        fontsize=8, color="gray",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_waterfall(explanation, index: int, title=None) -> Figure:
    """Break down one prediction: base value -> each feature's push -> prediction."""
    row = explanation[index]
    single = shap.Explanation(
        values=np.asarray(row.values, dtype=float),
        base_values=float(row.base_values),
        data=np.array([_format_feature_value(v) for v in row.data], dtype=object),
        feature_names=list(explanation.feature_names),
    )
    fig = plt.figure(figsize=(8, 4.8))
    shap.plots.waterfall(single, max_display=len(FEATURES), show=False)
    fig = plt.gcf()
    # shap writes raw numbers like "-72536.03"; add thousands separators.
    # The bar values are axis texts; f(x) / E[f(X)] are tick labels of twin axes.
    fig.canvas.draw()
    for ax in fig.axes:
        for text in ax.texts:
            text.set_text(_reformat_number_text(text.get_text()))
    for ax in fig.axes[1:]:
        labels = [_reformat_number_text(t.get_text()) for t in ax.get_xticklabels()]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(labels)
    prediction = single.base_values + single.values.sum()
    fig.suptitle(
        title or f"Why this prediction is {prediction:,.0f} {UNIT}",
        fontsize=12, y=1.0,
    )
    # No x-label: shap's E[f(X)] tick label sits in the same spot and would
    # overlap it.  The unit is already stated in the title.
    fig.axes[0].set_xlabel("")
    fig.axes[0].xaxis.set_major_formatter(FuncFormatter(_thousands))
    return fig


def plot_dependence(explanation, feature: str, title=None) -> Figure:
    """Scatter of SHAP value vs feature value for a numeric feature.

    Points are coloured by crop, because the effect of e.g. rainfall is very
    different for potatoes than for wheat.  ``feature`` may be the raw column
    name ("avg_temp") or its label ("Avg temperature (°C)").
    """
    raw = _resolve_feature(feature)
    if raw in CATEGORICAL_FEATURES:
        raise ValueError(
            f"'{feature}' is categorical; use plot_importance or the table instead."
        )
    j = FEATURES.index(raw)
    x = np.asarray(explanation.data)[:, j].astype(float)
    y = np.asarray(explanation.values, dtype=float)[:, j]
    crops = np.asarray(explanation.data)[:, FEATURES.index("Item")].astype(str)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")
    for k, crop in enumerate(sorted(set(crops))):
        mask = crops == crop
        ax.scatter(x[mask], y[mask], s=18, alpha=0.75, color=cmap(k % 10), label=crop)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(FEATURE_LABELS[raw])
    ax.set_ylabel(f"SHAP value ({UNIT})")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    if raw != "avg_temp":
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _p: f"{v:.0f}" if raw == "Year" else f"{v:,.0f}")
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Crop", fontsize=8, title_fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(title or f"Effect of {FEATURE_LABELS[raw]} on predicted yield")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plain-English explanation
# ---------------------------------------------------------------------------
def _format_feature_value(value) -> str:
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return f"{number:,.0f}" if abs(number) >= 10_000 else f"{number:.0f}"
    return f"{number:,.1f}"


def explain_single_prediction(explanation, index: int, top_n: int = 6) -> str:
    """Describe one prediction in a sentence, biggest contributions first.

    Example: "Starting from the average prediction of 74,000 hg/ha,
    Crop = Potatoes raised it by +120,000; Country = India lowered it by
    -15,000; ... giving a predicted yield of 179,000 hg/ha."
    """
    row = explanation[index]
    values = np.asarray(row.values, dtype=float)
    base = float(row.base_values)
    order = np.argsort(-np.abs(values))[:top_n]

    parts = []
    for i in order:
        verb = "raised it by" if values[i] >= 0 else "lowered it by"
        parts.append(
            f"{explanation.feature_names[i]} = {_format_feature_value(row.data[i])} "
            f"{verb} {values[i]:+,.0f}"
        )
    prediction = base + values.sum()
    return (
        f"Starting from the average prediction of {base:,.0f} {UNIT}, "
        + "; ".join(parts)
        + f" - giving a predicted yield of {prediction:,.0f} {UNIT}."
    )


__all__ = [
    "FEATURES",
    "FEATURE_LABELS",
    "explain_tree_pipeline",
    "explain_model",
    "additivity_error",
    "shap_importance_table",
    "plot_importance",
    "plot_beeswarm",
    "plot_waterfall",
    "plot_dependence",
    "explain_single_prediction",
]
