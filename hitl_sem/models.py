"""Patch classifier used by the segmentation loop, plus shared probability helpers."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def build_logistic_pipeline():
    """Return the (scaler + L2 logistic regression) pipeline and its parameter grid."""
    logreg_l2 = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        tol=1e-6,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", logreg_l2)
    ])

    param_grid = {"clf__C": np.logspace(-3, 1, 7)}

    return pipe, param_grid


def train_and_optimize(X_train, y_train, n_splits=5):
    """Grid-search C by stratified CV and return the best fitted model."""
    pipe, param_grid = build_logistic_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print(f"Best C: {grid.best_params_['clf__C']}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_


def predict_with_uncertainty(model, X, chunk=40000):
    """Return predictions, class probabilities and the Shannon entropy of those
    probabilities (high entropy = uncertain).

    Probabilities are computed in row blocks of ``chunk``. Pixel-mode inference
    builds a (H*W, 1024) float64 matrix (~3 GB for a 510x768 tile); predicting it
    in one shot allocates a second full copy and exhausts a 16 GB machine.
    Prediction is row-independent, so blocking is numerically identical.
    """
    n = X.shape[0]
    n_classes = len(model.classes_)

    probs = np.empty((n, n_classes), dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        probs[start:end] = model.predict_proba(X[start:end])

    y_pred = model.classes_[np.argmax(probs, axis=1)]
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

    return y_pred, probs, entropy


def safe_predict_proba(model, X):
    """Sigmoid probabilities for an SGDClassifier, guarded against under/overflow.

    Scores are clipped to [-15, 15] before the sigmoid; rows that still sum to
    zero fall back to a uniform distribution rather than dividing by zero.
    """
    scores = model.decision_function(X)
    scores = np.clip(scores, -15, 15)
    prob = 1 / (1 + np.exp(-scores))

    if prob.ndim == 1:
        prob = np.vstack([1 - prob, prob]).T

    row_sums = prob.sum(axis=1, keepdims=True)
    zero_mask = (row_sums == 0)
    if np.any(zero_mask):
        prob[zero_mask.flatten()] = 1.0 / prob.shape[1]
        row_sums[zero_mask] = 1.0

    return prob / row_sums
