"""Second stage of a hybrid PLS-SEM and artificial neural network (ANN) analysis.

Takes latent variable scores exported from SmartPLS 4 and, for each outcome:
  1. estimates the linear structural paths (OLS on standardised latent scores,
     which reproduces the PLS-SEM inner-model estimates) to give a linear ranking;
  2. trains a multilayer perceptron with one hidden layer (sigmoid activation,
     identity output, standardised inputs) on ten random 90/10 train-test splits;
  3. reports training and testing RMSE (mean and SD across the ten runs);
  4. runs a sensitivity analysis on all cases and converts it to normalised importance
     (importance / largest importance in the run, as a %), averaged over runs;
  5. compares the ANN ranking with the linear ranking.

This mirrors Section 3.13.3 of the thesis, which plans the same procedure in the
SPSS multilayer perceptron module (Chan and Chong, 2012; Leong et al., 2020).

Usage:
    python src/ann_stage.py --data data/synthetic_latent_scores.csv --outdir results \
        --outcome OCP --inputs CEDP CEMP GLED MLED DAIC \
        --outcome ECP --inputs CEDP CEMP GLED MLED DAIC
If no --outcome/--inputs pairs are given, the thesis defaults are used.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DEFAULT_MODELS = {
    "OCP": ["CEDP", "CEMP", "GLED", "MLED", "DAIC"],
    "ECP": ["CEDP", "CEMP", "GLED", "MLED", "DAIC"],
}
LABELS = {
    "CEDP": "CE design principles", "CEMP": "CE material practices",
    "GLED": "Geometry-based LED", "MLED": "Material-based LED",
    "DAIC": "Digital and analytical capability",
    "OCP": "Operational carbon performance", "ECP": "Embodied carbon performance",
}


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def linear_paths(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Standardised OLS coefficients on latent scores (= PLS-SEM structural paths)."""
    Xs = StandardScaler().fit_transform(X)
    ys = (y - y.mean()) / y.std(ddof=0)
    coef = LinearRegression().fit(Xs, ys).coef_
    return pd.Series(coef, index=X.columns, name="path_coefficient")


def sensitivity(model, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Increase in RMSE when each input is held at its mean (zero after scaling)."""
    base = rmse(y_test, model.predict(X_test))
    out = []
    for j in range(X_test.shape[1]):
        Xp = X_test.copy()
        Xp[:, j] = 0.0
        out.append(max(rmse(y_test, model.predict(Xp)) - base, 0.0))
    return np.array(out)


def run_ann(X: pd.DataFrame, y: pd.Series, hidden: int, runs: int, seed: int):
    splitter = ShuffleSplit(n_splits=runs, test_size=0.10, random_state=seed)
    rows, imps = [], []
    for k, (tr, te) in enumerate(splitter.split(X), start=1):
        scaler = StandardScaler().fit(X.iloc[tr])
        Xtr, Xte = scaler.transform(X.iloc[tr]), scaler.transform(X.iloc[te])
        ytr, yte = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()
        net = MLPRegressor(hidden_layer_sizes=(hidden,), activation="logistic",
                           solver="lbfgs", alpha=1e-3, max_iter=5000, random_state=seed + k)
        net.fit(Xtr, ytr)
        rows.append({"run": k, "n_train": len(tr), "n_test": len(te),
                     "rmse_train": rmse(ytr, net.predict(Xtr)),
                     "rmse_test": rmse(yte, net.predict(Xte))})
        # As in SPSS, importance is computed on the combined training and testing cases
        imp = sensitivity(net, scaler.transform(X), y.to_numpy())
        imps.append(100 * imp / imp.max() if imp.max() > 0 else imp)
    rmse_df = pd.DataFrame(rows)
    imp_df = pd.DataFrame(imps, columns=X.columns)
    imp_df.insert(0, "run", range(1, runs + 1))
    return rmse_df, imp_df


def plot_importance(summary: pd.DataFrame, outcome: str, path: str) -> None:
    s = summary.sort_values("mean_normalised_importance")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh([LABELS.get(i, i) for i in s.index], s["mean_normalised_importance"],
            xerr=s["sd_normalised_importance"], color="#2f6f8f", capsize=3)
    ax.set_xlabel("Mean normalised importance (%) across 10 runs")
    ax.set_title(f"ANN predictor importance: {LABELS.get(outcome, outcome)}")
    ax.set_xlim(0, 110)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def analyse(df: pd.DataFrame, outcome: str, inputs: list, outdir: str, hidden: int, runs: int, seed: int) -> pd.DataFrame:
    X, y = df[inputs], df[outcome]
    paths = linear_paths(X, y)
    rmse_df, imp_df = run_ann(X, y, hidden, runs, seed)

    rmse_df.to_csv(os.path.join(outdir, f"rmse_{outcome}.csv"), index=False)
    imp_df.round(2).to_csv(os.path.join(outdir, f"importance_runs_{outcome}.csv"), index=False)

    summary = pd.DataFrame({
        "path_coefficient": paths,
        "linear_rank": paths.abs().rank(ascending=False).astype(int),
        "mean_normalised_importance": imp_df[inputs].mean(),
        "sd_normalised_importance": imp_df[inputs].std(),
    })
    summary["ann_rank"] = summary["mean_normalised_importance"].rank(ascending=False).astype(int)
    summary["rank_agreement"] = np.where(summary["linear_rank"] == summary["ann_rank"], "same", "differs")
    summary = summary.sort_values("ann_rank")
    summary.round(3).to_csv(os.path.join(outdir, f"ranking_{outcome}.csv"))
    plot_importance(summary, outcome, os.path.join(outdir, f"importance_{outcome}.png"))

    print(f"\n=== {outcome}: {LABELS.get(outcome, outcome)} ===")
    print(f"RMSE training: {rmse_df.rmse_train.mean():.3f} (SD {rmse_df.rmse_train.std():.3f})")
    print(f"RMSE testing:  {rmse_df.rmse_test.mean():.3f} (SD {rmse_df.rmse_test.std():.3f})")
    print(summary.round(3).to_string())
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic_latent_scores.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--outcome", action="append", help="outcome construct (repeat with --inputs)")
    ap.add_argument("--inputs", action="append", nargs="+", help="input constructs for the preceding --outcome")
    ap.add_argument("--hidden", type=int, default=4, help="neurons in the hidden layer")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    models = dict(zip(args.outcome, args.inputs)) if args.outcome else DEFAULT_MODELS
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    for outcome, inputs in models.items():
        analyse(df, outcome, inputs, args.outdir, args.hidden, args.runs, args.seed)


if __name__ == "__main__":
    main()
