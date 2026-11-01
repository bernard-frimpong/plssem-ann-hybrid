"""Generate a synthetic set of latent variable scores that mimics a SmartPLS 4 export.

The data are NOT survey responses. They are simulated from the hypothesised
structural model of the MSc thesis "Developing an integrated circular economy and
net-zero carbon building framework for tropical developing countries" so that the
analysis pipeline can be shared publicly without exposing any respondent data.

Constructs (standardised latent scores, mean 0, SD 1):
    CEDP  circular economy design principles        (exogenous)
    CEMP  circular economy material practices        (exogenous)
    DAIC  digital and analytical capability          (moderator)
    GLED  geometry-based low-energy design           (mediator)
    MLED  material-based low-energy design           (mediator)
    OCP   operational carbon performance             (outcome)
    ECP   embodied carbon performance                (outcome, formative in the thesis)

Usage:
    python src/generate_synthetic_data.py --n 385 --seed 2026 --out data/synthetic_latent_scores.csv
"""
import argparse

import numpy as np
import pandas as pd


def standardise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=0)


def simulate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Correlated exogenous constructs
    cov = np.array([[1.00, 0.45, 0.30],
                    [0.45, 1.00, 0.25],
                    [0.30, 0.25, 1.00]])
    cedp, cemp, daic = rng.multivariate_normal(np.zeros(3), cov, size=n).T

    # Mediators (H3a strong, H3b weak)
    gled = standardise(0.45 * cedp + rng.normal(0, 0.89, n))
    mled = standardise(0.20 * cedp + 0.15 * daic + rng.normal(0, 0.96, n))

    # Operational carbon performance (H1b weak +, H2b -, H4a +, H4b +, H6b +)
    ocp = (0.08 * cedp - 0.15 * cemp + 0.40 * gled + 0.30 * mled
           + 0.12 * daic * gled
           + 0.10 * np.tanh(1.5 * gled)          # mild non-linearity for the ANN to find
           + rng.normal(0, 0.70, n))

    # Embodied carbon performance (H1a +, H2a strong +, H4c ~0, H4d -, H6a +)
    ecp = (0.25 * cedp + 0.45 * cemp + 0.00 * gled - 0.25 * mled
           + 0.15 * daic * mled
           + 0.15 * np.maximum(cemp, 0) ** 2 / 2  # threshold effect of material practices
           + rng.normal(0, 0.70, n))

    df = pd.DataFrame({
        "CEDP": standardise(cedp), "CEMP": standardise(cemp), "DAIC": standardise(daic),
        "GLED": gled, "MLED": mled,
        "OCP": standardise(ocp), "ECP": standardise(ecp),
    })
    df.insert(0, "respondent_id", [f"R{i:03d}" for i in range(1, n + 1)])
    return df.round(4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=385, help="number of synthetic respondents (thesis target: 385)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/synthetic_latent_scores.csv")
    args = ap.parse_args()
    df = simulate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic cases to {args.out}")


if __name__ == "__main__":
    main()
