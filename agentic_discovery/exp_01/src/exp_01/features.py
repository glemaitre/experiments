"""Domain feature engineering for the option-pricing challenge.

The instrument's pay-off is path-dependent on the *worst* of three underlyings
crossing a sequence of barriers, so summary statistics over the three-asset
basket (worst/best/median) and the basket's risk under correlation are
likely informative. Features stay in [0,1]-ish ranges where possible — GBDT
does not require this but it keeps everything interpretable.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def add_engineered(df: pl.DataFrame) -> pl.DataFrame:
    """Append derived columns to a frame containing the raw 23 features.

    The order of input columns is irrelevant; we reference them by name.
    Every new column is a pure function of the inputs (no leakage).
    """
    s = [pl.col("S1"), pl.col("S2"), pl.col("S3")]
    mu = [pl.col("mu1"), pl.col("mu2"), pl.col("mu3")]
    sig = [pl.col("sigma1"), pl.col("sigma2"), pl.col("sigma3")]

    # Basket variance for an equally-weighted basket given the 3x3 correlation
    # matrix: w'Sigma w with Sigma_ij = rho_ij * sigma_i * sigma_j.
    # With w = 1/3 each, this reduces to the expression below.
    basket_var = (
        sum(s_i * s_i for s_i in sig)
        + 2 * pl.col("rho12") * sig[0] * sig[1]
        + 2 * pl.col("rho13") * sig[0] * sig[2]
        + 2 * pl.col("rho23") * sig[1] * sig[2]
    ) / 9.0

    out = df.with_columns([
        # Order statistics over the three underlyings ("worst-of" matters most).
        pl.min_horizontal(s).alias("S_min"),
        pl.max_horizontal(s).alias("S_max"),
        pl.mean_horizontal(s).alias("S_mean"),
        (pl.max_horizontal(s) - pl.min_horizontal(s)).alias("S_range"),

        # Same for drifts and vols.
        pl.min_horizontal(mu).alias("mu_min"),
        pl.max_horizontal(mu).alias("mu_max"),
        pl.mean_horizontal(mu).alias("mu_mean"),
        pl.min_horizontal(sig).alias("sigma_min"),
        pl.max_horizontal(sig).alias("sigma_max"),
        pl.mean_horizontal(sig).alias("sigma_mean"),

        # Average pairwise correlation — proxies basket diversification.
        ((pl.col("rho12") + pl.col("rho13") + pl.col("rho23")) / 3.0).alias("rho_mean"),

        # Equal-weight basket variance / vol from the full Sigma matrix.
        basket_var.alias("basket_var"),
        basket_var.sqrt().alias("basket_vol"),
    ])

    # Time-scaled risk and return:  drift*T  and  vol*sqrt(T)  are the natural
    # log-normal moments at the deal's maturity.
    out = out.with_columns([
        (pl.col("mu_mean") * pl.col("Maturity")).alias("mu_T"),
        (pl.col("basket_vol") * pl.col("Maturity").sqrt()).alias("basket_vol_T"),
        (pl.col("sigma_max") * pl.col("Maturity").sqrt()).alias("sigma_max_T"),

        # Distances of the worst-of underlying to each barrier. Sign matters:
        # the worst-of process is the one tested against barriers.
        (pl.col("S_min") - pl.col("YetiBarrier")).alias("dist_S_min_to_yeti"),
        (pl.col("S_min") - pl.col("PhoenixBarrier")).alias("dist_S_min_to_phoenix"),
        (pl.col("S_min") - pl.col("PDIBarrier")).alias("dist_S_min_to_pdi"),
        (pl.col("S_min") - pl.col("PDIStrike")).alias("dist_S_min_to_strike"),

        # PDI moneyness scaled by gearing; PDIType near 0/1 indicates put/call.
        (
            (pl.col("PDIStrike") - pl.col("S_min"))
            * pl.col("PDIGearing")
            * (2.0 * pl.col("PDIType") - 1.0)
        ).alias("pdi_payoff_proxy"),

        # Reset frequency: more resets → more chances to recall.
        (pl.col("NbDates") / pl.col("Maturity").clip(1e-3, None)).alias("resets_per_T"),
    ])

    return out


ENGINEERED_COLS: tuple[str, ...] = (
    "S_min", "S_max", "S_mean", "S_range",
    "mu_min", "mu_max", "mu_mean",
    "sigma_min", "sigma_max", "sigma_mean",
    "rho_mean",
    "basket_var", "basket_vol",
    "mu_T", "basket_vol_T", "sigma_max_T",
    "dist_S_min_to_yeti", "dist_S_min_to_phoenix",
    "dist_S_min_to_pdi", "dist_S_min_to_strike",
    "pdi_payoff_proxy",
    "resets_per_T",
)


def feature_matrix(df: pl.DataFrame, *, raw_cols: tuple[str, ...]) -> np.ndarray:
    """Return a numpy matrix of [raw + engineered] features in fixed order."""
    enriched = add_engineered(df.select(raw_cols))
    cols = list(raw_cols) + list(ENGINEERED_COLS)
    return enriched.select(cols).to_numpy(), cols
