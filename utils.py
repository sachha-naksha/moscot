import os
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import moscot as mt
import moscot.plotting as mpl
from moscot.problems.time import TemporalProblem


def driver_tfs_for_subset(
    tp,
    old_cell_cluster: str,
    young_cell_cluster: str,
    t_young: float = 1.5,
    t_old: float = 3.5,
    cell_cluster_col: str = "age",
    timepoint_col: str = "age_pop",
    features: str = "human",
    qval_thresh: float = 0.05,
    corr_thresh: float = 0.1,
) -> pd.DataFrame:
    """
    Compute driver TFs for the transition from a young_cell_cluster subpopulation
    to an old_cell_cluster subpopulation using optimal transport pull-back.

    Parameters
    ----------
    tp          : moscot TemporalProblem (already fitted)
    old_cell_cluster     : cell cluster to pull back FROM (target at t_old)
    young_cell_cluster   : cell cluster to subset at t_young (source)
    t_young     : pseudotime of the source timepoint (default 1.5)
    t_old       : pseudotime of the target timepoint (default 3.5)
    cell_cluster_col     : obs column name for bins of cells (e.g. 'Annotation', 'age')
    timepoint_col: obs column name for temporal label used for OT (e.g. 'age_pop')
    features    : TF list passed to compute_feature_correlation
    qval_thresh : q-value cutoff for significance filter
    corr_thresh : absolute correlation cutoff for significance filter

    Returns
    -------
    pd.DataFrame with columns:
        corr, pval, qval, ci_low, ci_high, significant
    sorted by correlation descending, indexed by gene name.
    """
    key = f"{old_cell_cluster}_{young_cell_cluster}_pull"

    # --- 1. Pull back old_age cells at t_old onto all cells at t_young ---
    tp.pull(
        t_young,
        t_old,
        data=cell_cluster_col,
        subset=old_cell_cluster,
        key_added=key,
        normalize=True,
    )

    pull_sum = tp.adata.obs[key].sum()
    print(f"[pull] key='{key}' | weight sum = {pull_sum:.4f}")
    if pull_sum == 0:
        raise ValueError(
            f"All pull weights are 0. Check that cell cluster {old_cell_cluster} exists at "
            f"timepoint {t_old} in obs['{cell_cluster_col}']."
        )

    # --- 2. Report weight stats for young_age cells specifically ---
    mask = (
        (tp.adata.obs[cell_cluster_col] == young_cell_cluster) &
        (tp.adata.obs[timepoint_col] == t_young)
    )
    n_cells = mask.sum()
    print(f"[subset] {n_cells} cells with {cell_cluster_col}={young_cell_cluster} at t={t_young}")
    if n_cells == 0:
        raise ValueError(
            f"No cells found with {cell_cluster_col}={young_cell_cluster} at timepoint {t_young}."
        )
    print(tp.adata.obs.loc[mask, key].describe())

    # --- 3. Compute feature correlation across all t_young cells ---
    drivers = tp.compute_feature_correlation(
        obs_key=key,
        features=features,
        annotation={timepoint_col: [t_young, t_old]},
    )

    # --- 4. Rename columns to be age-pair specific ---
    prefix = f"{old_cell_cluster}_{young_cell_cluster}"
    drivers.columns = [c.replace(key, prefix) for c in drivers.columns]
    corr_col = f"{prefix}_corr"
    qval_col = f"{prefix}_qval"

    # --- 5. Add significance flag and sort ---
    drivers["significant"] = (
        (drivers[qval_col] < qval_thresh) &
        (drivers[corr_col].abs() > corr_thresh)
    )

    drivers = drivers.dropna(subset=[corr_col]).sort_values(corr_col, ascending=False)

    n_sig = drivers["significant"].sum()
    print(f"\n[result] {n_sig} significant drivers "
          f"(qval<{qval_thresh}, |corr|>{corr_thresh})")

    top_pos = drivers.head(10).style.set_caption(
        f"TOP 10 POSITIVE DRIVERS (cell cluster {young_cell_cluster} → {old_cell_cluster})"
    ).background_gradient(subset=[corr_col], cmap="Reds")

    top_neg = drivers.tail(10).iloc[::-1].style.set_caption(
        f"TOP 10 NEGATIVE DRIVERS (cell cluster {young_cell_cluster} → {old_cell_cluster})"
    ).background_gradient(subset=[corr_col], cmap="Blues_r")

    from IPython.display import display
    display(top_pos)
    display(top_neg)

    return drivers, drivers.head(10), drivers.tail(10).iloc[::-1]