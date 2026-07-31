import numpy as np

def get_target_statistics(arrays: dict, percentiles: list[int] = [25, 50, 75, 90, 95]) -> dict:
    """
    Compute quantile statistics and comparison counts from pre-loaded arrays.

    Args:
        arrays     : dict of {name: np.ndarray}, e.g.
                     {'target_value': arr1, 'target_highest': arr2, 'target_lowest': arr3}
        percentiles: list of integer percentile values to compute (default: [25,50,75,90,95])

    Returns:
        dict with quantile stats and, if 'target_highest'/'target_lowest' are present,
        comparison counts.
    """
    stats = {}

    # ── Quantile statistics ──────────────────────────────────────────────────
    quantiles = {}
    for name, arr in arrays.items():
        arr = np.asarray(np.squeeze(arr))
        quantiles[name] = dict(zip(percentiles, np.percentile(arr, percentiles).tolist()))
    stats['quantiles'] = quantiles

    # ── highest vs lowest comparison counts (if both keys present) ───────────
    if 'target_highest' in arrays and 'target_lowest' in arrays:
        highest = np.asarray(arrays['target_highest'])
        lowest  = np.asarray(arrays['target_lowest'])

        n_highest_gt_lowest = int(np.sum(highest > lowest))
        n_lowest_gt_highest = int(np.sum(lowest  > highest))
        n_equal             = len(highest) - n_highest_gt_lowest - n_lowest_gt_highest

        stats['comparison'] = {
            'highest_gt_lowest': n_highest_gt_lowest,
            'lowest_gt_highest': n_lowest_gt_highest,
            'equal':             n_equal,
            'total':             len(highest),
        }

    return stats