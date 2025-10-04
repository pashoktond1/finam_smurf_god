import numpy as np
import pandas as pd

def safe_average(cell):
    try:
        if hasattr(cell, '__iter__') and not isinstance(cell, str):
            # It's an iterable (list, tuple, etc.)
            valid_values = [x for x in cell if not pd.isna(x) and isinstance(x, (int, float))]
            return np.mean(valid_values) if valid_values else np.nan
        elif isinstance(cell, (int, float)) and not pd.isna(cell):
            # It's a single numeric value
            return cell
        else:
            # It's NaN or other non-numeric
            return np.nan
    except:
        return np.nan


def get_targets(val_df):
    # Get last 20 rows for each category
    last_20_per_ticker = val_df.sort_values("begin").groupby("ticker").tail(21)

    # Add a position column for pivoting (1-20, where 1 is oldest, 20 is newest in the last 20)
    last_20_per_ticker["position"] = last_20_per_ticker.groupby("ticker").cumcount()

    # Pivot to create columns p1 through p20
    pivoted = last_20_per_ticker.pivot_table(
        index="ticker",
        columns="position",
        values="close",
        aggfunc="first",  # Use 'first' in case of duplicates
    )
    # Rename columns to p1, p2, ..., p20
    pivoted.columns = [f"p{i}" for i in pivoted.columns]
    result = pivoted.reset_index().set_index("ticker", drop=True)

    this_day = result["p0"]

    result = result.drop("p0", axis=1)
    result = pd.DataFrame(
        result.values / this_day.values[:, None] - 1,
        index=result.index,
        columns=[f"p{i}" for i in range(1, 21)],
    )

    return result

def mae(targets, submission, horizon="p1"):
    return (targets[horizon] - submission[horizon]).abs().sum()

def averaged_mae(targets, submission):
    return sum([mae(targets, submission, horizon=f"p{i}") for i in range(1, 21)]) / 20

