import argparse
import numpy as np
import pandas as pd
from utils import mae, averaged_mae, get_targets

def compute_features(df, window_size):
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_data = df[mask].copy()

        # Momentum = percentage price change over window_size days
        ticker_data["momentum"] = ticker_data["close"].pct_change(window_size)

        # Average price over window_size days
        ticker_data["ma"] = ticker_data["close"].rolling(window_size).mean()

        # Distance from MA (normalized)
        ticker_data["distance_from_ma"] = (
            ticker_data["close"] - ticker_data["ma"]
        ) / ticker_data["ma"]

        # Update data
        df.loc[mask, "momentum"] = ticker_data["momentum"].values
        df.loc[mask, "ma"] = ticker_data["ma"].values
        df.loc[mask, "distance_from_ma"] = ticker_data["distance_from_ma"].values
    return df

def run_baseline(df, inertion=None, window_size=5, clip_abs=1e-2):
    """
    Runs proposed baseline for given df.

    - inertion: coefficeints before intertion for each prediction day for each ticker
    - window_size: rolling window size
    """
    df = df.copy()
    
    all_tickers = df['ticker'].unique()
    
    if inertion is None:
        inertion = {
            key: np.linspace(0, 0, 20)
            for key in all_tickers
        }

    # Get features
    df_w_features = compute_features(df, window_size)

    # create_predicions
    inertion_df = pd.DataFrame(inertion).T
    inertion_df.columns = [f"p{i+1}_inertion" for i in range(20)] 
    df_w_features = df_w_features.merge(inertion_df, left_on='ticker', right_index=True)
    
    for i in range(1, 21):
        inertion_col = f"p{i}"
        df_w_features[inertion_col] = (
            df_w_features["momentum"] * 
            df_w_features[f"{inertion_col}_inertion"]
        ).clip(-clip_abs, clip_abs)

    # create predicitons, select only last day
    last_day = df_w_features.loc[df.groupby("ticker")["begin"].idxmax()]

    return last_day.set_index("ticker")[[f"p{i+1}" for i in range(20)]]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run baseline solution for the competition")
    parser.add_argument("--train_path", type=str, required=True,
                       help="Path to training data (candles)")
    parser.add_argument("--output_path", type=str, default="submission.csv",
                       help="Path to save submission file (default: submission.csv)")

    args = parser.parse_args()

    df = pd.read_csv(args.train_path)

    last_40_per_ticker = df.sort_values("begin").groupby("ticker").tail(40)
    last_20_per_ticker = df.sort_values("begin").groupby("ticker").tail(20)
    
    train_targets = get_targets(df.drop(last_20_per_ticker.index))
    val_targets = get_targets(df)

    train = df.drop(last_40_per_ticker.index)
    val = df.drop(last_20_per_ticker.index)

    for i in range(1,21):
        horizon = f"p{i}"
        print(
            f"{horizon}\t|"
            f"\tlast20: {mae(run_baseline(val), val_targets, horizon)}\t|"
            f"\tlast40_20: {mae(run_baseline(train), train_targets, horizon)}"
        )

    print('-' * 40)
    print(
        averaged_mae(val_targets, run_baseline(val)), 
        "\t|\t",
        averaged_mae(run_baseline(train), train_targets)
    )

    # final run of full data
    submission = run_baseline(df)
    submission.to_csv(args.output_path)

