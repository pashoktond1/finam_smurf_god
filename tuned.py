import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import optuna
from functools import partial
from baseline import run_baseline
from utils import mae, get_targets, averaged_mae  
from optuna.samplers import TPESampler

def objective(trial, df, targets):

    tickers = df["ticker"].unique()
    clip_abs = trial.suggest_float("clip_abs", 0, 0.1)
    window_size = trial.suggest_int("window_size", 1, 14)

    start_interions = {
        ticker: trial.suggest_float(f"{ticker}_start_interion", 0, 0.1)
        for ticker in tickers
    }
    end_interion = {
        ticker: trial.suggest_float(f"{ticker}_end_interion", 0, 0.1)
        for ticker in tickers
    }

    submission = run_baseline(
        df,
        clip_abs=clip_abs,
        inertion={
            ticker: np.linspace(start_interions[ticker], end_interion[ticker], 20)
            for ticker in tickers
        },
        window_size = window_size
    )

    return averaged_mae(targets, submission)

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

    sampler = TPESampler(seed=1337)
    study = optuna.create_study()
    base_args = dict(
        clip_abs=0.2,
        window_size = 5
    )
    tickers = df["ticker"].unique()
    base_args.update(
        {
            ticker: np.linspace(0, 0, 20)
            for ticker in tickers
        }
    )
    study.enqueue_trial(base_args) 
    study.optimize(
        partial(objective, df=train, targets=train_targets), 
        n_trials=100
    )

    val_res = run_baseline(
        val,
        clip_abs=study.best_params["clip_abs"],
        inertion={
            ticker: np.linspace(
                study.best_params[f"{ticker}_start_interion"],
                study.best_params[f"{ticker}_end_interion"],
                20,
            )
            for ticker in tickers
        },
        window_size = study.best_params['window_size']
    )
    train_res = run_baseline(
        train,
        clip_abs=study.best_params["clip_abs"],
        inertion={
            ticker: np.linspace(
                study.best_params[f"{ticker}_start_interion"],
                study.best_params[f"{ticker}_end_interion"],
                20,
            )
            for ticker in tickers
        },
        window_size = study.best_params['window_size']
    )

    for i in range(1,21):
        horizon = f"p{i}"
        print(
            f"{horizon}\t|"
            f"\tlast20: {mae(val_res, val_targets, horizon)}\t|"
            f"\tlast40_20: {mae(train_res, train_targets, horizon)}"
        )

    print('-' * 40)
    print(f"{study.best_value=}")
    print(study.best_params)
    print('-' * 40)
    print(
        averaged_mae(val_targets, val_res), 
        "\t|\t",
        averaged_mae(train_res, train_targets)
    )

    submission = run_baseline(
        df,
        clip_abs=study.best_params["clip_abs"],
        inertion={
            ticker: np.linspace(
                study.best_params[f"{ticker}_start_interion"],
                study.best_params[f"{ticker}_end_interion"],
                20,
            )
            for ticker in tickers
        },
        window_size = study.best_params['window_size']
    )
    submission.to_csv(args.output_path)

