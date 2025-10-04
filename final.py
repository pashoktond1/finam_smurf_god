import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run baseline solution for the competition")
    parser.add_argument("--prices", type=str, required=True,
                       help="Path to training data (candles)")
    parser.add_argument("--news", type=str, required=True,
                       help="Path to training data (candles)")
    parser.add_argument("--output_path", type=str, default="submission.csv",
                       help="Path to save submission file (default: submission.csv)")
    args = parser.parse_args()

    prices = pd.read_csv(args.prices, index_col='ticker')
    news = pd.read_csv(args.news, index_col='ticker')
    final = prices + news
    print(final[['p1', 'p20']])
    final.to_csv(args.output_path)