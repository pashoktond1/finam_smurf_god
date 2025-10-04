# baseline
echo "--- baseline ---"
python baseline.py --train_path data/candles.csv --output_path output/submission_baseline.csv

# price based predicions
echo "--- price-based ---"
python tuned.py --train_path data/candles.csv --output_path output/submission_tuned.csv

# news based predictions
echo "--- news-based ---"
python news.py --candles_path data/candles.csv --news_path data/news.csv --output_path output/submission_news.csv

echo "--- final ---"
# final submission
python final.py --prices output/submission_tuned.csv --news output/submission_news.csv --output_path output/submission_final.csv
