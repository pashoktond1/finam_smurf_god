# xxx_SmUrF_G0D_xxx

To run everything, 

1) Put the private dataset to 'data' folder under names `news.csv` and `candles.csv`
2) Setup environment:
```
python -m venv venv 
pip install -r requirements.txt
source venv/bin/activate
```
This should work, if it doesn't - install `python==3.13`

3) Run main script:
```
export OPENROUTER_API=<API KEY> 
./run_all.sh
```

4) In `output` folder, there are three submission files:
* `submission_baseline.csv` - just zero submission
* `submission_tuned.csv` - predictions based on price history solely
* `submission_news.csv` - predictions based on news only
* `submission_final.csv` - final submission

> **NOTE:** the running time for news-based predictions can slightly vary due to variance in the response size. Our local run took 35 mins.