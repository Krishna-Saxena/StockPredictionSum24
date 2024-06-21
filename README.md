# StockPredictionSum24

## Steps to Running on your Machine
1. Run all the cells in [downlod_historical_data](./download_historical_data.ipynb)
    1. Change the values of `INTEREST_STOCKS` to tickers of stocks that you are interested in
2. Run all the cells in [make_windowed_data](./make_windowed_data.ipynb)
    1. Change the values of `METRIC_COL` (suggested is 'High'), `T_PAST` (" " 120), and `T_FUT` (" " 20)
3. Run or Develop a file similar to [ML_windowed_data_analysis](./ML_windowed_data_analysis.ipynb)
    1. Change the values of `T_PAST` and `T_FUT` to your desired values (make sure you have completed step 2 for these values)
    2. Note: metrics are reported with starting values scaled to 1
    3. Note: if you are using this for buying/sellign assets, remember to rescale outputs by multiplying `S0_{tr/te}_{past/fut}`