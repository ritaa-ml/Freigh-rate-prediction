Freight Rate Prediction

**Loom walkthrough:** https://www.loom.com/share/3bf57c381f0a4242b5760fa967bbb16b
Predicts the posted rate of freight loads. Trained on Jan-Oct 2025 loads and used to predict Nov-Dec 2025 loads plus the fixed December lane chart.

Files
freight_rate_model.ipynb: cleaning, features, validation, final model, predictions
requirements.txt: dependencies
validation_predictions.csv: 12,000 predictions (load_id,predicted_rate)
candidate_december.png: chart produced by score.py
How to run
Put train_test.csv, validation.csv and december_chart_inputs.csv in a data/ folder (in Colab, the upload cell asks for them).
pip install -r requirements.txt
Run all notebook cells in order. Outputs: validation_predictions.csv and outputs/december_predictions.csv.
Validate and create the chart: python score.py --predictions validation_predictions.csv --december-predictions outputs/december_predictions.csv
Approach
Cleaning: negative weights (sign errors) made positive, missing weight filled with the equipment median, missing market_index filled with the same-date median. No rows dropped.
Features: log distance, weight, equipment, cities, market_index, quote_signal, log(quote_signal x distance), and daily averages of quote_signal / market_index (they reveal surge periods).
Validation: time-based expanding window (train on earlier months, test on the next month, May-Oct). Never a random split, because the task is to predict the future from the past.
Model: HistGradientBoostingRegressor on log(rate) with absolute-error loss.
Result: MAE 112 vs 285 for a quote_signal x distance baseline (MAPE 4.8% vs 13.9%).
December: inputs lack quote_signal/market_index, so each date uses that day's averages from the December loads in validation.csv.
