# https://catboost.ai/en/docs/concepts/python-usages-examples#regression
from catboost import CatBoostRegressor
import pathlib

path = pathlib.Path(__file__).parent.resolve()

# Initialize data
train_data = [[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60]]
eval_data = [[2, 4, 6, 8], [1, 4, 50, 60]]
train_labels = [10, 20, 30]

# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=2, learning_rate=1, depth=2)

# Fit model
model.fit(train_data, train_labels, silent=True)

# Get predictions
preds_raw = model.predict(eval_data, prediction_type="RawFormulaVal")
print(f"Preds `RawFormulaVal`: {preds_raw}")

# Save model
model.save_model(f"{path}/regressor.cbm")
