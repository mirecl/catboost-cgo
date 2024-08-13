# https://catboost.ai/en/docs/concepts/python-usages-examples#binary-classification
from catboost import CatBoostClassifier
import pathlib

path = pathlib.Path(__file__).parent.resolve()

# Initialize data
cat_features = [0, 1]
train_data = [
    ["a", "b", 1, 4, 5, 6],
    ["a", "b", 4, 5, 6, 7],
    ["c", "d", 30, 40, 50, 60],
]
train_labels = [1, 1, -1]
eval_data = [["a", "b", 2, 4, 6, 8], ["a", "d", 1, 4, 50, 60]]

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=2, learning_rate=1, depth=2)

# Fit model
model.fit(train_data, train_labels, cat_features, silent=True)

# Get predicted RawFormulaVal
preds_raw = model.predict(eval_data, prediction_type="RawFormulaVal")
print(f"Preds `RawFormulaVal`: {preds_raw}")

# Get predicted Probability
preds_proba = model.predict_proba(eval_data)
print(f"Preds `Probability`: {[preds[1] for preds in preds_proba]}")

# Save model
model.save_model(f"{path}/classifier.cbm")
