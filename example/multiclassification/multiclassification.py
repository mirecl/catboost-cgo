# https://catboost.ai/en/docs/concepts/python-usages-examples#multiclassification
from catboost import Pool, CatBoostClassifier
import pathlib

path = pathlib.Path(__file__).parent.resolve()

train_data = [
    ["summer", 1924, 44],
    ["summer", 1932, 37],
    ["winter", 1980, 37],
    ["summer", 2012, 204],
]
eval_data = [
    ["winter", 1996, 197],
    ["winter", 1968, 37],
    ["summer", 2002, 77],
    ["summer", 1948, 59],
]

cat_features = [0]

train_label = ["France", "USA", "USA", "UK"]
eval_label = ["USA", "France", "USA", "UK"]

train_dataset = Pool(data=train_data, label=train_label, cat_features=cat_features)
eval_dataset = Pool(data=eval_data, label=eval_label, cat_features=cat_features)

# Initialize CatBoostClassifier
model = CatBoostClassifier(
    iterations=10, learning_rate=1, depth=2, loss_function="MultiClass", silent=True
)

# Fit model
model.fit(train_dataset)

# Get predicted RawFormulaVal
preds_raw = model.predict(eval_dataset, prediction_type="RawFormulaVal")
print(f"Preds `RawFormulaVal`: {preds_raw}")

# Get predicted probabilities for each class
preds_proba = model.predict_proba(eval_dataset)
print(f"Preds `Probability`: {preds_proba}")

# Get predicted classes
preds_class = model.predict(eval_dataset)
print(f"Preds `Class`: {preds_class}")

# Save model
model.save_model(f"{path}/multiclassification.cbm")
