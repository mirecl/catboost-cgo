from catboost import CatBoostClassifier, Pool
import pathlib

path = pathlib.Path(__file__).parent.resolve()

text_features = [0]

train_data = [
    ["good product", 4.5, 120.0],
    ["excellent quality", 4.8, 95.0],
    ["love this item", 4.7, 110.0],
    ["bad quality", 1.2, 30.0],
    ["terrible product", 1.0, 20.0],
    ["worst purchase ever", 1.1, 25.0],
    ["works as expected", 3.9, 80.0],
    ["not worth the price", 2.1, 40.0],
]
train_labels = [1, 1, 1, 0, 0, 0, 1, 0]

eval_data = [
    ["amazing value", 4.6, 100.0],
    ["poor quality", 1.5, 35.0],
]

# ---------------------------------------------------------------------------
# Build Pool objects — text_features marks which column carries text.
# Text processing params live on the model so they are stored in the .cbm
# file; this makes GetTextFeaturesCount / GetTextFeatureIndices work via
# the C API.
# ---------------------------------------------------------------------------
train_pool = Pool(
    data=train_data,
    label=train_labels,
    text_features=text_features,
    feature_names=["review_text", "rating", "price"],
)

eval_pool = Pool(
    data=eval_data,
    text_features=text_features,
    feature_names=["review_text", "rating", "price"],
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
model = CatBoostClassifier(
    iterations=50,
    learning_rate=0.1,
    depth=4,
    loss_function="Logloss",
    random_seed=42,
    tokenizers=[
        {"tokenizer_id": "Space", "separator_type": "ByDelimiter", "delimiter": " "}
    ],
    dictionaries=[{"dictionary_id": "Word", "occurrence_lower_bound": "1"}],
    feature_calcers=["BoW"],
)

model.fit(train_pool, eval_set=eval_pool, silent=True)

# ---------------------------------------------------------------------------
# Predict on eval samples
# ---------------------------------------------------------------------------
preds_raw = model.predict(eval_pool, prediction_type="RawFormulaVal")
print(f"Preds `RawFormulaVal` : {preds_raw.tolist()}")

preds_class = model.predict(eval_pool, prediction_type="Class")
print(f"Preds `Class`         : {preds_class.tolist()}")

model.save_model(f"{path}/text.cbm")
print(f"Model saved to {path}/text.cbm")
