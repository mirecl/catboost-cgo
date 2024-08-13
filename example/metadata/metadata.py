# https://catboost.ai/en/docs/concepts/python-reference_catboost_metadata
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import pathlib
from catboost import CatBoostRegressor

path = pathlib.Path(__file__).parent.resolve()

X, y = make_regression(n_samples=1000, n_features=10, n_informative=3, random_state=0)
X = pd.DataFrame(X)
X.columns = [f"Column={i}" for i in range(X.shape[1])]

cat_value_1 = ["A", "B", "C", "D", "E"]
cat_value_2 = ["some", "random", "categorical", "feature", "values", "testing"]
X.loc[:, "CatColumn_1"] = [
    cat_value_1[np.random.randint(0, len(cat_value_1))] for _ in range(X.shape[0])
]
X.loc[:, "CatColumn_2"] = [
    cat_value_2[np.random.randint(0, len(cat_value_2))] for _ in range(X.shape[0])
]

model = CatBoostRegressor(
    cat_features=["CatColumn_1", "CatColumn_2"], one_hot_max_size=300, iterations=100
)
model.fit(X, y, silent=True)

# get proxy reference for convenience
metadata = model.get_metadata()

# set some metadata key
metadata["example_key"] = "example_value"

# iterate metadata key-values
for key, value in metadata.items():
    if key == "training":
        continue
    print(f"{key.upper()}:\n{value}\n")

print(f"Used Features names: {model.feature_names_}")

# Save model
model.save_model(f"{path}/metadata.cbm")
