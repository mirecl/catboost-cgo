from catboost.datasets import titanic
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import pathlib

path = pathlib.Path(__file__).parent.resolve()

# Initialize data
train_df, _ = titanic()

# statistics
train_df.fillna(-999, inplace=True)

# Separate features and labels
X = train_df.drop("Survived", axis=1)
y = train_df.Survived

# Get non-float type feature index
cat_fea_idx = np.nonzero(X.dtypes != np.float64)[0]

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, train_size=0.75, random_state=42
)

# Initialize CatBoostClassifier
model = CatBoostClassifier(custom_loss=["Accuracy"], random_seed=42)

# Fit model
model.fit(
    X_train,
    y_train,
    cat_features=cat_fea_idx,
    eval_set=(X_validation, y_validation),
    silent=True,
)

# Initialize data
# fmt: off
X = [
    ["892", "3", "Kelly, Mr. James", "male", 34.5, "0", "0", "330911", 7.8292, "-999", "Q"],
    ["893", "3", "Wilkes, Mrs. James (Ellen Needs)", "female", 47.0, "1", "0", "363272", 7.0, "-999", "S"],
    ["894", "2", "Myles, Mr. Thomas Francis", "male", 62.0, "0", "0", "240276", 9.6875, "-999", "Q"],
    ["895", "3", "Wirz, Mr. Albert", "male", 27.0, "0", "0", "315154", 8.6625, "-999", "S"],
    ["896", "3", "Hirvonen, Mrs. Alexander (Helga E Lindqvist)", "female", 22.0, "1", "1", "3101298", 12.2875, "-999", "S"],
]
# fmt: on

# Get predicted Class
preds_class = model.predict(X)
print(f"Preds `Class`: {preds_class}")

# Get predicted Probability
preds_proba = model.predict_proba(X)
print(f"Preds `Probability`: {[preds[1] for preds in preds_proba]}")

# Save model
model.save_model(f"{path}/titanic.cbm")
