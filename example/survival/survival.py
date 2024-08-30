# https://github.com/catboost/tutorials/blob/master/regression/survival.ipynb
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sksurv.datasets import load_whas500
import pathlib

path = pathlib.Path(__file__).parent.resolve()

# Initialize data
data, target = load_whas500()
data = data.join(pd.DataFrame(target))

data['y_lower'] = data['lenfol']
data['y_upper'] = np.where(data['fstat'], data['lenfol'], -1)

stratifying_column = data['fstat']
data = data.drop(['fstat','lenfol'],axis=1)

train, test = train_test_split(data, test_size=0.2, stratify=stratifying_column, random_state=32)

features = data.columns.difference(['y_lower', 'y_upper'], sort=False)
cat_features = ['cvd','afb','sho','chf','av3', 'miord', 'mitype', 'gender']

train_pool = Pool(train[features], label=train.loc[:,['y_lower','y_upper']], cat_features=cat_features)
test_pool = Pool(test[features], label=test.loc[:,['y_lower','y_upper']], cat_features=cat_features)

# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=500, loss_function='SurvivalAft:dist=Normal', eval_metric='SurvivalAft', verbose=0)

# Fit model
model.fit(train_pool, eval_set=test_pool)

test = [['0', 60.0, '0', 20.4684, '0', '1', 52.0, '1', 84.0, 10.0, '0', '0', '0', 169.0], 
        ['0', 61.0, '0', 25.4607, '1', '0', 80.0, '1', 111.0, 5.0, '0', '0', '0', 130.0],
        ['0', 85.0, '0', 21.94843, '0', '1', 104.0, '1', 97.0, 9.0, '1', '0', '0', 198.0]]

# Get predicted Exponent
preds = model.predict(test, prediction_type='Exponent')
print(f"Preds `Exponent`: {preds}")

# Save model
model.save_model(f"{path}/survival.cbm")





