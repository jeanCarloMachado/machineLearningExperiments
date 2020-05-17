import streamlit as st
from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
import numpy as np
import grimoire.datascience as ds

boston = load_boston()

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target

st.write(data.head())

X, y = ds.xy_split_df(data)

data_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = ds.train_test_split(X, y)

xg_reg = xgb.XGBRegressor(objective='reg:linear',
                          colsample_bytree=0.3,
                          learning_rate=0.1,
                          max_depth=5,
                          alpha=10,
                          n_estimators=10)

preds = xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)

st.write(f"RMSE: {ds.rmse(preds, y_test)}")
"""
## K fold cross-validation

(cv method)
"""

params = {
    'objective': 'reg:linear',
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10,
}

cv_results = xgb.cv(dtrain=data_dmatrix,
                 params=params,
                 nfold=3,
                 num_boost_round=50,
                 early_stopping_rounds=10,
                 metrics="rmse",
                 as_pandas=True,
                 seed=123)

st.write(cv_results.head())
st.write((cv_results['test-rmse-mean']).tail(1))
"""
## Visualize boosting trees

"""


xg_reg =xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

import matplotlib.pyplot as plt
xgb.plot_tree(xg_reg, num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
st.pyplot()


"""
## Feature importance

One simple way of doing this involves counting the number of times each feature is split on across all boosting rounds (trees) in the model, and then visualizing the result as a bar graph, with the features ordered according to how many times they appear.
"""

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5,5]
st.pyplot()

