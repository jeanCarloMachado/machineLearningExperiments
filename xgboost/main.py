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
# breakpoint()

xg_reg = xgb.XGBRegressor(objective='reg:linear',
                          colsample_bytree=0.3,
                          learning_rate=0.1,
                          max_depth=5,
                          alpha=10,
                          n_estimators=10)

preds = xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)


st.write(f"RMSE: {ds.rmse(preds, y_test)}")
