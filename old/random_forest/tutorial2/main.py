import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

features = pd.read_csv('temps.csv')


st.write(features)

st.write(features.describe())


# one-hot encode the data

features = pd.get_dummies(features)

st.write(features.iloc[:,5:].head(5))

"""
## Labels
"""

# labels are what we want to predict
labels = np.array(features['actual'])
st.write(labels)

"""
## Features list
"""

features=features.drop('actual', axis = 1)
feature_list = list(features.columns)


features = np.array(features)
st.write(features)


"""
## Split the model
"""

train_features, test_features, train_labels, test_labels  = train_test_split(features, labels, test_size=0.25, random_state=42)

st.write('Training Features Shape:', train_features.shape)
st.write('Training Labels Shape:', train_labels.shape)
st.write('Testing Features Shape:', test_features.shape)
st.write('Testing Labels Shape:', test_labels.shape)

"""
## Establish baseline
"""

baseline_preds = test_features[:, feature_list.index('average')]
baseline_errors = abs(baseline_preds - test_labels)

st.write('Average baseline error: ' +  str(round(np.mean(baseline_errors),2 )))


"""
## Train model
"""

rf = RandomForestRegressor(n_estimators=1000, random_state = 42)
rf.fit(train_features, train_labels)


"""
## Prediction
"""

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)

st.write(f'Mean absolute error: {str(np.mean(errors))} ')


"""
## Accuracy
"""

mape = 100 * (errors / test_labels)

accuracy = 100 - np.mean(mape)
st.write(f'Accuracy: {accuracy}')


"""
## Print a tree
"""

from sklearn.tree import export_graphviz
import pydot

tree = rf.estimators_[5]

export_graphviz(tree, out_file='tree.dot', feature_names = feature_list, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

