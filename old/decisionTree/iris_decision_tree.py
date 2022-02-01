from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import streamlit as st

iris = load_iris()

X = iris.data[:, 2:]  # petal length and width
y = iris.target

"""
# Features (X)
"""

st.write(X)
"""
## Used features
"""
st.write(iris.feature_names[2:])
"""
## All features
"""
st.write(iris.feature_names)

"""
## Fit it
"""
with st.echo():
    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, y)

export_graphviz(tree_clf, out_file=("./iris_tree.dot"), feature_names=iris.feature_names[2:], class_names=iris.target_names, rounded=True, filled=True)



"""
## Estimate probabilities
"""

with st.echo():
    st.write(tree_clf.predict_proba([[5, 1.5]]))

