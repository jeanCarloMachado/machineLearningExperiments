from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn.datasets as d
import pandas as pd
import grimoire as g
import seaborn

# df = pd.array(d.make_moons())

# df
seaborn.get_dataset_names()

breakpoint()

# log_clf = LogisticRegression()
# rnd_clf = RandomForestClassifier()
# svm_clf = SVC()

# voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
#                               voting='hard')

# voting_clf.fit(X_train, y_train)
