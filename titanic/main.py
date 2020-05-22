import streamlit as st
import pandas as pd
import grimoire as g

"""
# Understanding the training data

Website [with description](https://www.kaggle.com/c/titanic/data?select=test.csv)

- Not everybody has an age defined


Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton


 - What is parch? Number of parents

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

"""
train = pd.read_csv('train.csv')

st.write(train.sample(n=5))
st.write(train.describe())

train = train.fillna(method='ffill')

"""
---

## Approaching the problem with a Naive Random forest

### prepare the data
"""
from sklearn.ensemble import RandomForestClassifier

y = train['Survived']
ignored_cols = ['Survived', 'Cabin', 'Embarked', 'Name', 'PassengerId', 'PClass', 'Sex', 'SibSp', 'Ticket']
X = train[train.columns.difference(ignored_cols)]

X



"""
### Fit the data
"""
with st.echo():
    clf = RandomForestClassifier()
    clf.fit(X, y)


"""
### Test Data raw
"""


test = pd.read_csv('test.csv')
test

passenger_ids =test['PassengerId'].copy()
test = test[train.columns.difference(ignored_cols)]
test = test.fillna(method='ffill')

test

"""
### Prediction
"""
result_array = clf.predict(test)
result_df = pd.DataFrame(result_array, columns=['Survived'])

result = pd.concat([passenger_ids, result_df ], axis=1)

result


result.to_csv(f'results/prediciton_results_{g.now_str()}.csv', index=False)
