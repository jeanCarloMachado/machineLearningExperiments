import streamlit as st
import pandas as pd
import grimoire as g
import grimoire.time as t

"""
# Titanic challenge

## Steps
 - Clean the data
    Encode other features to fit the model
"""


"""
## Understanding the training data


"""
train = pd.read_csv('train.csv')
with st.echo():
    st.write(train.sample(n=5))
    st.write(train.describe())

train = train.fillna(method='ffill')

"""
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

---


## Prepare the data
"""
from sklearn.ensemble import RandomForestClassifier

y = train['Survived']

with st.echo():
    def cleanup_df(df):
        ignored_cols = ['Survived', 'Cabin', 'Embarked', 'Name', 'PassengerId', 'PClass', 'SibSp', 'Ticket']
        df = df[df.columns.difference(ignored_cols)]
        df['Sex'] = df['Sex'].map(lambda x: 0 if x == 'male' else 1)
        return df.fillna(method='ffill')

X = cleanup_df(train)
X



"""
## Approaching the problem with a Naive Random forest

### Fit the data
"""
with st.echo():
    clf = RandomForestClassifier()
    clf.fit(X, y)


"""
### Test Data raw

The test data does not contain if the passenger survived or not
"""


test = pd.read_csv('test.csv')
test

passenger_ids = test['PassengerId'].copy()
test = cleanup_df(test)
"""
### Cleaned test data
"""

test

"""
### Prediction
[Make submission here](https://www.kaggle.com/c/titanic/data?select=test.csv)
"""

with st.echo():
    result_array = clf.predict(test)
    result_df = pd.DataFrame(result_array, columns=['Survived'])
    result = pd.concat([passenger_ids, result_df ], axis=1)
    result.to_csv(f'results/prediciton_results_{t.Date.now_str(datetime_format=True)}.csv', index=False)

result
