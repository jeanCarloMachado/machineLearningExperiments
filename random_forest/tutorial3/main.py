from grimoire import *
import grimoire as g
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
## Dataset
"""
dataset = pd.read_csv('petrol_consumption.csv')
head = dataset.head()

head



"""
## Divide into attributes and labels
"""

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


y

"""
## Train test split
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


X_train


"""
## Scaling the data (although is not the most important in random forest)
"""


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


"""
## Train the algorithm
"""

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

y_pred


st.write(g.ds.report_prediction_error(y_pred, y_test))

