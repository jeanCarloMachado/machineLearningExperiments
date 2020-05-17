from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import streamlit as st
"""
# First deep learning with keras
"""

dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
"""
## Dataset
"""
st.write(dataset)

X = dataset[:, 0:8]
y = dataset[:, 8]
"""
## Define the keras model
"""
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
"""
## Compile the model
"""
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=10)

# _, accuracy = model.evaluate(X, y)
# print('Accuracy %.2f' % (accuracy * 100))
predictions = model.predict_classes(X)
for i in range(5):
    print(f"{X[i].tolist()} => {predictions[i]} (expected {y[i]})")
