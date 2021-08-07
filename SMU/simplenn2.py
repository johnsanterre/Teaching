# Adapted from https://iamtrask.github.io/2015/07/12/basic-python-network/
 
import numpy as np
import keras 

X = [[np.random.random()*2,
      np.random.random()*2,
      np.random.random()] for _ in range(10000)]

Y = [x[0]**x[1]+x[2]*6*np.random.normal() for x in X]


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
clf = RandomForestRegressor()
clf.fit(X[:900],Y[:900])
mean_squared_error(clf.predict(X[900:]), Y[900:])

xgboost


model = keras.Sequential()
model.add(keras.layers.Dense(units = 10, activation = 'linear', input_shape=[3]))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 1, activation = 'linear'))
model.compile(loss='mse', optimizer="adam")

model.fit(X, Y, epochs=3000)



