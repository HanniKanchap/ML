## MLP 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

networkModel = Sequential()
networkModel.add(Dense(6,input_dim=2,activation='relu'))
networkModel.add(Dense(1,activation = 'sigmoid'))

networkModel.compile(loss = 'binary_crossentropy', optimizer = 'adam' ,metrics = ['accuracy'])

networkModel.fit(X,y,epochs=500, verbose=0)

loss, acc = networkModel.evaluate(X, y, verbose=0)
print(f"Accuracy: {acc*100:.2f}%")

# Predictions
preds = networkModel.predict(X)
print("Predictions:\n", preds.round())