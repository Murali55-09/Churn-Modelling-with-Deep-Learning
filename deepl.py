import tensorflow as tf
# print(tf.__version__)

##import some basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# print(np.__version__)
# # print(plt.__version__)
# print(pd.__version__)

#dataset reading
dataset = pd.read_csv('Churn_Modelling.csv')
#print(dataset.head)     #use to read some values of the dataset

# Divide the dataset in independent and dependent feature 

X = dataset.iloc[: ,3:13]     #independent feature ilpc used to trave through dataset
y = dataset.iloc[:,13]        #Dependent variable

# print(X.head())

## Feature Engineering
geography = pd.get_dummies(X['Geography'] ,drop_first=True)     #use to show only two variables
# print(geography)
gender = pd.get_dummies(X['Gender'], drop_first=True)       #hard code, drop_first = true, show two columns

## Concatinate these variables with dataframes

X = X.drop(['Geography', 'Gender'], axis =1)        #droping it from the dataset, axis = 1 means drop the columns not rows

pd.concat([X, geography, gender], axis = 1)

#Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_test)

## Creating ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU, ELU, ReLU 
from tensorflow.keras.layers import Dropout

### Lets initialiaze the ANN 
classifier = Sequential()

## Adding the input layer
classifier.add(Dense(units = 11, activation = 'relu'))

## Adding the First Hidden Layer
classifier.add(Dense(units = 7, activation ='relu'))

## Adding the Second Hidden Layer
classifier.add(Dense(units = 6, activation ='relu'))

## Adding the output layer
classifier.add(Dense(1, activation = 'sigmoid'))

classifier.compile(optimizer = 'Adam', loss='binary_crossentropy', metrics=['accuracy'])

import tensorflow
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)

## earliy stopping
import tensorflow  as tf
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

## model training
model_history = classifier.fit(X_train, y_train, validation_split = 0.33, batch_size = 10, epochs = 1000, callbacks= early_stopping)

print(model_history.history.keys())

## Sumary history for Accuarcy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

## Prediction on dataset
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)

## Make the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

## Calculate the Accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)