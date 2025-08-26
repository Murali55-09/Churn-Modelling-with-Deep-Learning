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
print(dataset.head)     #use to read some values of the dataset

# Divide the dataset in independent and dependent feature 

X = dataset.iloc[: ,3:13]     #independent feature ilpc used to trave through dataset
y = dataset.iloc[:,13]      #Dependent variable

# print(X.head())

## Feature Engineering
geography = pd.get_dummies(X['Geography'] ,drop_first=True)     #use to show only two variables
# print(geography)
gender = pd.get_dummies(X['Gender'], drop_first=True)

## Concatinate these variables with dataframes

X = X.drop(['Geograohy', 'Gender'], axis =1)