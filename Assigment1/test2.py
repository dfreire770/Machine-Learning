# Imports
import numpy as np 
import pandas as pd

def minkowskiDistance(A, B, p=2):
    # This function will calculate the Minkowski distance
    # The default value for p is 2
    
    return pow(pow(abs(B - A),p).sum(axis=1),1/p)

def partition( feature_matrix, target_vector, t, shuffle = True):
   
    if shuffle:
        shuffler = np.random.permutation(len(feature_matrix))
        feature_matrix = feature_matrix[shuffler]
        target_vector = target_vector[shuffler]

    train_split_limit = int(1-t * len(feature_matrix))
    test_split_limit = int(1-t * len(feature_matrix))

    x_train = feature_matrix[:train_split_limit]
    x_test = feature_matrix[test_split_limit:]
       
    y_train = target_vector[:train_split_limit]
    y_test = target_vector[test_split_limit:]

    return x_train, x_test, y_train, y_test

df = pd.read_csv('winequality-white.csv',sep=';')

X = df[['alcohol','density']]

#X = np.array(df.drop(columns=['quality']))

Y = df['quality']

Y = np.asarray([0 if val <= 5 else 1 for val in Y])

X = np.array(X)
Y = np.array(Y)

x_train, x_test, y_train, y_test = partition(X,Y,0.2,shuffle=True)