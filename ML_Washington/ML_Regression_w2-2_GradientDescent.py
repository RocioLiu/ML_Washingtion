import numpy as np
import pandas as pd

sales = pd.read_csv('kc_house_data.csv')
train_data = pd.read_csv('./kc_house_train_data.csv')
test_data = pd.read_csv('./kc_house_test_data.csv')
sales.columns

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int,
              'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float,
              'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = sales.astype(dtype_dict)
train_data = train_data.astype(dtype_dict)
test_data = test_data.astype(dtype_dict)

# 3. Next write a function that takes a data set, a list of features (e.g. [‘sqft_living’, ‘bedrooms’]),
# to be used as inputs, and a name of the output (e.g. ‘price’). This function should return a features_matrix
# (2D array) consisting of first a column of ones followed by columns containing the values of the input features
# in the data set in the same order as the input list. It should also return an output_array which is an array of
# the values of the output in the data set (e.g. ‘price’).
def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the ‘features’ list into the SFrame ‘features_sframe’

    features_matrix = np.array(data_sframe[features])
    output_array = np.array(data_sframe['price'])
    return(features_matrix, output_array)

# 4. If the features matrix (including a column of 1s for the constant) is stored as a 2D array (or matrix) and
# the regression weights are stored as a 1D array then the predicted output is just the dot product between the
# features matrix and the weights (with the weights on the right). Write a function ‘predict_output’ which accepts
# a 2D array ‘feature_matrix’ and a 1D array ‘weights’ and returns a 1D array ‘predictions’. e.g. in python:
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions

# 5. If we have a the values of a single input feature in an array ‘feature’ and the prediction ‘errors’
# (predictions - output) then the derivative of the regression cost function with respect to the weight of
# ‘feature’ is just twice the dot product between ‘feature’ and ‘errors’. Write a function that accepts a
# ‘feature’ array and ‘error’ array and returns the ‘derivative’ (a single number). e.g. in python:
def feature_derivative(errors, feature):
    derivative = 2 * np.dot(feature, errors)
    return(derivative)

#


import numpy as np
