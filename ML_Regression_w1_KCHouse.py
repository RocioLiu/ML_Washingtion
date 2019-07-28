import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from sklearn_linear_model import LinearRegression

# 1. Load the data
sales = pd.read_csv('./ML_Washington/kc_house_data.csv')
train_data = pd.read_csv('./ML_Washington/kc_house_train_data.csv')
test_data = pd.read_csv('./ML_Washington/kc_house_test_data.csv')
sales.columns

train_index = sample(range(len(sales(train))), )
# convert the type of DataFrame to the specific type
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float,
              'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}
sales.dtypes
sales = sales.astype(dtype_dict)
sales.dtypes

# split the while dataset into X and y ('price')
# X = sales.drop('price', axis=1)
# y = sales['price']


# 2. Split data into 80% training and 20% test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
random.seed(123)
train_index = random.sample(range(len(sales)), int(round(len(sales) * 0.8, 0)))
# sales_train = sales.iloc[train_index,:]
test_index = np.arange(len(sales))[~train_index]

train_idx = set(train_index)
test_index = [i for i in range(len(sales)) if i not in train_idx]

train_data = sales.iloc[train_index]
test_data = sales.iloc[test_index]


# 3. Write a generic function that accepts a column of data (e.g, an SArray) ‘input_feature’
# and another column ‘output’ and returns the Simple Linear Regression parameters ‘intercept’ and ‘slope’.
# Use the closed form solution from lecture to calculate the slope and intercept. e.g. in python:
def simple_linear_regression(input_feature, output):
    X = np.array(input_feature).reshape(-1, 1)
    y = np.array(output).reshape(-1, 1)
    n = len(X)

    slope = ((np.sum(X * y)) - (np.sum(y) * np.sum(X) / n)) / ((np.sum(X ** 2)) - (np.sum(X) ** 2) / n)
    intercept = (np.sum(y) / n) - (slope * np.sum(X) / n)

    return (intercept, slope)


# 4. Use your function to calculate the estimated slope and intercept on the training data to
# predict ‘price’ given ‘sqft_living’. e.g. in python with SFrames using:
input_feature = train_data['sqft_living']
output = train_data['price']

(squarefeet_intercept, squarefeet_slope) = simple_linear_regression(input_feature, output)


# 5. Write a function that accepts a column of data ‘input_feature’, the ‘slope’, and the ‘intercept’ you learned,
# and returns an a column of predictions ‘predicted_output’ for each entry in the input column. e.g. in python:
def get_regression_predictions(input_feature, intercept, slope):
    predicted_output = intercept + slope * input_feature

    return (predicted_output)


# 6. Quiz Question: Using your Slope and Intercept from (4), What is the predicted price for a house with 2650 sqft?
get_regression_predictions(2650, squarefeet_intercept, squarefeet_slope)
# 700074.8459475137

# 7. Write a function that accepts column of data: ‘input_feature’, and ‘output’ and the regression parameters
# ‘slope’ and ‘intercept’ and outputs the Residual Sum of Squares (RSS). e.g. in python:
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    RSS = np.sum((output - (intercept + slope * input_feature)) ** 2)
    return (RSS)


# 8. Quiz Question: According to this function and the slope and intercept from (4) What is the RSS for the
# simple linear regression using squarefeet to predict prices on TRAINING data?
get_residual_sum_of_squares(input_feature, output, squarefeet_intercept, squarefeet_slope)
# 1201918354177283.0

# 9. Note that although we estimated the regression slope and intercept in order to predict the output from
# the input, since this is a simple linear relationship with only two variables we can invert the linear
# function to estimate the input given the output!
# Write a function that accept a column of data:‘output’ and the regression parameters ‘slope’ and ‘intercept’
# and outputs the column of data: ‘estimated_input’. Do this by solving the linear function
# output = intercept + slope*input for the ‘input’ variable
def inverse_regression_predictions(output, intercept, slope):
    estimated_input = (output - intercept)/slope
    return(estimated_input)


# 10. Quiz Question: According to this function and the regression slope and intercept from (3) what is
# the estimated square-feet for a house costing $800,000?
inverse_regression_predictions(800000, squarefeet_intercept, squarefeet_slope)
# 3004.3962451522766


# 11. Instead of using ‘sqft_living’ to estimate prices we could use ‘bedrooms’ (a count of the number of
# bedrooms in the house) to estimate prices. Using your function from (3) calculate the Simple Linear Regression
# slope and intercept for estimating price based on bedrooms. Save this slope and intercept for later
# (you might want to call them e.g. bedroom_slope, bedroom_intercept).
input_feature = train_data['bedrooms']
(bedroom_intercept, bedroom_slope) = simple_linear_regression(input_feature, output)


# 12. Now that we have 2 different models compute the RSS from BOTH models on TEST data.

# 13. Quiz Question: Which model (square feet or bedrooms) has lowest RSS on TEST data? Think about
# why this might be the case.
input_feature = test_data['sqft_living']
output = test_data['price']
get_residual_sum_of_squares(input_feature, output, squarefeet_intercept, squarefeet_slope)
# 275402933617812.12

input_feature = test_data['bedrooms']
get_residual_sum_of_squares(input_feature, output, bedroom_intercept, bedroom_slope)
# 493364585960300.9

