import numpy as np
import pandas as pd

sales = pd.read_csv('./ML_Washington/kc_house_data.csv')
train_data = pd.read_csv('./ML_Washington/kc_house_train_data.csv')
test_data = pd.read_csv('./ML_Washington/kc_house_test_data.csv')
sales.columns

# convert the type of DataFrame to the specific type
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales.dtypes
sales = sales.astype(dtype_dict)
sales.dtypes

# 3. consider transformations of existing variables
# Add 4 new variables in both your train_data and test_data.
# ‘bedrooms_squared’ = ‘bedrooms’*‘bedrooms’
# ‘bed_bath_rooms’ = ‘bedrooms’*‘bathrooms’
# ‘log_sqft_living’ = log(‘sqft_living’)
# ‘lat_plus_long’ = ‘lat’ + ‘long’
train_data['bedrooms_squared'] = train_data['bedrooms']*train_data['bedrooms']
train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']
train_data['log_sqft_living'] = np.log(train_data['sqft_living'])
train_data['lat_plus_long'] = train_data['lat'] + train_data['long']

test_data['bedrooms_squared'] = test_data['bedrooms']*test_data['bedrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']
test_data['log_sqft_living'] = np.log(test_data['sqft_living'])
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

# 4. Quiz Question: what are the mean (arithmetic average) values of your 4 new variables on TEST data?
# (round to 2 digits)
round(np.mean(test_data['bedrooms_squared']),2)
round(np.mean(test_data['bed_bath_rooms']), 2)
round(np.mean(test_data['log_sqft_living']), 2)
round(np.mean(test_data['lat_plus_long']), 2)

# 5. estimate the regression coefficients/weights for predicting ‘price’ for the following three models:
# (In all 3 models include an intercept
# Model 1: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’, and ‘long’
# Model 2: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,‘long’, and ‘bed_bath_rooms’
# Model 3: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,‘long’, ‘bed_bath_rooms’, ‘bedrooms_squared’,
#          ‘log_sqft_living’, and ‘lat_plus_long’

# 6. Quiz Question: What is the sign (positive or negative) for the coefficient/weight for ‘bathrooms’ in Model 1?
