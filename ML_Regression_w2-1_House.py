import numpy as np
import pandas as pd
from sklearn_linear_model import LinearRegression

sales = pd.read_csv('./kc_house_data.csv')
train_data = pd.read_csv('./kc_house_train_data.csv')
test_data = pd.read_csv('./kc_house_test_data.csv')
sales.columns

# convert the type of DataFrame to the specific type
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales.dtypes
sales = sales.astype(dtype_dict)
train_data = sales.astype(dtype_dict)
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
round(np.mean(test_data['bedrooms_squared']),2)  # 12.45
round(np.mean(test_data['bed_bath_rooms']), 2)  # 7.5
round(np.mean(test_data['log_sqft_living']), 2)  # 7.55
round(np.mean(test_data['lat_plus_long']), 2)  # -74.65

# 5. estimate the regression coefficients/weights for predicting ‘price’ for the following three models:
# (In all 3 models include an intercept
# Model 1: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’, and ‘long’
# Model 2: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,‘long’, and ‘bed_bath_rooms’
# Model 3: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,‘long’, ‘bed_bath_rooms’, ‘bedrooms_squared’,
#          ‘log_sqft_living’, and ‘lat_plus_long’

# 6. Quiz Question: What is the sign (positive or negative) for the coefficient/weight for ‘bathrooms’ in Model 1?
X_train1 = np.array(train_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']])
y_train = np.array(train_data['price'])
lin_reg1 = LinearRegression()
lin_model1 = lin_reg1.fit(X_train1, y_train)
lin_model1.coef_
# array([ 3.07689645e+02, -5.36173619e+04,  1.66560499e+04,  6.57759890e+05, -3.12438238e+05])

# 7. Quiz Question: What is the sign (positive or negative) for the coefficient/weight for ‘bathrooms’ in Model 2
X_train2 = np.array(train_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long','bed_bath_rooms']])
y_train = np.array(train_data['price'])
lin_reg2 = LinearRegression()
lin_model2 = lin_reg2.fit(X_train2, y_train)
lin_model2.coef_
# array([ 3.01742311e+02, -1.06944022e+05, -7.12197299e+04,  6.54185065e+05,
#        -2.97173612e+05,  2.57539799e+04])

X_train3 = np.array(train_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long','bed_bath_rooms','bedrooms_squared',
                             'log_sqft_living', 'lat_plus_long']])
y_train = np.array(train_data['price'])
lin_reg3 = LinearRegression()
lin_model3 = lin_reg3.fit(X_train3, y_train)
lin_model3.coef_
# array([ 5.22172105e+02, -5.91053093e+03,  9.03707948e+04,  5.29927263e+05,
#        -4.04640292e+05, -1.48849175e+04,  7.61373841e+02, -5.43835165e+05,
#         1.25286971e+05])

# 8. Is the sign for the coefficient the same in both models? Think about why this might be the case.
# No.

# 9. Now using your three estimated models compute the RSS (Residual Sum of Squares) on the Training data.
X_test1 = np.array(test_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']])
X_test2 = np.array(test_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long','bed_bath_rooms']])
X_test3 = np.array(test_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long','bed_bath_rooms','bedrooms_squared',
                             'log_sqft_living', 'lat_plus_long']])
y_test = np.array(test_data['price'])
y_pred1 = lin_model1.predict(X_test1)
y_pred2 = lin_model2.predict(X_test2)
y_pred3 = lin_model3.predict(X_test3)

RSS1 = np.sum((y_test - y_pred1)**2)
RSS2 = np.sum((y_test - y_pred2)**2)
RSS3 = np.sum((y_test - y_pred3)**2)

# 10. Quiz Question: Which model (1, 2 or 3) had the lowest RSS on TRAINING data?
