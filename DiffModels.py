import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Read the mtcars dataset
data = pd.read_csv('mtcars.csv')

# Prepare the features and target variables
X = data[['wt', 'hp', 'am']]
y = data['mpg']

# Scale the features
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Concatenate 'wt', 'hp', and 'am' columns
scaled_X = np.concatenate([scaled_X[:, 0:1], scaled_X[:, 1:2], scaled_X[:, 2:3]], axis=1)
scaled_X = scaled_X[:, 0] + scaled_X[:, 1] + scaled_X[:, 2]

# Reshape the feature array
scaled_X = scaled_X.reshape(-1, 1)

# Split the data into training and test sets
split = int(0.8 * len(scaled_X))
X_train, X_test = scaled_X[:split], scaled_X[split:]
y_train, y_test = y[:split], y[split:]

# Linear regression
reg_linear = LinearRegression()
reg_linear.fit(X_train, y_train)
pred_linear = reg_linear.predict(X_test)
mape_linear = mean_absolute_percentage_error(y_test, pred_linear)

# Polynomial regression
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X_train)
reg_poly = LinearRegression()
reg_poly.fit(X_poly, y_train)
X_test_poly = poly_features.transform(X_test)
pred_poly = reg_poly.predict(X_test_poly)
mape_poly = mean_absolute_percentage_error(y_test, pred_poly)

# Decision tree regression
reg_dt = DecisionTreeRegressor()
reg_dt.fit(X_train, y_train)
pred_dt = reg_dt.predict(X_test)
mape_dt = mean_absolute_percentage_error(y_test, pred_dt)

# SVM regression
reg_svm = SVR()
reg_svm.fit(X_train, y_train)
pred_svm = reg_svm.predict(X_test)
mape_svm = mean_absolute_percentage_error(y_test, pred_svm)

# Random Forest regression
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
pred_rf = reg_rf.predict(X_test)
mape_rf = mean_absolute_percentage_error(y_test, pred_rf)

# Plotting the data and regression lines
plt.scatter(scaled_X, y, color='b', label='Actual')
plt.plot(X_test, pred_linear, color='r', label='Linear Regression')
plt.plot(X_test, pred_poly, color='g', label='Polynomial Regression')
plt.plot(X_test, pred_dt, color='m', label='Decision Tree Regression')
plt.plot(X_test, pred_svm, color='y', label='SVM Regression')
plt.plot(X_test, pred_rf, color='c', label='Random Forest Regression')
plt.xlabel('wt+hp+am')
plt.ylabel('mpg')
plt.legend()
plt.show()

# Print the MAPE values
print('Mean Absolute Percentage Error (MAPE):')
print('Linear Regression:', mape_linear)
print('Polynomial Regression:', mape_poly)
print('Decision Tree Regression:', mape_dt)
print('SVM Regression:', mape_svm)
print('Random Forest Regression:', mape_rf)
