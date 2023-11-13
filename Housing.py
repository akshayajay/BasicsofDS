import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.tree import plot_tree

# Read the data
data = pd.read_csv('//Users/akshaya/Desktop/Internship/housing.csv')

# Prepare the features and target variables
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']]
y = data['price']

# Perform one-hot encoding on categorical columns
X_encoded = pd.get_dummies(X)

# Scale the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the data into training and test sets
split = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split], X_scaled[split:]
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

# Plot the predicted values
plt.plot(y_test.index, pred_linear, color='r', label='Linear Regression')
plt.plot(y_test.index, pred_poly, color='g', label='Polynomial Regression')
plt.plot(y_test.index, pred_dt, color='m', label='Decision Tree Regression')
plt.plot(y_test.index, pred_svm, color='y', label='SVM Regression')
plt.plot(y_test.index, pred_rf, color='c', label='Random Forest Regression')
plt.ylabel('price')
plt.legend()
plt.show()

# Decision tree regression
reg_dt = DecisionTreeRegressor(max_depth=3)
reg_dt.fit(X_train, y_train)
pred_dt = reg_dt.predict(X_test)

# Plot the decision tree
plt.figure(figsize=(100, 60))
plot_tree(reg_dt, feature_names=X_encoded.columns, filled=True)
plt.show()

# Print the MAPE values
print('Mean Absolute Percentage Error:')
print('Linear Regression:', mape_linear)
print('Polynomial Regression:', mape_poly)
print('Decision Tree Regression:', mape_dt)
print('SVM Regression:', mape_svm)
print('Random Forest Regression:', mape_rf)