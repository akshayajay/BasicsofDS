import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Set the parameters
quarterly_period = 66
monthly_period = 22
two_week_period = 10
weekly_period = 5
prediction_horizon = 1

# Define the start and end dates
start_date = '2022-01-01'
end_date = '2022-12-31'

# Download Nifty index data
nifty_data = yf.download('^NSEI', start=start_date, end=end_date, progress=True)
nifty_data.sort_index(ascending=True, inplace=True)

# Calculate the percentage changes for the predictors
quarterly_returns = nifty_data['Adj Close'].pct_change(quarterly_period).shift(prediction_horizon)
monthly_returns = nifty_data['Adj Close'].pct_change(monthly_period).shift(prediction_horizon)
two_week_returns = nifty_data['Adj Close'].pct_change(two_week_period).shift(prediction_horizon)

# Calculate the percentage change in weekly returns (target variable)
weekly_returns = nifty_data['Adj Close'].pct_change(weekly_period).shift(prediction_horizon)

# Create a DataFrame with the predictors and target variable
data = pd.DataFrame({
    'Quarterly Returns': quarterly_returns,
    'Monthly Returns': monthly_returns,
    'Two-Week Returns': two_week_returns,
    'Weekly Returns': weekly_returns
})
data.dropna(inplace=True)

# Split the data into training and test sets
X = data[['Quarterly Returns', 'Monthly Returns', 'Two-Week Returns']]
y = data['Weekly Returns']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
# Add constant term to X_train
X_train_with_const = sm.add_constant(X_train)

# Create an OLS model
ols_model = sm.OLS(y_train, X_train_with_const)

# Fit the OLS model
ols_results = ols_model.fit()

# Print the summary of the OLS model
print(ols_results.summary())

# Train the SVM model
svm_model = SVR()
svm_model.fit(X_train, y_train)

# Train the Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Train the Decision Tree model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Train the Polynomial Regression model
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)

# Make predictions on the test set for each model
linear_pred = linear_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
poly_pred = poly_model.predict(X_test)

# Create a DataFrame of predicted and actual values for each model
df_linear = pd.DataFrame({'Actual': y_test, 'Predicted': linear_pred})
df_svm = pd.DataFrame({'Actual': y_test, 'Predicted': svm_pred})
df_rf = pd.DataFrame({'Actual': y_test, 'Predicted': rf_pred})
df_dt = pd.DataFrame({'Actual': y_test, 'Predicted': dt_pred})
df_poly = pd.DataFrame({'Actual': y_test, 'Predicted': poly_pred})

# Sort the index in ascending order for each DataFrame
df_linear.sort_index(ascending=True, inplace=True)
df_svm.sort_index(ascending=True, inplace=True)
df_rf.sort_index(ascending=True, inplace=True)
df_dt.sort_index(ascending=True, inplace=True)
df_poly.sort_index(ascending=True, inplace=True)

# Display the DataFrames
print("Linear Regression:")
print(tabulate(df_linear, headers='keys', tablefmt='psql'))
print("SVM:")
print(tabulate(df_svm, headers='keys', tablefmt='psql'))
print("Random Forest:")
print(tabulate(df_rf, headers='keys', tablefmt='psql'))
print("Decision Tree:")
print(tabulate(df_dt, headers='keys', tablefmt='psql'))
print("Polynomial Regression:")
print(tabulate(df_poly, headers='keys', tablefmt='psql'))

# Calculate R2 value for Linear Regression
linear_r2 = r2_score(y_test, linear_pred)

# Calculate R2 value for SVM
svm_r2 = r2_score(y_test, svm_pred)

# Calculate R2 value for Random Forest
rf_r2 = r2_score(y_test, rf_pred)

# Calculate R2 value for Decision Tree
dt_r2 = r2_score(y_test, dt_pred)

# Calculate R2 value for Polynomial Regression
poly_r2 = r2_score(y_test, poly_pred)

# Display R2 values
print("Linear Regression - R2:", linear_r2)
print("SVM - R2:", svm_r2)
print("Random Forest - R2:", rf_r2)
print("Decision Tree - R2:", dt_r2)
print("Polynomial Regression - R2:", poly_r2)

# Get feature importances from Random Forest
feature_importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame of feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df.sort_values('Importance', ascending=False, inplace=True)

# Display the feature importances
print("Random Forest Variable Importance:")
print(tabulate(importance_df, headers='keys', tablefmt='psql'))

# Plot for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, linear_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.xlabel('Actual Percentage Change')
plt.ylabel('Predicted Percentage Change')
plt.title('Linear Regression: Predicted vs Actual Percentage Change')
plt.legend()
plt.grid(True)
plt.show()

# Plot for SVM
plt.figure(figsize=(10, 6))
plt.scatter(y_test, svm_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.xlabel('Actual Percentage Change')
plt.ylabel('Predicted Percentage Change')
plt.title('SVM: Predicted vs Actual Percentage Change')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.xlabel('Actual Percentage Change')
plt.ylabel('Predicted Percentage Change')
plt.title('Random Forest: Predicted vs Actual Percentage Change')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Decision Tree
plt.figure(figsize=(10, 6))
plt.scatter(y_test, dt_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.xlabel('Actual Percentage Change')
plt.ylabel('Predicted Percentage Change')
plt.title('Decision Tree: Predicted vs Actual Percentage Change')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Polynomial Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, poly_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.xlabel('Actual Percentage Change')
plt.ylabel('Predicted Percentage Change')
plt.title('Polynomial Regression: Predicted vs Actual Percentage Change')
plt.legend()
plt.grid(True)
plt.show()
