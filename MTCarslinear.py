import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# Read the CSV file into a DataFrame
filepath = "/Users/akshaya/Desktop/mtcars.csv"
data = pd.read_csv(filepath)

# Initialize lists to store results
variables = []
apes = []
highest_errors = []
lowest_errors = []

# Define a function to plot scatter plot with linear regression and calculate APE
def plot_regression(x, y):
    sns.scatterplot(x=x, y=y, data=data)
    plt.title(f'{y} vs {x}')
    plt.xlabel(x)
    plt.ylabel(y)
    
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(data[[x]], data[[y]])
    
    # Get predicted values
    predicted = reg.predict(data[[x]])
    
    # Calculate absolute percentage error (APE)
    actual = data[y]
    ape = mean_absolute_percentage_error(actual, predicted)
    
    # Append results to lists
    variables.append(f'{y} vs {x}')
    apes.append(ape)
    highest_errors.append(ape.max())
    lowest_errors.append(ape.min())
    
    # Plot the regression line
    plt.plot(data[x], predicted, color='red')
    
    plt.show()

# Scatter plot with linear regression: 'mpg' vs 'wt'
plot_regression('wt', 'mpg')

# Scatter plot with linear regression: 'hp' vs 'wt'
plot_regression('wt', 'hp')

# Scatter plot with linear regression: 'cyl' vs 'hp'
plot_regression('hp', 'cyl')

# Scatter plot with linear regression: 'disp' vs 'cyl'
plot_regression('cyl', 'disp')

# Create a DataFrame with results
results = pd.DataFrame({
    'Variable': variables,
    'APE': apes,
    'Highest Error': highest_errors,
    'Lowest Error': lowest_errors
})

# Display the results table
print(results)

# Initialize lists to store results
variables = []
predicted_values = []

# Define a function to calculate predicted values
def calculate_predicted_values(x, y):
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(data[[x]], data[[y]])
    
    # Get predicted values
    predicted = reg.predict(data[[x]])
    
    return predicted

# Calculate predicted values for each scatter plot
predicted_mpg_wt = calculate_predicted_values('wt', 'mpg')
predicted_hp_wt = calculate_predicted_values('wt', 'hp')
predicted_cyl_hp = calculate_predicted_values('hp', 'cyl')
predicted_disp_cyl = calculate_predicted_values('cyl', 'disp')

# Append results to lists
variables.extend(['mpg vs wt', 'hp vs wt', 'cyl vs hp', 'disp vs cyl'])
predicted_values.extend([predicted_mpg_wt, predicted_hp_wt, predicted_cyl_hp, predicted_disp_cyl])

# Create a DataFrame with results
results = pd.DataFrame({
    'Variable': variables,
    'Predicted Values': predicted_values
})

# Display the results table
results_df = pd.DataFrame(results)
print(results_df)

# Combine 'wt' and 'hp' into a single variable
combined_var = data['wt'] + data['hp']

# Plot combined variable against 'mpg'
plt.scatter(combined_var, data['mpg'])
plt.xlabel('Combined Variable (wt + hp)')
plt.ylabel('Miles per Gallon (mpg)')
plt.title('Miles per Gallon vs Combined Variable (wt + hp)')
reg = LinearRegression()
reg.fit(combined_var.values.reshape(-1, 1), data['mpg'])
predicted = reg.predict(combined_var.values.reshape(-1, 1))
plt.plot(combined_var, predicted, color='red', linewidth=2, label='Linear Regression')

plt.legend()
plt.show()