import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# Read the CSV file into a DataFrame
filepath = "/Users/akshaya/Desktop/mtcars.csv"
data = pd.read_csv(filepath)

# Initialize lists to store results
variables = []
predicted_values = []
ape_values = []

# Define a function to calculate predicted values and APE
def calculate_predicted_values(x, y):
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(data[x], data[y])
    
    # Get predicted values
    predicted = reg.predict(data[[x]])
    
    # Calculate absolute percentage error (APE)
    actual = data[y]
    ape = mean_absolute_percentage_error(actual, predicted)
    
    return predicted, ape

# Calculate predicted values and APE for each scatter plot
predicted_mpg_wt, ape_mpg_wt = calculate_predicted_values('wt', 'mpg')
predicted_mpg_wt, ape_mpg_wt = calculate_predicted_values(['wt','hp'], 'mpg')

predicted_hp_wt, ape_hp_wt = calculate_predicted_values('wt', 'hp')
predicted_cyl_hp, ape_cyl_hp = calculate_predicted_values('hp', 'cyl')
predicted_disp_cyl, ape_disp_cyl = calculate_predicted_values('cyl', 'disp')

# Append results to lists
variables.extend(['mpg vs wt', 'hp vs wt', 'cyl vs hp', 'disp vs cyl'])
predicted_values.extend([predicted_mpg_wt, predicted_hp_wt, predicted_cyl_hp, predicted_disp_cyl])
ape_values.extend([ape_mpg_wt, ape_hp_wt, ape_cyl_hp, ape_disp_cyl])

# Create a DataFrame with results
results = pd.DataFrame({
    'Variable': variables,
    'Predicted Values': predicted_values,
    'APE': ape_values
})


# Calculate average APE
average_ape = results['APE'].mean()

# Display the results table and average APE
print(results)
print("\nMAPE:", average_ape.round(3))
