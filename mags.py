import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr

# Data for the 5 income quintiles
income_quintiles = np.array([1, 2, 3, 4, 5])

# Percent of people eating fruits and vegetables daily
fruit_veg_percent = np.array([20.4, 21.2, 21.1, 22.3, 22.8])

# Percent with high blood pressure
blood_pressure_percent = np.array([23.2, 22.6, 18.6, 18.1, 16.9])

# Percent with a regular healthcare provider
healthcare_percent = np.array([79.9, 82.9, 83.2, 84.0, 84.1])

# Function to compute R² value
def compute_r_squared(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2

# Calculate R² values for each dataset
r2_fruit_veg = compute_r_squared(income_quintiles, fruit_veg_percent)
r2_blood_pressure = compute_r_squared(income_quintiles, blood_pressure_percent)
r2_healthcare = compute_r_squared(income_quintiles, healthcare_percent)

# Calculate Pearson correlation coefficient for high blood pressure and healthcare provider
pearson_corr, _ = pearsonr(blood_pressure_percent, healthcare_percent)

# Plotting
plt.figure(figsize=(8, 5))

# Plotting each dataset with labels for R² values
plt.plot(income_quintiles, fruit_veg_percent, 'o', color='green', label=f"Healthy Eating (Fruits & Veg) (R² = {r2_fruit_veg:.3f}, Slope = {np.polyfit(income_quintiles, fruit_veg_percent, 1)[0]:.3f})")
plt.plot(income_quintiles, blood_pressure_percent, 'o', color='red', label=f"High Blood Pressure (R² = {r2_blood_pressure:.3f}, Slope = {np.polyfit(income_quintiles, blood_pressure_percent, 1)[0]:.3f})")
plt.plot(income_quintiles, healthcare_percent, 'o', color='blue', label=f"With Healthcare Provider (R² = {r2_healthcare:.3f}, Slope = {np.polyfit(income_quintiles, healthcare_percent, 1)[0]:.3f})")

# Line of best fit for Healthy Eating (Fruits & Veg)
slope_veg, intercept_veg = np.polyfit(income_quintiles, fruit_veg_percent, 1)
plt.plot(income_quintiles, slope_veg * income_quintiles + intercept_veg, '--', color='green')

# Line of best fit for High Blood Pressure
slope_bp, intercept_bp = np.polyfit(income_quintiles, blood_pressure_percent, 1)
plt.plot(income_quintiles, slope_bp * income_quintiles + intercept_bp, '--', color='red')

# Line of best fit for Healthcare Provider
slope_healthcare, intercept_healthcare = np.polyfit(income_quintiles, healthcare_percent, 1)
plt.plot(income_quintiles, slope_healthcare * income_quintiles + intercept_healthcare, '--', color='blue')

# Adding labels and title
plt.xlabel("Income Quintile")
plt.ylabel("Percentage")
plt.title("Health Metrics by Income Quintile")

# Adding legend and grid
plt.legend()
plt.grid(True)

# Show the plot
plt.xticks(income_quintiles)
plt.show()

# Output Pearson correlation coefficient
print(f"Pearson correlation coefficient between High Blood Pressure and Healthcare Provider: {pearson_corr:.3f}")
