import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define the years (X values)
years = np.array([2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)

# Household income, first quintile data (selected indicators)
data = {
   "Perceived health, very good or excellent": [71.4, 69.5, 70.8, 68.4, 62, 60.9],
    "Perceived mental health, very good or excellent": [76.3, 72.4, 71, 62.7, 59.5, 57.1],
    "Perceived life stress, most days quite a bit or extremely stressful": [22.5, 22.6, 22.5, 23.3, 25, 24]
}

# Convert data into DataFrame
df = pd.DataFrame(data, index=years.flatten())

# Function to compute regression patterns and R^2 value
def calculate_regression_with_r2(data, title, ylabel):
    plt.figure(figsize=(12, 6))

    for column in data.columns:
        values = data[column].values.reshape(-1, 1)

        # Fit linear regression model
        model = LinearRegression()
        model.fit(years, values)
        predictions = model.predict(years)

        # Calculate R^2 value
        r2 = model.score(years, values)
        
        # Plot original data and regression line
        plt.plot(years, values, label=f"{column} (actual)", marker='o')
        plt.plot(years, predictions, linestyle="--", label=f"{column} (trend) - R²: {r2:.2f}")
        
    plt.title(f"Regression Patterns: {title}")
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend(title="Indicators", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model

# Perform regression analysis and plot trends with R² values
calculate_regression_with_r2(df, "Health and Stress Indicators (Fifth Quintile)", "Percentage (%)")
