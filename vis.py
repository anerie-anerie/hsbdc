import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import r2_score

# Define the years (X values) and extend it to include 2026 for prediction
years = np.array([2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)
extended_years = np.array([2018, 2019, 2020, 2021, 2022, 2023, 2026]).reshape(-1, 1)

# Data for "Life Satisfaction" for different quintiles
life_satisfaction_data = {
    "First Quintile": [86.3, 86.8, 86.9, 86, 79.8, 78.4],
    "Second Quintile": [91.8, 91.7, 91.7, 90.6, 85.9, 83.7],
    "Third Quintile": [94.1, 94.5, 94, 93.5, 88.1, 85.5],
    "Fourth Quintile": [95.7, 95.6, 95.5, 94.5, 90, 87.7],
    "Fifth Quintile": [97, 96.4, 96.4, 95.1, 90.4, 89.6]
}

# Convert data into DataFrame
df_life_satisfaction = pd.DataFrame(life_satisfaction_data, index=years.flatten())

# Define colors for each quintile
colors = {
    "First Quintile": 'b',   # Blue
    "Second Quintile": 'g',  # Green
    "Third Quintile": 'r',   # Red
    "Fourth Quintile": 'c',  # Cyan
    "Fifth Quintile": 'm'    # Magenta
}

# Function to compute polynomial regression and plot trends
def calculate_polynomial_with_trendline(data, degree, title, ylabel):
    plt.figure(figsize=(12, 6))

    for column in data.columns:
        values = data[column].values

        # Fit a polynomial regression
        poly_fit = Polynomial.fit(years.flatten(), values, degree)
        poly_predictions = poly_fit(extended_years.flatten())

        # Calculate R² value
        r2 = r2_score(values, poly_fit(years.flatten()))

        # Plot original data and polynomial predictions
        plt.plot(years, values, label=f"{column} (actual)", marker='o', color=colors[column])
        plt.plot(extended_years, poly_predictions, linestyle="--", label=f"{column} (predicted)", color=colors[column])

        # Set R² value positioning based on the quintile
        if column in ["Second Quintile", "Fourth Quintile"]:
            # Shift the R² value to the right
            r2_x_position = extended_years[-1] - 1
        else:
            # Place R² value at the center of the trendline
            r2_x_position = extended_years[len(extended_years) // 2]

        # Add R² label at the desired position on the trendline
        plt.text(
            r2_x_position, 
            poly_predictions[-1] if column in ["Second Quintile", "Fourth Quintile"] else poly_predictions[len(extended_years) // 2], 
            f'R²: {r2:.2f}', 
            color=colors[column],
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)  # Add a background for readability
        )

        # Adjust positioning of the predicted value for 2026
        if column == "Fourth Quintile":
            # Position the predicted value higher for the Fourth Quintile
            predicted_text_y = poly_predictions[-1]  # Shift higher
        else:
            # Keep the predicted value at the trendline for the Second Quintile and others
            predicted_text_y = poly_predictions[-1]  

        # Add the predicted value for 2026 to the graph
        plt.text(
            extended_years[-1] + 0.2,  # Slight offset to the right
            predicted_text_y,           # Adjusted Y position
            f'{poly_predictions[-1]:.2f}%',  # Display predicted percentage
            color=colors[column],
            fontsize=10,
            verticalalignment='center',
            horizontalalignment='left'
        )

        # Print the predicted percentage for 2026 in the console
        print(f"Predicted life satisfaction for {column} in 2026: {poly_predictions[-1]:.2f}%")

    plt.title(f"Life Satisfaction Trends with Polynomial Regression: {title}")
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend(title="Quintiles", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Perform polynomial regression analysis for life satisfaction data
calculate_polynomial_with_trendline(df_life_satisfaction, degree=2, title="by Income Quintiles", ylabel="Satisfaction (%)")
