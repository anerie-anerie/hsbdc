import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Clean and prepare the data

# Data for youth health characteristics (merged daily/normal categories)
data = {
    'Indicator': [
        'Perceived health (excellent or very good)',
        'Smoking (daily or occasionally)',
        'Cannabis use (past 12 months)',
        'E-cigarette use (past 30 days)',
        'Physical activity (60+ min/day)',
        'Headache'
    ],
    '2019': [423500, 32500, 41900, 96000, 105500, 232300],
    '2023': [210500, 100400, 496000, 256800, 187700, 381700]
}

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Visualize the trends between 2019 and 2023

def plot_trends(df):
    plt.figure(figsize=(10, 6))
    for index, row in df.iterrows():
        plt.plot(['2019', '2023'], [row['2019'], row['2023']], label=row['Indicator'], marker='o')
    
    plt.title("Youth Health Characteristics Trend (2019-2023)")
    plt.xlabel("Year")
    plt.ylabel("Number of Youth (in thousands)")
    plt.legend()
    plt.xticks(['2019', '2023'])
    plt.grid(True)
    plt.show()

# Plot the trends
plot_trends(df)

# Step 3: Prepare data for Random Forest Regression
# We need to create a "long" format DataFrame for regression
df_long = pd.melt(df, id_vars=['Indicator'], value_vars=['2019', '2023'], var_name='Year', value_name='Value')

# Convert Year to numeric (2019 -> 0, 2023 -> 1) for regression
df_long['Year'] = df_long['Year'].map({'2019': 0, '2023': 1})

# Step 4: Train Random Forest Regression model

# Prepare the features (X) and the target (y)
X = df_long[['Year']].values  # Feature: Year (2019 = 0, 2023 = 1)
y = df_long['Value'].values   # Target: Health characteristic values

# Split the data into training and testing sets (though we have only two data points, typically this would be larger)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestRegressor and fit the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Predict future trends (for 2025)

# Predict for 2025 (using Year = 2 as a placeholder for 2025)
predicted_2025 = rf_model.predict([[2]])

print(f"Predicted average value for youth health characteristics in 2025: {predicted_2025[0]:,.0f}")

# Step 6: Evaluate the model (for testing purposes)

y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Step 7: Visualize the predictions for the future years
years = [2019, 2023, 2025]
predictions = list(y) + [predicted_2025[0]]

plt.figure(figsize=(10, 6))
plt.plot(years, predictions, marker='o', label="Predicted Trend", color='orange')
plt.title("Predicted Trend for Youth Health Characteristics (2019-2025)")
plt.xlabel("Year")
plt.ylabel("Average Youth Health (in thousands)")
plt.xticks(years)
plt.grid(True)
plt.legend()
plt.show()
