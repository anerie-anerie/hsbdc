import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv('categories_with_city_and_type.csv')

# Step 2: Clean and preprocess the data
df = df.dropna(subset=['type'])

# Convert 'type' column to lowercase
df['type'] = df['type'].str.lower()

# Step 3: Create a count plot to compare facility types by urban/rural location
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='facility_category', hue='type', palette={'urban': 'blue', 'rural': 'green'})
plt.title('Comparison of Healthcare Facility Types by Urban vs Rural')
plt.xlabel('Facility Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Location', loc='upper right')
plt.tight_layout()

# Step 4: Show the plot
plt.show()
