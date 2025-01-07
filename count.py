import pandas as pd

# Load the CSV file
df = pd.read_csv('categories_with_city_and_type.csv')

# Count the number of occurrences of 'Urban' and 'Rural' in the 'type' column
type_counts = df['type'].value_counts()

# Display the results
print(type_counts)
