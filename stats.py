import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# Step 1: Load the dataset
df = pd.read_csv("categories_with_city_and_type.csv")

# Step 2: Function to calculate the proportions and perform the Z-test
def compare_facility_category_proportions(df):
    # Group by facility_category and type (urban or rural)
    facility_counts = df.groupby(['facility_category', 'type']).size().unstack(fill_value=0)
    
    # Total number of facilities in urban and rural
    total_urban = df[df['type'] == 'Urban'].shape[0]
    total_rural = df[df['type'] == 'Rural'].shape[0]

    results = []

    for category in facility_counts.index:
        urban_count = facility_counts.loc[category, 'Urban']
        rural_count = facility_counts.loc[category, 'Rural']
        
        # Perform Z-test for the two proportions (urban vs rural)
        count = [urban_count, rural_count]
        nobs = [total_urban, total_rural]
        
        z_stat, p_value = proportions_ztest(count, nobs)
        
        # Calculate the proportions
        urban_proportion = urban_count / total_urban
        rural_proportion = rural_count / total_rural
        
        # Store results
        results.append({
            'Facility Category': category,
            'Urban Proportion': urban_proportion,
            'Rural Proportion': rural_proportion,
            'Z-statistic': z_stat,
            'P-value': p_value,
            'Most Disparate': 'Urban' if urban_proportion > rural_proportion else 'Rural'
        })
    
    return pd.DataFrame(results)

# Step 3: Run the comparison function
result_df = compare_facility_category_proportions(df)

# Step 4: Print the results
print(result_df)

# Step 5: Identify the category with the most disparity (most disproportionate)
most_disparate_category = result_df.loc[result_df['Z-statistic'].idxmax()]
print(f"Facility category with the most disparity: {most_disparate_category['Facility Category']}")
