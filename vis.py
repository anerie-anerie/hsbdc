import pandas as pd
import folium

# Step 1: Load the dataset
df = pd.read_csv('categories_with_city_and_type.csv')  # Ensure this file includes 'latitude', 'longitude', 'facility_category', 'type'

# Step 2: Remove rows where 'type' is NaN or missing
df = df.dropna(subset=['type'])

# Step 3: Convert 'type' column to lowercase to ensure case-insensitive matching
df['type'] = df['type'].str.lower()

# Step 4: Define the color mapping for facility categories
category_colors = {
    'long-term & chronic care': 'blue',
    'mental health & rehabilitation': 'green',
    'acute care & hospitals': 'red',
    'community & primary care': 'purple',  # Updated to purple for community & primary care
    'specialized care & support services': 'yellow',  # Updated to yellow for specialized care & support services
    'unknown': 'gray'  # Handle unknown types
}

# Step 5: Initialize the map (centered on a general location in Canada)
m = folium.Map(location=[56.1304, -106.3468], zoom_start=5)

# Step 6: Add markers for each facility
for idx, row in df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    category = row['facility_category'].lower()  # Ensure category is also lower case
    city_type = row['type']

    # Default color based on category
    color = category_colors.get(category, 'gray')  # 'gray' as a fallback for unknown categories

    # Set opacity to 0.3 for rural and 1 for urban
    opacity = 0.3 if city_type == 'rural' else 1.0

    # Create a circle marker on the map
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=opacity,
        popup=f'{category} in {row["city"]} ({city_type})'
    ).add_to(m)

# Step 7: Add a legend
legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 270px; height: 200px; 
                background-color: white; border: 2px solid black; z-index: 9999; 
                font-size: 14px; padding: 10px;">
        <b>Facility Categories</b><br>
        <i style="background-color: blue; width: 20px; height: 20px; display: inline-block;"></i> Long-Term & Chronic Care<br>
        <i style="background-color: red; width: 20px; height: 20px; display: inline-block;"></i> Mental Health & Rehabilitation<br>
        <i style="background-color: green; width: 20px; height: 20px; display: inline-block;"></i> Acute Care & Hospitals<br>
        <i style="background-color: purple; width: 20px; height: 20px; display: inline-block;"></i> Community & Primary Care<br>
        <i style="background-color: yellow; width: 20px; height: 20px; display: inline-block;"></i> Specialized Care & Support Services<br>
    </div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Step 8: Save the map as an HTML file
m.save('facility_map.html')

# Step 9: Print confirmation
print("Map saved as 'facility_map_with_updated_colors.html'")
