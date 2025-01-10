import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare the data
data = {
    "Category": [
        "Perceived health, very good or excellent", "Perceived health, fair or poor",
        "Perceived mental health, very good or excellent", "Perceived mental health, fair or poor",
        "Asthma", "Diabetes", "Epilepsy", "Anxiety disorder (5 to 17 years old)",
        "Mood disorder (5 to 17 years old)", "Eating disorder (5 to 17 years old)",
        "Learning disability or learning disorder (5 to 17 years old)", 
        "Headache (5 to 17 years old)", "Stomach ache (5 to 17 years old)", 
        "Difficulties in getting to sleep (5 to 17 years old)", "Backache (5 to 17 years old)"
    ],
    "Males": [
        88.5, 1.75, 82.9, 4.1, 8.3, 0.27, 0.43, 4.4, 1.57, 0.19, 10.6, 11.2, 8.9, 24, 8.5
    ],
    "Females": [
        89.1, 1.71, 83.5, 4.1, 5.6, 0.24, 0.47, 5.7, 2.6, 0.45, 6.1, 19.9, 16.3, 31, 13.8
    ]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Separate perceived health values (very good or excellent and fair or poor)
perceived_health = df[df["Category"].str.contains("Perceived health")]
diseases = df[~df["Category"].str.contains("Perceived health")]

# Calculate average perceived health (very good or excellent) for each gender
avg_perceived_health_males = perceived_health.loc[perceived_health["Category"] == "Perceived health, very good or excellent", "Males"].values[0]
avg_perceived_health_females = perceived_health.loc[perceived_health["Category"] == "Perceived health, very good or excellent", "Females"].values[0]

# Calculate disease prevalence comparison by gender
avg_disease_males = diseases["Males"].mean()
avg_disease_females = diseases["Females"].mean()

# Difference in perceived health between genders
diff_perceived_health = avg_perceived_health_males - avg_perceived_health_females

# Diseases with the largest gender difference in prevalence
diseases["Gender Difference"] = diseases["Males"] - diseases["Females"]
diseases_sorted = diseases.sort_values(by="Gender Difference", ascending=False)

# Print the results
print(f"Average Perceived Health (Very Good or Excellent):")
print(f"Males: {avg_perceived_health_males}%")
print(f"Females: {avg_perceived_health_females}%")
print(f"Difference in perceived health (Males - Females): {diff_perceived_health}%\n")

print(f"Average Disease Prevalence:")
print(f"Males: {avg_disease_males}%")
print(f"Females: {avg_disease_females}%\n")

print("Diseases with the Largest Gender Difference in Prevalence:")
print(diseases_sorted[["Category", "Gender Difference"]].head())

# Visualizations

# 1. Perceived Health Comparison Bar Plot
plt.figure(figsize=(8, 6))
sns.barplot(x=["Males", "Females"], y=[avg_perceived_health_males, avg_perceived_health_females], palette="Set2")
plt.title("Average Perceived Health (Very Good or Excellent) Comparison")
plt.ylabel("Percentage (%)")
plt.show()

# 2. Disease Prevalence Comparison Bar Plot
plt.figure(figsize=(8, 6))
sns.barplot(x=["Males", "Females"], y=[avg_disease_males, avg_disease_females], palette="Set1")
plt.title("Average Disease Prevalence Comparison")
plt.ylabel("Percentage (%)")
plt.show()

# 3. Disease Gender Difference Visualization
plt.figure(figsize=(10, 8))
sns.barplot(x="Gender Difference", y="Category", data=diseases_sorted, palette="coolwarm")
plt.title("Gender Difference in Disease Prevalence")
plt.xlabel("Difference (Males - Females) Percentage")
plt.show()
