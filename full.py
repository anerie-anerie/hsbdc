import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Years
years = [2018, 2019, 2020, 2021, 2022, 2023]

# High Blood Pressure and Healthcare Access Data
high_blood_pressure_data = {
    'First Quintile': [19.6, 20.6, 20.7, 22.5, 24.6, 23.2],
    'Second Quintile': [20.8, 21.2, 19.6, 21.8, 21.7, 22.6],
    'Third Quintile': [18.0, 18.3, 18.8, 17.9, 19.6, 18.6],
    'Fourth Quintile': [17.0, 17.7, 17.0, 18.0, 17.9, 18.1],
    'Fifth Quintile': [16.6, 16.5, 15.4, 15.6, 17.3, 16.9]
}

healthcare_access_data = {
    'First Quintile': [77.3, 79.7, 81.5, 81.2, 82.3, 79.9],
    'Second Quintile': [83.7, 85.0, 84.6, 84.5, 86.7, 82.9],
    'Third Quintile': [87.1, 86.2, 85.9, 86.0, 85.1, 83.2],
    'Fourth Quintile': [87.0, 87.6, 87.6, 87.3, 87.3, 84.0],
    'Fifth Quintile': [90.0, 88.3, 89.6, 88.0, 87.5, 84.1]
}

# Flatten data into DataFrame
high_blood_pressure_flat = np.array(list(high_blood_pressure_data.values())).flatten()
healthcare_access_flat = np.array(list(healthcare_access_data.values())).flatten()

combined_df = pd.DataFrame({
    'High Blood Pressure': high_blood_pressure_flat,
    'Healthcare Access': healthcare_access_flat
})

# Categorize high blood pressure
bins = [0, 18, 20, 22, 27]
labels = ['Low', 'Medium', 'High', 'Very High']
combined_df['BP Category'] = pd.cut(combined_df['High Blood Pressure'], bins=bins, labels=labels)

# Encode the target variable
label_encoder = LabelEncoder()
combined_df['BP Category'] = label_encoder.fit_transform(combined_df['BP Category'])

# Define features (X) and target (y)
X = combined_df[['Healthcare Access']]
y = combined_df['BP Category']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Classification Report
target_names = label_encoder.classes_
classes_in_y_test = np.unique(y_test)
target_names_filtered = [target_names[i] for i in classes_in_y_test]

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names_filtered, labels=classes_in_y_test))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=classes_in_y_test)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_filtered, yticklabels=target_names_filtered)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Coefficient Visualization
coefs = model.coef_

plt.figure(figsize=(10, 6))
for i, category in enumerate(target_names_filtered):
    plt.bar(category, coefs[i][0], label=f"{category} (Coefficient)")

plt.title('Logistic Regression Coefficients for Blood Pressure Categories')
plt.xlabel('Blood Pressure Category')
plt.ylabel('Coefficient Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
