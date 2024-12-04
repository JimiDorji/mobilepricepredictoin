import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('static/mobile_price_classification.csv')

# Print the columns to inspect
print("Columns in dataset:", data.columns)

# Remove duplicates
data = data.drop_duplicates()

# Adjusted selected features based on correct column names
selected_features = [
    'battery', 'ram', 'cpu core', 'cpu freq', 'resoloution', 'ppi', 'RearCam', 'thickness'
]

# Create X with only the selected 8 features
X = data[selected_features]

# Use actual price for regression
y = data['Price']  # Change target to actual price

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (recommended for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, 'scaler_8_features.pkl')

# Train the RandomForest model (regression)
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(model, 'best_model_8_features.pkl')
print("Model and scaler saved successfully with 8 features.")

# Evaluate the model
y_pred = model.predict(X_test_scaled)

# Calculate R^2 score (Good measure for regression models)
r2_score = model.score(X_test_scaled, y_test)
print(f"Model R^2 score on test data: {r2_score:.2f}")

# Plot feature importances to see which features are most important
feature_importances = model.feature_importances_
feature_names = X.columns
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
