import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Paths
input_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\data\processed\customer_data_cleaned.csv"
model_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\src\kmeans_model.pkl"
scaler_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\src\scaler.pkl"
required_features_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\data\processed\required_features.pkl"

# Step 1: Load Data
data = pd.read_csv(input_path)

# Step 2: Select Numerical Columns
numerical_cols = data.select_dtypes(include='number').columns

# Step 3: Scale the Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numerical_cols])

# Step 4: Train the KMeans Model
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Step 5: Save the KMeans Model and Scaler
joblib.dump(kmeans, model_path)     # Save KMeans model
joblib.dump(scaler, scaler_path)    # Save Scaler as scaler.pkl

# Step 6: Save Required Features
required_features = numerical_cols.tolist()
joblib.dump(required_features, required_features_path)

# Step 7: Save Clustered Data
output_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\data\processed\clustered_data.csv"
data.to_csv(output_path, index=False)

print("Training Complete. KMeans model and scaler saved successfully.")
