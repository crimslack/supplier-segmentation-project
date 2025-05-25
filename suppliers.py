import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Read Excel file
df = pd.read_excel("supplier_ranking_grades.xlsx", sheet_name="Sheet2")

# Display first 5 rows
print(df.head())

# Check data types
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())

# Simplify column names
df.columns = [
    "supplier", "quality", "quantity", "payment", "service", "reputation",
    "flexibility", "finance", "assets", "employees", "price", "delivery_time", "location"
]

# Display first 5 rows again
print(df.head())
df.columns

# Convert all columns (except 'supplier') to numeric (non-convertible values become NaN)
for col in df.columns:
    if col != "supplier":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Check missing values again
print(df.isnull().sum())

# Drop rows with missing values
df_clean = df.dropna()

# Display cleaned dataset shape
print("Cleaned data shape:", df_clean.shape)

# Scale numerical features (excluding 'supplier')
features = df_clean.drop(columns=["supplier"])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert to Pandas DataFrame if needed
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(scaled_df, method='ward'))
plt.title("Supplier Segmentation - Dendrogram")
plt.xlabel("Suppliers")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# Create clustering model
cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = cluster.fit_predict(scaled_df)

# Add cluster labels to the original data
df_clean["cluster"] = labels

# Calculate mean values for each cluster
cluster_summary = df_clean.groupby("cluster").mean(numeric_only=True)
print("Cluster Averages:\n", cluster_summary)

# Print the number of suppliers in each cluster
print("\nNumber of suppliers per cluster:")
print(df_clean["cluster"].value_counts())

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_df)

df_clean["pca1"] = pca_result[:, 0]
df_clean["pca2"] = pca_result[:, 1]

# Visualize clusters in 2D PCA space
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_clean, x="pca1", y="pca2", hue="cluster", palette="Set2")
plt.title("Supplier Clustering Result (2D PCA)")
plt.show()
