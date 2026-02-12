import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

query_embeddings = np.load("sample_1000.npy", allow_pickle=True)
pca = PCA()
pca.fit(query_embeddings)

# Compute cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find the number of components for 90% and 95% variance
num_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
num_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Number of components to retain 90% variance: {num_components_90}")
print(f"Number of components to retain 95% variance: {num_components_95}")

# Plot explained variance
plt.figure(figsize=(8,5))
plt.plot(cumulative_variance, marker='o', label="Cumulative Variance")
plt.axhline(y=0.90, color='r', linestyle='--', label="90% Variance")
plt.axhline(y=0.95, color='g', linestyle='--', label="95% Variance")
plt.axvline(x=num_components_90, color='r', linestyle='--')
plt.axvline(x=num_components_95, color='g', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of PCA Components')
plt.legend()
plt.grid()
plt.show()
