import numpy as np
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import json

num_dimensions = 50
min_cluster_size = 2
input_file = "embeddings.memmap"

model = umap.UMAP(n_components=num_dimensions, n_jobs=-1)
query_embeddings = np.memmap(input_file, dtype=np.float32, mode="r", shape=(29965, 4096))
# query_embeddings = np.memmap(input_file, dtype=np.float32, mode="r", shape=(30876, 4096))
# query_embeddings = np.load("dimensionality_experiments/sample_1000.npy", allow_pickle=True)

projected_data = model.fit_transform(query_embeddings)

clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=-1)
cluster_labels = clusterer.fit_predict(projected_data)
            
score = silhouette_score(projected_data, cluster_labels)
unique_labels, counts = np.unique(cluster_labels, return_counts=True)
noise = np.sum(cluster_labels == -1)
num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

results = {
    "silhouette_score": float(score),
    "num_clusters": num_clusters,
    "noise": int(noise),
    "cluster_counts": {int(label): int(count) for label, count in zip(unique_labels, counts)},
    "cluster_labels": cluster_labels.tolist()
}

# Save results as JSON
output_file = "cluster_results.json"
with open(output_file, "w") as file:
    json.dump(results, file)

print(f"Saved results to {output_file}")
