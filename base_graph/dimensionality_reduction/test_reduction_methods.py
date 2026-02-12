import numpy as np
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
import umap
from sklearn.manifold import TSNE
import hdbscan


class ReductionMethod:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.projected_data = None
        self.cluster_labels = None
        self.silhouette_score = None
        self.num_clusters = None
        self.num_noise = None

num_dimensions_list = [5, 10, 25, 50, 100, 339, 482]

for num_dimensions in tqdm(num_dimensions_list, desc="Testing dimensions"):
    models = [
        ReductionMethod(name="ica", model=FastICA(n_components=num_dimensions)),
        ReductionMethod(name="pca", model=PCA(n_components=num_dimensions)),
        ReductionMethod(name="lle", model=LocallyLinearEmbedding(n_components=num_dimensions)),
        ReductionMethod(name="iso", model=Isomap(n_components=num_dimensions)),
        ReductionMethod(name="mds", model=MDS(n_components=num_dimensions)),
        ReductionMethod(name="umap", model=umap.UMAP(n_components=num_dimensions)),
        ReductionMethod(name="tsne", model=TSNE(n_components=num_dimensions)),
    ]

    query_embeddings = np.load("sample_1000.npy", allow_pickle=True)
    # query_embeddings = np.memmap("../../embeddings/query_embeddings.memmap", dtype=np.float32, mode="r", shape=(33312, 4096))

    for model in tqdm(models, desc="Projecting data"):
        try:
            model.projected_data = model.model.fit_transform(query_embeddings)
        except Exception as e:
            print(e)

    for model in models:
        if model.projected_data is None:
            model.silhouette_score = np.nan
            continue

        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
            model.cluster_labels = clusterer.fit_predict(model.projected_data)
        
            model.silhouette_score = silhouette_score(model.projected_data, model.cluster_labels)
            unique_labels, counts = np.unique(model.cluster_labels, return_counts=True)
            model.num_noise = np.sum(model.cluster_labels == -1)
            model.num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        except Exception as e:
            print(e)
            model.silhouette_score = np.nan

    import csv

    # Open a CSV file for writing
    output_file = f"results_d{num_dimensions}.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write the header row
        writer.writerow(["Model", "Dimensions", "Silhouette", "Clusters", "Noise"])
        
        # Write the data for each model
        for model in models:
            writer.writerow([
                model.name, 
                num_dimensions,  # Assuming 'model.model.n_components' is the number of dimensions used
                f"{model.silhouette_score:.4f}",  # Format silhouette score
                model.num_clusters, 
                model.num_noise
            ])

    print(f"Results saved to {output_file}")
