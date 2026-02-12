import json
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

def analyze_connectivity(jsonl_path, plot=True):
    """
    Analyze how connected the document similarity graph is.

    Args:
        jsonl_path (str): Path to the JSONL file.
        plot (bool): Whether to plot component size distribution.
    """
    G = nx.Graph()

    print(f"Loading graph from {jsonl_path} ...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading documents"):
            data = json.loads(line)
            doc_id = data["doc_id"]
            neighbors = data.get("neighbors", [])
            for n in neighbors:
                G.add_edge(doc_id, n)

    print("\nGraph summary:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    print("\nComputing connected components ...")
    components = list(nx.connected_components(G))
    component_sizes = sorted([len(c) for c in components], reverse=True)

    print(f"  Total connected components: {len(components)}")
    print(f"  Largest component size: {component_sizes[0]:,}")
    print(f"  Smallest component size: {component_sizes[-1]:,}")

    # Percentage of nodes in largest component
    perc = component_sizes[0] / G.number_of_nodes() * 100
    print(f"  Percentage in largest component: {perc:.2f}%")

    if plot:
        plt.figure(figsize=(8,5))
        plt.hist(component_sizes, bins=50, log=True)
        plt.title("Distribution of Connected Component Sizes")
        plt.xlabel("Component size")
        plt.ylabel("Count (log scale)")
        plt.show()

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_components": len(components),
        "component_sizes": component_sizes,
        "largest_component_ratio": perc
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze graph connectivity in document similarity data.")
    parser.add_argument("jsonl_path", help="Path to the JSONL file containing doc_id and neighbors.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting of component size distribution.")
    args = parser.parse_args()

    analyze_connectivity(args.jsonl_path, plot=not args.no_plot)
