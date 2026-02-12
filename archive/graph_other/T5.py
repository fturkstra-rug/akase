import argparse
import torch

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph", help="File with PyG graph.", required=True, type=set)
    return parser.parse_args()

def main():
    args = create_arg_parser()
    
    data = torch.load(args.graph)

    # Total number of issue nodes
    num_issues = data['issue'].x.size(0)

    # Sample N random issue node indices
    N = 5
    sampled_issue_indices = torch.randint(0, num_issues, (N,))
    print(f"Sampled issue node indices: {sampled_issue_indices.tolist()}")

    # Get edges: issue -> value
    attains_edges = data['issue', 'attains', 'value'].edge_index
    constrains_edges = data['issue', 'constrains', 'value'].edge_index

    # Convert to dictionaries for quick lookup
    from collections import defaultdict

    def build_issue_to_value_dict(edge_index):
        mapping = defaultdict(list)
        for src, dst in edge_index.t().tolist():
            mapping[src].append(dst)
        return mapping

    attains_dict = build_issue_to_value_dict(attains_edges)
    constrains_dict = build_issue_to_value_dict(constrains_edges)

    # Show connections for sampled issues
    for issue_idx in sampled_issue_indices.tolist():
        attains_values = attains_dict.get(issue_idx, [])
        constrains_values = constrains_dict.get(issue_idx, [])

        print(f"\nIssue {issue_idx}:")
        print(f"  Attains values: {attains_values}")
        print(f"  Constrains values: {constrains_values}")

    # Select issue nodes
    # Find corresponding value nodes
    # Sample argument nodes

if __name__ == "__main__":
    main()
