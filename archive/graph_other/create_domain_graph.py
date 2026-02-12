import json
from graph_tool.all import Graph
from tqdm import tqdm
import pandas as pd
import argparse


# pre_load = True

# if pre_load:
#     g.load("argument_graph.gt")
# else:
#     


# from collections import Counter
# num_vertices = g.num_vertices()
# num_edges = g.num_edges()
# print("📊 Graph Statistics:")
# print(f"  • Number of nodes: {num_vertices}")
# print(f"  • Number of edges: {num_edges}")
# isolated_nodes = [v for v in g.vertices() if v.out_degree() + v.in_degree() == 0]
# isolated_types = [g.vp["type"][v] for v in isolated_nodes]

# # Count occurrences
# type_counts = Counter(isolated_types)

# # Print the results
# print("📌 Isolated Nodes by Type:")
# for t, count in type_counts.items():
#     print(f"  • {t}: {count}")

# # Calculate density
# v_type = g.vp["type"]
# num_issues = sum(1 for v in g.vertices() if v_type[v] == "issue")
# num_arguments = sum(1 for v in g.vertices() if v_type[v] == "argument")

# # Compute bipartite density
# if num_issues > 0 and num_arguments > 0:
#     max_possible_edges = num_issues * num_arguments
#     density = num_edges / max_possible_edges
# else:
#     density = 0.0  # Handle edge case with 0 nodes

# print(f"Graph density: {density:.4f}")

# # Store degrees
# issue_in_degrees = []
# argument_out_degrees = []

# for v in g.vertices():
#     if v_type[v] == "issue":
#         issue_in_degrees.append(v.in_degree())
#     elif v_type[v] == "argument":
#         argument_out_degrees.append(v.out_degree())

# # Compute stats
# def describe(degrees):
#     return {
#         "avg": sum(degrees) / len(degrees) if degrees else 0,
#         "min": min(degrees) if degrees else 0,
#         "max": max(degrees) if degrees else 0,
#         "count": len(degrees)
#     }

# issue_stats = describe(issue_in_degrees)
# arg_stats = describe(argument_out_degrees)

# # Print results
# print("📊 Issue Node (In-Degree) Stats:")
# print(f"  Count: {issue_stats['count']}")
# print(f"  Average # of arguments per issue: {issue_stats['avg']:.2f}")
# print(f"  Min: {issue_stats['min']}, Max: {issue_stats['max']}")

# print("\n📊 Argument Node (Out-Degree) Stats:")
# print(f"  Count: {arg_stats['count']}")
# print(f"  Average # of issues per argument: {arg_stats['avg']:.2f}")
# print(f"  Min: {arg_stats['min']}, Max: {arg_stats['max']}")
    

def main():
    DOMAINS = ['Arts', 'Religion', 'Health', 'Science & Technology', 'History', 'Economy', 'Literature', 'Other', 'International Relations', 'Environment', 'Philosophy', 'Education', 'Sports', 'Politics & Government', 'Law', 'Society & Culture']

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', help='Extract all nodes related to this domain.', required=True, choices=DOMAINS)
    parser.add_argument('-i', '--input_file', help='File with preprocessed scraping data.', required=True)
    parser.add_argument('-v', '--values_file', help='File with predicted human values', required=False)
    args = parser.parse_args()

    # Prepare graph with node/edge attributes
    g = Graph(directed=True)
    type_prop = g.new_vertex_property("string")
    text_prop = g.new_vertex_property("string")
    uuid_prop = g.new_vertex_property("string")
    edge_prop = g.new_edge_property("string")

    # Read in the graph data
    with open(args.input_file, 'r') as f:
        seed_data = json.load(f)

    # Create nodes
    # Issue Node: type, text, uuid, domain
    # Argument Node: type, text, uuid
    # Value Node: type, text

    # Create edges
    # Argument edges: supports, attacks
    # Value edges: attains, constrains

    for row in tqdm(seed_data):

        if not (args.domain in row.get('domain', '')):
            continue
        
        # ********************************************************************************
        # Create Issue Nodes
        # ********************************************************************************
        issue_v = g.add_vertex()

        type_prop[issue_v] = "issue"
        text_prop[issue_v] = row.get("issue", "")
        uuid_prop[issue_v] = row.get("uuid", "")

        # ********************************************************************************
        # Create Argument Nodes / Argument --> Issue Edges
        # ********************************************************************************

        # Dictionary to keep track of argument text -> vertex (to prevent duplicate argument nodes)
        argument_nodes = {}
        row_arguments = row.get("arguments", {})
        
        if row_arguments:
            for stance in ["pro", "con"]:
                arguments = row_arguments.get(f"{stance}_arguments", [])
                
                for i, arg in enumerate(arguments):
                    if arg in argument_nodes:
                        arg_v = argument_nodes[arg]
                    elif arg:
                        arg_v = g.add_vertex()
                        type_prop[arg_v] = "argument"
                        text_prop[arg_v] = arg
                        uuid_prop[arg_v] = f"{0}-{1}-{2}".format(uuid_prop[issue_v], stance, i)
                        argument_nodes[arg] = arg_v

                    # Add support / attack edges
                    edge = g.add_edge(arg_v, issue_v)
                    edge_prop[edge] = "supports" if stance == "pro" else "attacks"

    # ********************************************************************************
    # Create Value Nodes / Issue --> Value & Argument --> Value Edges
    # ********************************************************************************

    
    if args.values_file:
        human_values = [
            "Self-direction: thought",
            "Self-direction: action",
            "Stimulation",
            "Hedonism",
            "Achievement",
            "Power: dominance",
            "Power: resources",
            "Face",
            "Security: personal",
            "Security: societal",
            "Tradition",
            "Conformity: rules",
            "Conformity: interpersonal",
            "Humility",
            "Benevolence: caring",
            "Benevolence: dependability",
            "Universalism: concern",
            "Universalism: nature",
            "Universalism: tolerance"
        ]

        df = pd.read_csv(args.values_file, sep='\t')

        # Collapse all sentence predictions into one.
        # Text-ID,  Sentence-ID,    Val1    Val2    Val3
        # example   1               0       1       0
        # example   2               0       0       0
        # example   3               1       1       0

        # Becomes

        # Text-ID,  Sentence-ID,    Val1    Val2    Val3
        # example   1               1       1       0
        value_columns = [f"{value} {attainment}" for attainment in ["constrained", "attained"] for value in human_values]
        grouped = df.groupby("Text-ID")[value_columns].max()

        # Map issue/arg nodes to uuids
        uuids_to_nodes = {}
        for v in g.vertices():
            uuids_to_nodes[uuid_prop[v]] = v

        # Create human value nodes and link them to issues/arguments
        none_node_count = 0
        for value in tqdm(human_values):
            value_v = g.add_vertex()
            type_prop[value_v] = "value"
            text_prop[value_v] = value
            uuid_prop[value_v] = ""

            for edge_type in ["attained", "constrained"]:

                # Find all rows where this value is present
                column_name = value + ' ' + edge_type
                value_df = grouped[grouped[column_name] == 1]

                # Find the nodes that belong to these rows
                nodes = df['Text-ID'].apply(lambda x: uuids_to_nodes.get(x, None))

                # Link each of these nodes to the current value
                for node in nodes:
                    if node is None:
                        none_node_count += 1
                        continue
                    edge = g.add_edge(node, value_v)
                    edge_prop[edge] = "attains" if edge_type == "attained" else "constrains"

        print(f"Detected {none_node_count} None nodes")``

    # Attach the property maps to the graph
    g.vertex_properties["type"] = type_prop
    g.vertex_properties["text"] = text_prop
    g.vertex_properties["uuid"] = uuid_prop
    g.edge_properties["type"] = edge_prop

    g.save(f"graph_{args.domain}.gt")



if __name__ == "__main__":
    main()




