import pandas as pd
import json

df1 = pd.read_json('../seed_data/data_collection/seed_data_panda.json')
df2 = pd.read_csv('graph_run_education.tsv', sep='\t')

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

value_columns = [f"{value} {attainment}" for attainment in ["constrained", "attained"] for value in human_values]
grouped = df2.groupby("Text-ID")[value_columns].max()
filtered = grouped[grouped.index.str.contains("-pro|-con")]
print(filtered.sum(axis=1).sum())
exit()

# print(set(df1.uuid).issubset(df2['Text-ID']))

pro_arg_count = 0
con_arg_count = 0
unique_uuids = set()
for _, row in df1.iterrows():
    args = row.get('arguments', {})
    unique_uuids.add(row['uuid'])
    if args:
        try:
            pro_arg_count += len(args['pro_arguments'])
            con_arg_count += len(args['con_arguments'])
        except KeyError:
            print('no')

print(pro_arg_count)
print(con_arg_count)
print(len(unique_uuids))

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
