from graph_tool.all import Graph

g = Graph(directed=True)
g.load("argument_graph.gt")

# Print number of vertices and eges
num_vertices = g.num_vertices()
num_edges = g.num_edges()
print(f"Number of nodes: {num_vertices}")
print(f"Number of edges: {num_edges}")

# Calculate bipartite density
v_type = g.vp["type"]
num_issues = sum(1 for v in g.vertices() if v_type[v] == "issue")
num_arguments = sum(1 for v in g.vertices() if v_type[v] == "argument")

if num_issues > 0 and num_arguments > 0:
    max_possible_edges = num_issues * num_arguments
    density = num_edges / max_possible_edges
else:
    density = 0.0  # Handle edge case with 0 nodes

# Check in/out degreess
issue_in_degrees = []
argument_out_degrees = []

for v in g.vertices():
    if v_type[v] == "issue":
        issue_in_degrees.append(v.in_degree())
    elif v_type[v] == "argument":
        argument_out_degrees.append(v.out_degree())

def describe(degrees):
    return {
        "avg": sum(degrees) / len(degrees) if degrees else 0,
        "min": min(degrees) if degrees else 0,
        "max": max(degrees) if degrees else 0,
        "count": len(degrees)
    }

issue_stats = describe(issue_in_degrees)
arg_stats = describe(argument_out_degrees)

# Print results
print("Issue Node (In-Degree):")
print(f"  Count: {issue_stats['count']}")
print(f"  Average # of arguments per issue: {issue_stats['avg']:.2f}")
print(f"  Min: {issue_stats['min']}, Max: {issue_stats['max']}")

print("Argument Node (Out-Degree):")
print(f"  Count: {arg_stats['count']}")
print(f"  Average # of issues per argument: {arg_stats['avg']:.2f}")
print(f"  Min: {arg_stats['min']}, Max: {arg_stats['max']}")
