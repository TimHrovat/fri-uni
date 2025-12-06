import random

def generate_directed_weighted_graph(num_nodes, num_edges, output_file='graph_output.txt'):
    """
    Generate a random directed weighted graph and save it to a file.
    
    Parameters:
    - num_nodes: Number of nodes in the graph
    - num_edges: Number of edges in the graph
    - output_file: Name of the output file (default: 'graph_output.txt')
    
    The weights are randomly chosen between 1 and 10 (inclusive).
    """
    
    # Validate input
    max_possible_edges = num_nodes * (num_nodes - 1)  # directed graph allows self-loops excluded
    if num_edges > max_possible_edges:
        raise ValueError(f"Too many edges! Maximum possible edges for {num_nodes} nodes is {max_possible_edges}")
    
    if num_nodes < 1:
        raise ValueError("Number of nodes must be at least 1")
    
    if num_edges < 0:
        raise ValueError("Number of edges cannot be negative")
    
    # Generate all possible edges (excluding self-loops)
    all_possible_edges = [(i, j) for i in range(1, num_nodes + 1) 
                          for j in range(1, num_nodes + 1) if i != j]
    
    # Randomly select edges
    selected_edges = random.sample(all_possible_edges, min(num_edges, len(all_possible_edges)))
    
    # Assign random weights to edges
    edges_with_weights = [(u, v, random.randint(1, 10)) for u, v in selected_edges]
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(f"{num_nodes}\n")
        for u, v, weight in edges_with_weights:
            f.write(f"{u} {v} {weight}\n")
    
    print(f"Graph generated successfully with {num_nodes} nodes and {num_edges} edges.")
    print(f"Output saved to '{output_file}'")
    
    return edges_with_weights


# Example usage
if __name__ == "__main__":
    # Generate a graph with 5 nodes and 8 edges
    generate_directed_weighted_graph(1000, 10000)
    
    # Read and display the generated graph
    print("\nGenerated graph content:")
    with open('graph_output.txt', 'r') as f:
        print(f.read())
