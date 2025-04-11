import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

def build_ingredient_graph(recipe_json_path, output_graph_path=None, min_co_occurrence=1, alpha=0.6):
    """
    Build and visualize an ingredient co-occurrence graph from recipe data
    
    Args:
        recipe_json_path: Path to the JSON file containing recipe data
        output_graph_path: Path to save the visualization (optional)
        min_co_occurrence: Minimum number of co-occurrences to include an edge
        alpha: Weight parameter balancing co-occurrence vs nutritional similarity
               where wij = α · cij + (1 − α) · KL(ni||nj)
    
    Returns:
        G: NetworkX graph of ingredient relationships
        co_occurrence_matrix: The ingredient co-occurrence matrix
        ingredient_idx: Mapping of ingredients to indices
    """
    # Load recipe data
    print(f"Loading recipe data from {recipe_json_path}...")
    with open(recipe_json_path, 'r') as f:
        recipes = json.load(f)
    
    print(f"Loaded {len(recipes)} recipes")
    
    # Extract all unique ingredients
    all_ingredients = set()
    for recipe in recipes:
        for ingredient in recipe['ingredients']:
            all_ingredients.add(ingredient['text'])
    
    ingredient_list = sorted(list(all_ingredients))
    ingredient_idx = {ing: i for i, ing in enumerate(ingredient_list)}
    
    print(f"Found {len(ingredient_list)} unique ingredients")
    
    # Initialize co-occurrence matrix
    n_ingredients = len(ingredient_list)
    co_occurrence_matrix = np.zeros((n_ingredients, n_ingredients))
    
    # Calculate co-occurrences
    print("Building co-occurrence matrix...")
    for recipe in recipes:
        recipe_ingredients = [ing['text'] for ing in recipe['ingredients']]
        # For each pair of ingredients in the recipe
        for i, ing1 in enumerate(recipe_ingredients):
            idx1 = ingredient_idx[ing1]
            for j, ing2 in enumerate(recipe_ingredients):
                if i != j:  # Don't count ingredient with itself
                    idx2 = ingredient_idx[ing2]
                    co_occurrence_matrix[idx1, idx2] += 1
    
    # Calculate nutritional similarity using KL divergence if nutrient data is available
    nutrient_similarity_matrix = np.zeros((n_ingredients, n_ingredients))
    
    # Extract nutritional vectors for each ingredient (if available)
    nutrient_vectors = {}
    for recipe in recipes:
        for i, ingredient in enumerate(recipe['ingredients']):
            ing_name = ingredient['text']
            if 'nutr_per_ingredient' in recipe and i < len(recipe['nutr_per_ingredient']):
                nutrient_data = recipe['nutr_per_ingredient'][i]
                if ing_name not in nutrient_vectors:
                    nutrient_vectors[ing_name] = []
                # Collect all nutrient vectors for this ingredient
                nutrient_vectors[ing_name].append([
                    nutrient_data.get('fat', 0),
                    nutrient_data.get('nrg', 0),
                    nutrient_data.get('pro', 0),
                    nutrient_data.get('sat', 0),
                    nutrient_data.get('sod', 0),
                    nutrient_data.get('sug', 0)
                ])
    
    # Average nutrient vectors for each ingredient
    for ing_name, vectors in nutrient_vectors.items():
        if vectors:
            nutrient_vectors[ing_name] = np.mean(vectors, axis=0)
    
    # Calculate KL divergence between nutrient vectors
    for i, ing1 in enumerate(ingredient_list):
        for j, ing2 in enumerate(ingredient_list):
            if i != j and ing1 in nutrient_vectors and ing2 in nutrient_vectors:
                vec1 = nutrient_vectors[ing1]
                vec2 = nutrient_vectors[ing2]
                
                # Avoid zero values for KL divergence
                vec1 = np.maximum(vec1, 1e-10)
                vec2 = np.maximum(vec2, 1e-10)
                
                # Normalize vectors to sum to 1 (probability distributions)
                vec1 = vec1 / np.sum(vec1)
                vec2 = vec2 / np.sum(vec2)
                
                # Calculate symmetric KL divergence
                kl_div = 0.5 * (entropy(vec1, vec2) + entropy(vec2, vec1))
                
                # Convert to similarity (higher = more similar)
                nutrient_similarity = 1.0 / (1.0 + kl_div)
                nutrient_similarity_matrix[i, j] = nutrient_similarity
    
    # Normalize matrices
    co_occurrence_norm = co_occurrence_matrix / (np.max(co_occurrence_matrix) or 1)
    nutrient_similarity_norm = nutrient_similarity_matrix / (np.max(nutrient_similarity_matrix) or 1)
    
    # Combine matrices using alpha weight
    combined_similarity = alpha * co_occurrence_norm + (1 - alpha) * nutrient_similarity_norm
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes (ingredients)
    for ingredient in ingredient_list:
        G.add_node(ingredient)
    
    # Add edges for co-occurrences that meet the minimum threshold
    print("Building graph...")
    for i in range(n_ingredients):
        for j in range(i+1, n_ingredients):  # Only upper triangle to avoid duplicates
            if co_occurrence_matrix[i, j] >= min_co_occurrence:
                weight = combined_similarity[i, j]
                if weight > 0:
                    G.add_edge(
                        ingredient_list[i], 
                        ingredient_list[j], 
                        weight=weight,
                        co_occurrence=co_occurrence_matrix[i, j],
                        nutrient_similarity=nutrient_similarity_matrix[i, j]
                    )
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Filter to keep only the main connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Filtered to largest connected component: {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Visualize graph
    if output_graph_path:
        visualize_ingredient_graph(G, output_path=output_graph_path)
    
    return G, co_occurrence_matrix, ingredient_idx

def visualize_ingredient_graph(G, output_path=None, layout='spring', node_size_factor=100, edge_width_factor=1.0):
    """
    Visualize the ingredient graph
    
    Args:
        G: NetworkX graph of ingredients
        output_path: Path to save the visualization
        layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai')
        node_size_factor: Factor to scale node sizes
        edge_width_factor: Factor to scale edge widths
    """
    plt.figure(figsize=(20, 16))
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Calculate node sizes based on degree
    node_sizes = [G.degree(node) * node_size_factor for node in G.nodes()]
    
    # Get edge weights for line thickness
    edge_weights = [G[u][v]['weight'] * edge_width_factor for u, v in G.edges()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3, edge_color='gray')
    
    # Only label nodes with degree > 1 to reduce clutter
    high_degree_nodes = {node: node for node in G.nodes() if G.degree(node) > 1}
    nx.draw_networkx_labels(G, pos, labels=high_degree_nodes, font_size=8, font_weight='bold')
    
    plt.title("Ingredient Co-occurrence Graph", fontsize=20)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def find_ingredient_substitutions(G, ingredient, top_n=5, exclude_similar=None):
    """
    Find potential substitutions for an ingredient based on the graph
    
    Args:
        G: NetworkX graph of ingredients
        ingredient: The ingredient to find substitutes for
        top_n: Number of substitutes to return
        exclude_similar: List of ingredients to exclude from results
    
    Returns:
        List of (substitute, similarity_score) tuples
    """
    if ingredient not in G:
        print(f"Ingredient '{ingredient}' not found in graph")
        return []
    
    if exclude_similar is None:
        exclude_similar = []
    
    # Get all neighbors with their edge weights
    neighbors = [(neighbor, G[ingredient][neighbor]['weight']) 
                 for neighbor in G.neighbors(ingredient) 
                 if neighbor not in exclude_similar]
    
    # Sort by weight (higher = more similar)
    neighbors.sort(key=lambda x: x[1], reverse=True)
    
    return neighbors[:top_n]

def analyze_recipe_compatibility(G, recipe_ingredients):
    """
    Analyze compatibility of ingredients in a recipe
    
    Args:
        G: NetworkX graph of ingredients
        recipe_ingredients: List of ingredients in the recipe
    
    Returns:
        Compatibility matrix and overall compatibility score
    """
    n = len(recipe_ingredients)
    compatibility_matrix = np.zeros((n, n))
    
    # Build compatibility matrix
    for i in range(n):
        for j in range(i+1, n):
            ing1 = recipe_ingredients[i]
            ing2 = recipe_ingredients[j]
            
            if ing1 in G and ing2 in G and G.has_edge(ing1, ing2):
                compatibility = G[ing1][ing2]['weight']
            else:
                compatibility = 0
                
            compatibility_matrix[i, j] = compatibility
            compatibility_matrix[j, i] = compatibility
    
    # Calculate overall compatibility (average of non-zero weights)
    nonzero_weights = compatibility_matrix[compatibility_matrix > 0]
    if len(nonzero_weights) > 0:
        overall_compatibility = np.mean(nonzero_weights)
    else:
        overall_compatibility = 0
    
    return compatibility_matrix, overall_compatibility

def main():
    # Path to your recipe JSON file
    # recipe_json_path = "chunk_12317.json"
    recipe_json_path = "/Users/shrutisriram/Desktop/foodgraph/recipes_with_nutritional_info.json"
    
    # Build the graph
    G, co_occurrence_matrix, ingredient_idx = build_ingredient_graph(
        recipe_json_path, 
        output_graph_path="ingredient_graph.png",
        min_co_occurrence=1,
        alpha=0.6
    )
    
    # Display top 5 co-occurring ingredient pairs
    edge_data = [(u, v, d['co_occurrence']) for u, v, d in G.edges(data=True)]
    edge_data.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 co-occurring ingredient pairs:")
    for u, v, co_occurrence in edge_data[:5]:
        print(f"{u} - {v}: {co_occurrence}")
    
    # Find substitutes for a sample ingredient
    sample_ingredient = "milk, fluid, 1% fat, without added vitamin a and vitamin d"
    if sample_ingredient in G:
        print(f"\nPotential substitutes for '{sample_ingredient}':")
        substitutes = find_ingredient_substitutions(G, sample_ingredient)
        for sub, score in substitutes:
            print(f"- {sub} (similarity: {score:.3f})")
    
    # Analyze compatibility of ingredients in the first recipe
    with open(recipe_json_path, 'r') as f:
        recipes = json.load(f)
    
    if recipes:
        first_recipe = recipes[0]
        recipe_ingredients = [ing['text'] for ing in first_recipe['ingredients']]
        
        print(f"\nAnalyzing compatibility for recipe: {first_recipe['title']}")
        compatibility_matrix, overall_score = analyze_recipe_compatibility(G, recipe_ingredients)
        
        print(f"Overall ingredient compatibility score: {overall_score:.3f}")
        
        # Save co-occurrence matrix to CSV
        ingredient_list = [ing for ing in ingredient_idx.keys()]
        co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=ingredient_list, columns=ingredient_list)
        co_occurrence_df.to_csv("ingredient_co_occurrence_matrix.csv")
        print("\nCo-occurrence matrix saved to 'ingredient_co_occurrence_matrix.csv'")

if __name__ == "__main__":
    main()