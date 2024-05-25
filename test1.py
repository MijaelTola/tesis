import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

# Número de nodos
n = 1000

# Número de simulaciones para promediar los resultados
num_simulations = 10

# Parámetros para probar en cada modelo
p_values = [0.001, 0.01, 0.05, 0.1]  # Para Erdős-Rényi
m_values = [1, 2, 5, 10]             # Para Barabási-Albert
beta_values = [0.1, 0.3, 0.5, 0.7, 1.0]  # Para Watts-Strogatz
k = 4  # Grado inicial para Watts-Strogatz

# Función para calcular la distribución de grados
def plot_degree_distribution(G, title):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins=np.linspace(0, max(degrees), max(degrees) + 1), alpha=0.75, density=True)
    plt.title(title)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Función para calcular el coeficiente de clustering
def average_clustering(G):
    return nx.average_clustering(G)

# Función para calcular el camino más corto promedio
def average_shortest_path_length(G):
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        return float('inf')  # Si el grafo no es conexo

# Experimentos con Erdős-Rényi
print("Erdős-Rényi Model Experiments")
for p in p_values:
    clustering_coeffs = []
    path_lengths = []
    for _ in range(num_simulations):
        G = nx.erdos_renyi_graph(n, p)
        clustering_coeffs.append(average_clustering(G))
        if nx.is_connected(G):
            path_lengths.append(average_shortest_path_length(G))
    print(f"p = {p}: Average Clustering Coefficient = {mean(clustering_coeffs)}, Average Shortest Path Length = {mean(path_lengths) if path_lengths else 'Disconnected'}")
    plot_degree_distribution(G, f"Erdős-Rényi Graph with p={p}")

# Experimentos con Barabási-Albert
print("\nBarabási-Albert Model Experiments")
for m in m_values:
    clustering_coeffs = []
    path_lengths = []
    for _ in range(num_simulations):
        G = nx.barabasi_albert_graph(n, m)
        clustering_coeffs.append(average_clustering(G))
        if nx.is_connected(G):
            path_lengths.append(average_shortest_path_length(G))
    print(f"m = {m}: Average Clustering Coefficient = {mean(clustering_coeffs)}, Average Shortest Path Length = {mean(path_lengths) if path_lengths else 'Disconnected'}")
    plot_degree_distribution(G, f"Barabási-Albert Graph with m={m}")

# Experimentos con Watts-Strogatz
print("\nWatts-Strogatz Model Experiments")
for beta in beta_values:
    clustering_coeffs = []
    path_lengths = []
    for _ in range(num_simulations):
        G = nx.watts_strogatz_graph(n, k, beta)
        clustering_coeffs.append(average_clustering(G))
        if nx.is_connected(G):
            path_lengths.append(average_shortest_path_length(G))
    print(f"beta = {beta}: Average Clustering Coefficient = {mean(clustering_coeffs)}, Average Shortest Path Length = {mean(path_lengths) if path_lengths else 'Disconnected'}")
    plot_degree_distribution(G, f"Watts-Strogatz Graph with beta={beta}")

