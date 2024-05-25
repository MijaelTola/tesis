import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función para generar un grafo Erdős-Rényi G(n, p)
def erdos_renyi_graph(n, p):
    G = {}
    for i in range(n):
        G[i] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G[i].add(j)
                G[j].add(i)
    return G

# Función para generar un grafo Barabási-Albert G(n, m)
def barabasi_albert_graph(n, m):
    G = {}
    for i in range(m):
        G[i] = set()
    for i in range(m):
        for j in range(i + 1, m):
            G[i].add(j)
            G[j].add(i)
    
    node_list = list(range(m)) * m
    
    for i in range(m, n):
        G[i] = set()
        targets = set()
        while len(targets) < m:
            node = random.choice(node_list)
            if node not in targets and node != i:
                targets.add(node)
        for t in targets:
            G[i].add(t)
            G[t].add(i)
        node_list.extend(targets)
        node_list.extend([i] * m)
    
    return G

# Función para generar un grafo Watts-Strogatz G(n, k, beta)
def watts_strogatz_graph(n, k, beta):
    G = {}
    for i in range(n):
        G[i] = set()
    for i in range(n):
        for j in range(1, k // 2 + 1):
            left = (i - j) % n
            right = (i + j) % n
            G[i].add(left)
            G[i].add(right)
            G[left].add(i)
            G[right].add(i)
    
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if random.random() < beta:
                u = (i + j) % n
                if len(G[i]) > (k // 2):
                    G[i].remove(u)
                    G[u].remove(i)
                    v = random.randint(0, n-1)
                    while v == i or v in G[i]:
                        v = random.randint(0, n-1)
                    G[i].add(v)
                    G[v].add(i)
    return G

# Helper function to calculate the clustering coefficient
def clustering_coefficient(G, node):
    neighbors = G[node]
    if len(neighbors) < 2:
        return 0.0
    possible_links = len(neighbors) * (len(neighbors) - 1) / 2
    actual_links = sum(1 for i in neighbors for j in neighbors if i < j and j in G[i])
    return actual_links / possible_links

# Helper function to calculate the average clustering coefficient
def average_clustering_coefficient(G):
    return np.mean([clustering_coefficient(G, node) for node in G])

# Helper function to find the shortest path length using BFS
def shortest_path_length(G, start):
    queue = [(start, 0)]
    visited = {start}
    total_path_length = 0
    reachable_nodes = 0
    
    while queue:
        current, distance = queue.pop(0)
        total_path_length += distance
        reachable_nodes += 1
        for neighbor in G[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return total_path_length, reachable_nodes

# Helper function to calculate the average shortest path length
def average_shortest_path_length(G):
    total_path_length = 0
    total_reachable_nodes = 0
    
    for node in G:
        path_length, reachable_nodes = shortest_path_length(G, node)
        total_path_length += path_length
        total_reachable_nodes += reachable_nodes
    
    if total_reachable_nodes == 0:
        return float('inf')
    
    return total_path_length / total_reachable_nodes

# Analyze Erdős-Rényi model and collect results in a DataFrame
def analyze_erdos_renyi():
    n = 100
    results = []
    for p in np.arange(0.1, 1.1, 0.1):
        G = erdos_renyi_graph(n, p)
        ccp = average_clustering_coefficient(G)
        cpmc = average_shortest_path_length(G)
        results.append({
            "p": round(p, 2),
            "Average Clustering Coefficient": ccp,
            "Average Shortest Path Length": cpmc
        })
    return pd.DataFrame(results)

# Analyze Barabási-Albert model and collect results in a DataFrame
def analyze_barabasi_albert():
    n = 100
    results = []
    for m in range(1, 11):
        G = barabasi_albert_graph(n, m)
        ccp = average_clustering_coefficient(G)
        cpmc = average_shortest_path_length(G)
        results.append({
            "m": m,
            "Average Clustering Coefficient": ccp,
            "Average Shortest Path Length": cpmc
        })
    return pd.DataFrame(results)

# Analyze Watts-Strogatz model and collect results in a DataFrame
def analyze_watts_strogatz():
    n = 100
    k = 4
    results = []
    for beta in np.arange(0, 1.1, 0.1):
        G = watts_strogatz_graph(n, k, beta)
        ccp = average_clustering_coefficient(G)
        cpmc = average_shortest_path_length(G)
        results.append({
            "beta": round(beta, 1),
            "Average Clustering Coefficient": ccp,
            "Average Shortest Path Length": cpmc
        })
    return pd.DataFrame(results)

# Run analyses and show results
df_er = analyze_erdos_renyi()
df_ba = analyze_barabasi_albert()
df_ws = analyze_watts_strogatz()

print("Erdős-Rényi Model Results")
print(df_er)
print("\nBarabási-Albert Model Results")
print(df_ba)
print("\nWatts-Strogatz Model Results")
print(df_ws)

