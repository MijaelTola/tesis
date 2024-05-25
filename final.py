import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.stats import ttest_ind

# Asegurar que el directorio para las imágenes existe
if not os.path.exists('images'):
    os.makedirs('images')

# Implementación de los modelos de grafos
def erdos_renyi_graph(n, p):
    G = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G[i].add(j)
                G[j].add(i)
    return G

def barabasi_albert_graph(n, m):
    G = {i: set() for i in range(m)}
    for i in range(m):
        for j in range(i + 1, m):
            G[i].add(j)
            G[j].add(i)
    
    target_nodes = list(range(m))
    total_degree = 2 * m * (m - 1) / 2
    
    for i in range(m, n):
        G[i] = set()
        targets = np.random.choice(target_nodes, m, replace=False)
        
        for target in targets:
            G[i].add(target)
            G[target].add(i)
        
        target_nodes.extend([i] * m)
        target_nodes.extend(targets)
        total_degree += 2 * m
    
    return G

def watts_strogatz_graph(n, k, p):
    G = {i: set() for i in range(n)}
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
            if random.random() < p:
                right = (i + j) % n
                if len(G[i]) > (k // 2 + 1):
                    G[i].remove(right)
                    G[right].remove(i)
                    new_node = random.randint(0, n - 1)
                    while new_node == i or new_node in G[i]:
                        new_node = random.randint(0, n - 1)
                    G[i].add(new_node)
                    G[new_node].add(i)
    return G

# Cálculo de métricas
def clustering_coefficient(G):
    cc_values = []
    for node in G:
        neighbors = list(G[node])
        if len(neighbors) < 2:
            cc_values.append(0)
            continue
        links = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in G[neighbors[i]]:
                    links += 1
        cc_values.append(2 * links / (len(neighbors) * (len(neighbors) - 1)))
    return np.mean(cc_values)

def average_shortest_path_length(G):
    path_lengths = []
    for start in G:
        distances = {node: float('inf') for node in G}
        distances[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in G[node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        path_lengths.extend([dist for dist in distances.values() if dist != float('inf')])
    return np.mean(path_lengths)

# Simulación y cálculo de métricas
n = 100
trials = 100

ccp_er, cpmc_er = [], []
ccp_ba, cpmc_ba = [], []
ccp_ws, cpmc_ws = [], []

for _ in range(trials):
    G_er = erdos_renyi_graph(n, 0.1)
    ccp_er.append(clustering_coefficient(G_er))
    cpmc_er.append(average_shortest_path_length(G_er))
    
    G_ba = barabasi_albert_graph(n, 2)
    ccp_ba.append(clustering_coefficient(G_ba))
    cpmc_ba.append(average_shortest_path_length(G_ba))
    
    G_ws = watts_strogatz_graph(n, 4, 0.1)
    ccp_ws.append(clustering_coefficient(G_ws))
    cpmc_ws.append(average_shortest_path_length(G_ws))

# Gráficos de las distribuciones
def plot_distribution(values, title, file_name):
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
    plt.title(f'Distribución de {title}')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(f'images/{file_name}.png')
    plt.close()

plot_distribution(ccp_er, 'CCP para Erdős-Rényi', 'ccp_er')
plot_distribution(ccp_ba, 'CCP para Barabási-Albert', 'ccp_ba')
plot_distribution(ccp_ws, 'CCP para Watts-Strogatz', 'ccp_ws')

plot_distribution(cpmc_er, 'CPMC para Erdős-Rényi', 'cpmc_er')
plot_distribution(cpmc_ba, 'CPMC para Barabási-Albert', 'cpmc_ba')
plot_distribution(cpmc_ws, 'CPMC para Watts-Strogatz', 'cpmc_ws')

# Pruebas t de Student
t_ccp_er_ba, p_ccp_er_ba = ttest_ind(ccp_er, ccp_ba)
t_ccp_er_ws, p_ccp_er_ws = ttest_ind(ccp_er, ccp_ws)
t_ccp_ba_ws, p_ccp_ba_ws = ttest_ind(ccp_ba, ccp_ws)

t_cpmc_er_ba, p_cpmc_er_ba = ttest_ind(cpmc_er, cpmc_ba)
t_cpmc_er_ws, p_cpmc_er_ws = ttest_ind(cpmc_er, cpmc_ws)
t_cpmc_ba_ws, p_cpmc_ba_ws = ttest_ind(cpmc_ba, cpmc_ws)

# Salida de los resultados
print(f"CCP t-test Erdős-Rényi vs Barabási-Albert: t = {t_ccp_er_ba:.4f}, p = {p_ccp_er_ba:.4g}")
print(f"CCP t-test Erdős-Rényi vs Watts-Strogatz: t = {t_ccp_er_ws:.4f}, p = {p_ccp_er_ws:.4g}")
print(f"CCP t-test Barabási-Albert vs Watts-Strogatz: t = {t_ccp_ba_ws:.4f}, p = {p_ccp_ba_ws:.4g}")

print(f"CPMC t-test Erdős-Rényi vs Barabási-Albert: t = {t_cpmc_er_ba:.4f}, p = {p_cpmc_er_ba:.4g}")
print(f"CPMC t-test Erdős-Rényi vs Watts-Strogatz: t = {t_cpmc_er_ws:.4f}, p = {p_cpmc_er_ws:.4g}")
print(f"CPMC t-test Barabási-Albert vs Watts-Strogatz: t = {t_cpmc_ba_ws:.4f}, p = {p_cpmc_ba_ws:.4g}")

