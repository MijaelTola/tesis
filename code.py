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

def calcular_componentes_conexas(G):
    visitado = set()
    componentes = []

    def bfs_component(G, inicio):
        cola = [inicio]
        componente = set()

        while cola:
            nodo = cola.pop(0)
            if nodo not in visitado:
                visitado.add(nodo)
                componente.add(nodo)
                cola.extend(G[nodo] - visitado)

        return componente

    for nodo in G:
        if nodo not in visitado:
            componente = bfs_component(G, nodo)
            componentes.append(componente)

    return componentes

def robustez_frente_fallos(G, num_eliminaciones):
    nodos = list(G.keys())
    componentes_conexas = []

    for _ in range(num_eliminaciones):
        nodo_a_eliminar = random.choice(nodos)
        nodos.remove(nodo_a_eliminar)
        G.pop(nodo_a_eliminar)
        
        for vecino in list(G.values()):
            vecino.discard(nodo_a_eliminar)

        componentes_conexas.append(len(max(calcular_componentes_conexas(G), key=len)))

    return componentes_conexas

def robustez_frente_ataques(G, num_eliminaciones):
    nodos = list(G.keys())
    nodos_ordenados = sorted(nodos, key=lambda x: len(G[x]), reverse=True)
    componentes_conexas = []

    for i in range(num_eliminaciones):
        nodo_a_eliminar = nodos_ordenados[i]
        nodos.remove(nodo_a_eliminar)
        G.pop(nodo_a_eliminar)
        
        for vecino in list(G.values()):
            vecino.discard(nodo_a_eliminar)

        componentes_conexas.append(len(max(calcular_componentes_conexas(G), key=len)))

    return componentes_conexas

def plot_robustez(componentes_conexas, tipo, modelo, file_name):
    plt.plot(range(len(componentes_conexas)), componentes_conexas)
    plt.xlabel('Número de Eliminaciones')
    plt.ylabel('Tamaño de la Componente Conexa Más Grande')
    plt.title(f'Robustez de la Red Frente a {tipo} - {modelo}')
    plt.savefig(f'images/{file_name}.png')
    plt.close()

# Simulación y cálculo de métricas
n = 100
trials = 100
num_eliminaciones = 20

ccp_er, cpmc_er = [], []
ccp_ba, cpmc_ba = [], []
ccp_ws, cpmc_ws = [], []

robustez_fallos_er, robustez_ataques_er = [], []
robustez_fallos_ba, robustez_ataques_ba = [], []
robustez_fallos_ws, robustez_ataques_ws = [], []

for _ in range(trials):
    G_er = erdos_renyi_graph(n, 0.1)
    ccp_er.append(clustering_coefficient(G_er))
    cpmc_er.append(average_shortest_path_length(G_er))
    robustez_fallos_er.append(robustez_frente_fallos(G_er.copy(), num_eliminaciones))
    robustez_ataques_er.append(robustez_frente_ataques(G_er.copy(), num_eliminaciones))
    
    G_ba = barabasi_albert_graph(n, 2)
    ccp_ba.append(clustering_coefficient(G_ba))
    cpmc_ba.append(average_shortest_path_length(G_ba))
    robustez_fallos_ba.append(robustez_frente_fallos(G_ba.copy(), num_eliminaciones))
    robustez_ataques_ba.append(robustez_frente_ataques(G_ba.copy(), num_eliminaciones))
    
    G_ws = watts_strogatz_graph(n, 4, 0.1)
    ccp_ws.append(clustering_coefficient(G_ws))
    cpmc_ws.append(average_shortest_path_length(G_ws))
    robustez_fallos_ws.append(robustez_frente_fallos(G_ws.copy(), num_eliminaciones))
    robustez_ataques_ws.append(robustez_frente_ataques(G_ws.copy(), num_eliminaciones))

# Promediar resultados de robustez a través de los trials
mean_robustez_fallos_er = np.mean(robustez_fallos_er, axis=0)
mean_robustez_ataques_er = np.mean(robustez_ataques_er, axis=0)
mean_robustez_fallos_ba = np.mean(robustez_fallos_ba, axis=0)
mean_robustez_ataques_ba = np.mean(robustez_ataques_ba, axis=0)
mean_robustez_fallos_ws = np.mean(robustez_fallos_ws, axis=0)
mean_robustez_ataques_ws = np.mean(robustez_ataques_ws, axis=0)

# Gráficos de robustez
plot_robustez(mean_robustez_fallos_er, "Fallos Aleatorios", "Erdős-Rényi", "er_fallos")
plot_robustez(mean_robustez_ataques_er, "Ataques Dirigidos", "Erdős-Rényi", "er_ataques")

plot_robustez(mean_robustez_fallos_ba, "Fallos Aleatorios", "Barabási-Albert", "ba_fallos")
plot_robustez(mean_robustez_ataques_ba, "Ataques Dirigidos", "Barabási-Albert", "ba_ataques")

plot_robustez(mean_robustez_fallos_ws, "Fallos Aleatorios", "Watts-Strogatz", "ws_fallos")
plot_robustez(mean_robustez_ataques_ws, "Ataques Dirigidos", "Watts-Strogatz", "ws_ataques")

# Pruebas t de Student para Robustez
t_fallos_er_ba, p_fallos_er_ba = ttest_ind(mean_robustez_fallos_er, mean_robustez_fallos_ba)
t_fallos_er_ws, p_fallos_er_ws = ttest_ind(mean_robustez_fallos_er, mean_robustez_fallos_ws)
t_fallos_ba_ws, p_fallos_ba_ws = ttest_ind(mean_robustez_fallos_ba, mean_robustez_fallos_ws)

t_ataques_er_ba, p_ataques_er_ba = ttest_ind(mean_robustez_ataques_er, mean_robustez_ataques_ba)
t_ataques_er_ws, p_ataques_er_ws = ttest_ind(mean_robustez_ataques_er, mean_robustez_ataques_ws)
t_ataques_ba_ws, p_ataques_ba_ws = ttest_ind(mean_robustez_ataques_ba, mean_robustez_ataques_ws)

# Pruebas t de Student para clustering coefficient y average shortest path length
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

print(f"Robustez Fallos t-test Erdős-Rényi vs Barabási-Albert: t = {t_fallos_er_ba:.4f}, p = {p_fallos_er_ba:.4g}")
print(f"Robustez Fallos t-test Erdős-Rényi vs Watts-Strogatz: t = {t_fallos_er_ws:.4f}, p = {p_fallos_er_ws:.4g}")
print(f"Robustez Fallos t-test Barabási-Albert vs Watts-Strogatz: t = {t_fallos_ba_ws:.4f}, p = {p_fallos_ba_ws:.4g}")

print(f"Robustez Ataques t-test Erdős-Rényi vs Barabási-Albert: t = {t_ataques_er_ba:.4f}, p = {p_ataques_er_ba:.4g}")
print(f"Robustez Ataques t-test Erdős-Rényi vs Watts-Strogatz: t = {t_ataques_er_ws:.4f}, p = {p_ataques_er_ws:.4g}")
print(f"Robustez Ataques t-test Barabási-Albert vs Watts-Strogatz: t = {t_ataques_ba_ws:.4f}, p = {p_ataques_ba_ws:.4g}")
