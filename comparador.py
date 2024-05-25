import random

# Modelo Erdős-Rényi
def grafo_erdos_renyi(n, p):
    G = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G[i].add(j)
                G[j].add(i)
    return G

# Distribución de grados
def distribucion_grados(G):
    grado_nodos = {}
    for i in G:
        grado = len(G[i])
        if grado in grado_nodos:
            grado_nodos[grado] += 1
        else:
            grado_nodos[grado] = 1
    return grado_nodos

# BFS para camino más corto
def bfs_camino_corto(G, inicio):
    distancias = {vertice: float('infinity') for vertice in G}
    distancias[inicio] = 0
    cola = [inicio]
    while cola:
        u = cola.pop(0)
        for v in G[u]:
            if distancias[v] == float('infinity'):
                distancias[v] = distancias[u] + 1
                cola.append(v)
    return distancias

# Eficiencia de la red global
def eficiencia_red_global(G):
    n = len(G)
    suma_eficiencias = 0
    for u in G:
        for v in G:
            if u != v:
                path_length = bfs_camino_corto(G, u).get(v, float('infinity'))
                if path_length < float('infinity'):
                    suma_eficiencias += 1 / path_length
    return suma_eficiencias / (n * (n - 1))

# Robustez de la red
def robustez_red(G, estrategia='grado'):
    import copy
    G_copia = copy.deepcopy(G)
    nodos = list(G_copia.keys())
    if estrategia == 'grado':
        nodos.sort(key=lambda x: len(G_copia[x]), reverse=True)
    elif estrategia == 'aleatorio':
        random.shuffle(nodos)
    
    robustez = []
    for nodo in nodos:
        G_copia.pop(nodo)
        for vecinos in G_copia.values():
            vecinos.discard(nodo)
        robustez.append(eficiencia_red_global(G_copia))
    return robustez

# Ejemplo de uso
G = grafo_erdos_renyi(10, 0.5)
print("Distribución de Grados:", distribucion_grados(G))
print("Robustez de la Red (eliminación por grado):", robustez_red(G, 'grado'))

