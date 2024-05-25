import networkx as nx
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Función para calcular la media y desviación estándar
def calculate_metrics(graphs):
    ccp_values = [nx.average_clustering(G) for G in graphs]
    cpmc_values = []
    for G in graphs:
        try:
            lengths = nx.average_shortest_path_length(G)
            cpmc_values.append(lengths)
        except nx.NetworkXError:
            cpmc_values.append(float('inf'))
    
    ccp_mean = np.mean(ccp_values)
    cpmc_mean = np.mean(cpmc_values)
    ccp_std = np.std(ccp_values)
    cpmc_std = np.std(cpmc_values)
    
    return ccp_mean, cpmc_mean, ccp_std, cpmc_std

# Simular Erdős-Rényi
def simulate_erdos_renyi(n, p, trials):
    return [nx.erdos_renyi_graph(n, p) for _ in range(trials)]

# Simular Barabási-Albert
def simulate_barabasi_albert(n, m, trials):
    return [nx.barabasi_albert_graph(n, m) for _ in range(trials)]

# Simular Watts-Strogatz
def simulate_watts_strogatz(n, k, p, trials):
    return [nx.watts_strogatz_graph(n, k, p) for _ in range(trials)]

# Parámetros
n = 100  # Nodos
p = 0.1  # Probabilidad de conexión en Erdős-Rényi
m = 2    # Número de enlaces a añadir en cada paso en Barabási-Albert
k = 4    # Número de vecinos más cercanos en Watts-Strogatz
beta = 0.1  # Probabilidad de reconexión en Watts-Strogatz
trials = 100  # Número de simulaciones para promediar

# Simulaciones
ER_graphs = simulate_erdos_renyi(n, p, trials)
BA_graphs = simulate_barabasi_albert(n, m, trials)
WS_graphs = simulate_watts_strogatz(n, k, beta, trials)

# Calcular métricas
ER_ccp_mean, ER_cpmc_mean, ER_ccp_std, ER_cpmc_std = calculate_metrics(ER_graphs)
BA_ccp_mean, BA_cpmc_mean, BA_ccp_std, BA_cpmc_std = calculate_metrics(BA_graphs)
WS_ccp_mean, WS_cpmc_mean, WS_ccp_std, WS_cpmc_std = calculate_metrics(WS_graphs)

# Imprimir resultados
print("Erdős-Rényi Model: CCP mean = {:.4f}, CPMC mean = {:.4f}".format(ER_ccp_mean, ER_cpmc_mean))
print("Barabási-Albert Model: CCP mean = {:.4f}, CPMC mean = {:.4f}".format(BA_ccp_mean, BA_cpmc_mean))
print("Watts-Strogatz Model: CCP mean = {:.4f}, CPMC mean = {:.4f}".format(WS_ccp_mean, WS_cpmc_mean))

# Realizar prueba t de Student entre ER y WS para CCP
t_stat, p_val = ttest_ind(
    [nx.average_clustering(G) for G in ER_graphs],
    [nx.average_clustering(G) for G in WS_graphs],
    equal_var=False
)

print("\nT-test between Erdős-Rényi and Watts-Strogatz CCP:")
print("T-statistic = {:.4f}, P-value = {:.4f}".format(t_stat, p_val))

# Gráficos para visualizar las distribuciones de CCP
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.hist([nx.average_clustering(G) for G in ER_graphs], bins=10, alpha=0.7, label="ER")
plt.xlabel('CCP')
plt.ylabel('Frequency')
plt.title('Erdős-Rényi CCP Distribution')

plt.subplot(1, 3, 2)
plt.hist([nx.average_clustering(G) for G in BA_graphs], bins=10, alpha=0.7, label="BA")
plt.xlabel('CCP')
plt.title('Barabási-Albert CCP Distribution')

plt.subplot(1, 3, 3)
plt.hist([nx.average_clustering(G) for G in WS_graphs], bins=10, alpha=0.7, label="WS")
plt.xlabel('CCP')
plt.title('Watts-Strogatz CCP Distribution')

plt.tight_layout()
plt.show()

