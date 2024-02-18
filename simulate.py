# 
from random import random
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# N = 30

# # roughly 2 edges per node
# p = N/(N*(N-1)/2)

# graph = nx.fast_gnp_random_graph(N, p)


# similarity_metrics = {
# 	"jacard": nx.jaccard_coefficient(G, [0, 33]),
# 	"simrank": nx.simrank_similarity(G, source=0, target=33),

# }

def get_k_largest_idx(matrix, k):
	rows, cols = np.shape(matrix)

	best_vals = []
	best_indices = []

	for i in range(k):
		best = float('-inf')
		best_row, best_col = -1, -1

		for row in range(0, rows):
			for col in range(row+1, cols):
				if matrix[row][col] > best and (row, col) not in best_indices and matrix[row][col] not in best_vals:
					best = matrix[row][col]
					best_row, best_col = row, col

		best_vals += [best]
		best_indices += [(best_row, best_col)]

	return list(zip([idx[0] for idx in best_indices], [idx[1] for idx in best_indices], best_vals))


# p is measure of whether we should keep edge, p \in [0, 1]
# returns matrix of nodes that are still connected
def perturbed_graph_connectivity(graph, p):
	nodes = list(graph)
	total_nodes = len(nodes)


	if p < 0 or p > 1:
		raise Exception(f"p={p} is not within [0, 1]")

	perturbed_graph = graph.copy()

	for edge in graph.edges():
		if random() > p:
			perturbed_graph.remove_edge(edge[0], edge[1])


	connected = np.zeros((total_nodes, total_nodes))

	for source_node in range(0, total_nodes):
		for target_node in range(source_node+1, total_nodes):
			if nx.has_path(perturbed_graph, source_node, target_node):
				connected[source_node][target_node] = 1
			else:
				connected[source_node][target_node] = 0

	return connected


def avg_perturbed_graph_connectivity(graph, p, n):
	samples = []

	for i in range(n):
		samples.append(perturbed_graph_connectivity(graph, p))

	return np.array(samples).mean(axis=0)


# n = sample_size
# numerical integral over p
def percolation_similarity_integration(graph, n, p_interval=0.1):
	nodes = list(graph)
	total_nodes = len(nodes)

	percolation_similarity = np.zeros((total_nodes, total_nodes))

	p = p_interval
	while p < 1:
		percolation_similarity += p_interval*avg_perturbed_graph_connectivity(graph, p, n)
		p += p_interval

	return percolation_similarity


#returns 3d matrix of size (1/p_interval, len(nodes, len(nodes)))
def percolation_similarity_samples(graph, n, p_interval):
	nodes = list(graph)
	total_nodes = len(nodes)

	samples = []

	p = p_interval
	while p < 1:
		samples.append(avg_perturbed_graph_connectivity(graph, p, n))
		p += p_interval 

	return np.array(samples)



# return p that is closest to get half nodes connected
def percolation_similarity_linear_search(graph, n, p_interval=0.1):
	samples = percolation_similarity_samples(graph, n, p_interval)

	# print(samples.shape)
	# print(samples[:, 0, 1])

	# # plot of "p curve"
	# for idx1, idx2 in [(0, 1), (32, 33)]:
	# 	connectivity_along_p = samples[:, idx1, idx2]
	# 	plt.plot([p_interval*k for k in range(1, len(connectivity_along_p)+1)], connectivity_along_p, '-o')
	# 	plt.title(f"Indices {idx1} and {idx2}")
	# 	plt.show()

	normalized = 1-2*abs(samples - 0.5)

	print(normalized.shape)

	indices = np.argmax(normalized, axis=0)

	print(indices)

	p_closest_to_half = indices*p_interval


	total_nodes = len(list(graph))

	# this sets the p value for nodes that aren't connected at all to have p=1 (still unconnected) by default
	for source_node in range(0, total_nodes):
		for target_node in range(source_node+1, total_nodes):
			if not nx.has_path(graph, source_node, target_node):
				p_closest_to_half[source_node][target_node] = 1


	# similarity measure, if connected as p=0, then very strong connectivity so strong similarity
	return 1-p_closest_to_half



def similarity(graph, metric):
	nodes = list(graph)
	total_nodes = len(nodes)
	similarities = np.zeros((total_nodes, total_nodes))

	mask = np.zeros_like(similarities)
	mask[np.tril_indices_from(mask, k=1)] = True

	for i in range(0, total_nodes):
		for j in range(i, total_nodes):
			if metric == "jaccard":
				similarity = nx.jaccard_coefficient(graph, [(i, j)])
				# print(list(similarity)[0][2])
				similarities[i][j] = list(similarity)[0][2]
			if metric == "simrank":
				similarity = nx.simrank_similarity(graph, source=i, target=j)
				# print(list(similarity)[0][2])
				similarities[i][j] = similarity
			if metric == "resistance":
				if i != j:
					similarity = nx.resistance_distance(graph, nodeA=i, nodeB=j)
					similarities[i][j] = -1*similarity
				else:
					mask[i][i] = True


	print(get_k_largest_idx(similarities, 5))
	ax = sns.heatmap(similarities, linewidth=0.5, cmap="YlGn", mask=mask)
	plt.title(metric)
	plt.show()


karate = nx.karate_club_graph()
# similarity(karate, "simrank")

# print(perturbed_graph_connectivity(karate, 0.2))

# ------------
# percolation_similarity_0_2 = (percolation_similarity(karate, 100, p_interval=0.05))
# print(get_k_largest_idx(percolation_similarity_0_2, 5))


# ax = sns.heatmap(percolation_similarity_0_2, linewidth=0.5, cmap="YlGn")
# plt.show()
# -----------

# sizes = [10, 10]
# probs = [[0.8, 0.5], [0.5, 0.1]]
# core_periphery = nx.stochastic_block_model(sizes, probs, seed=0)

# PS_core_periphery = (percolation_similarity_integration(core_periphery, 20, p_interval=0.1))
# # print(get_k_largest_idx(PS_core_periphery, 5))


# PS_linear_search = (percolation_similarity_linear_search(core_periphery, 100, p_interval=0.01))


# mask = np.zeros_like(PS_linear_search)
# mask[np.tril_indices_from(mask, k=1)] = True
# ax = sns.heatmap(PS_linear_search, linewidth=0.5, cmap="YlGn", mask=mask)
# plt.show()

# similarity(core_periphery, "jaccard")

# -------

clique = nx.caveman_graph(1, 2)

PS_linear_search = (percolation_similarity_linear_search(clique, 100, p_interval=0.01))


mask = np.zeros_like(PS_linear_search)
mask[np.tril_indices_from(mask, k=1)] = False
ax = sns.heatmap(PS_linear_search, linewidth=0.5, cmap="YlGn", mask=mask)
plt.show()

# similarity(karate, "jaccard")






# removal_p = 0.1

# while removal_p < 1:
# 	temp_graph = graph.copy()
# 	for edge in temp_graph.edges():
# 		print(edge)
# 		temp_graph.remove_edge(*edge)


# 	removal_p += 0.1


# class Graph:

# 	# adjacency matrix
# 	# invert with probability p
# 	# is connected
# 	def __init__(self, adjacency):
# 		self.adjacency = adjacency


# class Erdos_Renyi_Graph:

# 	def _zero_or_one(p):
# 		if random.random() < p:
# 			return 1
# 		return 0

# 	# generate graph with probability of edge existing being p
# 	def __init__(self, size, p):
# 		self.p = p

# 		adj