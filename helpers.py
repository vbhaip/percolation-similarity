#vbhaip
from random import random
import networkx as nx
import numpy as np
from scipy.stats import rankdata

from matplotlib import pyplot as plt



def zero_out_lower_triangular(matrix):
	return matrix - np.tril(matrix, -1)

def matrix_map(matrix, index_map):
	new_matrix = np.zeros_like(matrix)

	for row in range(0, len(matrix)):
		for col in range(0, len(matrix[0])):
			row_mapped = index_map[row]
			col_mapped = index_map[col]

			# use max bc we have upper triangular matrix
			new_matrix[row][col] = max(matrix[row_mapped][col_mapped], matrix[col_mapped][row_mapped])

	return zero_out_lower_triangular(new_matrix)

# given a 2d-matrix, returns 2d-matrix with rank of items
def get_rank_matrix(matrix):
	# set values below diagonal to be -1, so it doesn't mess up other entries which are 0 in the ranking
	matrix = matrix - np.tril(matrix+1, -1)

	# rank_matrix = matrix.ravel().argsort().argsort().reshape(matrix.shape)
	rank_matrix = (rankdata(matrix.ravel(),method='min')-1).reshape(matrix.shape)

	#values under diagonal will be ranked the lowest, so we want to correct for that
	n = len(rank_matrix)
	normalized_rank_matrix = rank_matrix - (n-1)*n/2

	return normalized_rank_matrix

# given a 2d-matrix, returns a list of length k of the indices with the highest values and what the values are
# ex. get_k_largest_idx([[100, 102], [103, 101]], 2) --> [(1, 0, 103), (0, 1, 102)]
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



# edge_removal_rate is measure of whether we should keep edge \in [0, 1]
# returns 0-1 matrix of nodes that are still connected
def perturbed_graph_connectivity(graph, edge_removal_rate):
	nodes = list(graph)
	total_nodes = len(nodes)


	if edge_removal_rate < 0 or edge_removal_rate > 1:
		raise Exception(f"edge_removal_rate={edge_removal_rate} is not within [0, 1]")

	perturbed_graph = graph.copy()

	for edge in graph.edges():
		if random() > edge_removal_rate:
			perturbed_graph.remove_edge(edge[0], edge[1])


	connected = np.zeros((total_nodes, total_nodes))

	for i, source_node in enumerate(list(graph)):
		for j, target_node in enumerate(list(graph)):
			if i > j:
				continue
			if nx.has_path(perturbed_graph, source_node, target_node):
				connected[i][j] = 1
			else:
				connected[i][j] = 0

	return connected


# gets the average over n trials of perturbed_graph_connectivity
def avg_perturbed_graph_connectivity(graph, edge_removal_rate, n):
	samples = []

	for i in range(n):
		samples.append(perturbed_graph_connectivity(graph, edge_removal_rate))

	return np.array(samples).mean(axis=0)


# n = sample_size
# numerical integral over p
def percolation_similarity_integration(graph, sample_size, p_interval=0.1):
	nodes = list(graph)
	total_nodes = len(nodes)

	percolation_similarity = np.zeros((total_nodes, total_nodes))

	p = p_interval
	while p < 1:
		percolation_similarity += p_interval*avg_perturbed_graph_connectivity(graph, p, sample_size)
		p += p_interval

	return percolation_similarity


#returns 3d matrix of size (1/p_interval, len(nodes, len(nodes)))
def percolation_similarity_samples(graph, sample_size, p_interval):
	nodes = list(graph)
	total_nodes = len(nodes)

	samples = []

	p = p_interval
	while p < 1:
		samples.append(avg_perturbed_graph_connectivity(graph, p, sample_size))
		p += p_interval 

	return np.array(samples)



# return p that is closest to get half nodes connected, using a linear search
def percolation_similarity_threshold(graph, n, p_interval=0.1):
	samples = percolation_similarity_samples(graph, n, p_interval)

	# maps [0, 0.5, 1] -> [-0.5, 0, 0.5] -> [0.5, 0, 0.5] -> [0, 1, 0]
	# highest value is closets to 0
	normalized = 1-2*abs(samples - 0.5)
	indices = np.argmax(normalized, axis=0)


	p_closest_to_half = indices*p_interval


	total_nodes = len(list(graph))

	# this sets the p value for nodes that aren't connected at all to have p=1 (still unconnected) by default
	for i, source_node in enumerate(list(graph)):
		for j, target_node in enumerate(list(graph)):
			if i > j:
				continue
			if not nx.has_path(graph, source_node, target_node):
				p_closest_to_half[i][j] = 1


	# similarity measure, if connected as p=0, then very strong connectivity so strong similarity
	return 1-p_closest_to_half


# display p curve
def generate_p_curve(graph, n, indices, p_interval=0.1):
	samples = percolation_similarity_samples(graph, n, p_interval)

	for idx1, idx2 in indices:
		connectivity_along_p = samples[:, idx1, idx2]
		plt.plot([p_interval*k for k in range(1, len(connectivity_along_p)+1)], connectivity_along_p, '-o')
		plt.title(f"Indices {idx1} and {idx2}")
		plt.show()


#find the top k biggest differences in ranked data, and return results
def find_differences(rank_matrix_1, rank_matrix_2, k = 10):
	rm1 = rank_matrix_1
	rm2 = rank_matrix_2

	diff = abs(rm1 - rm2)

	indices = get_k_largest_idx(diff, k)

	output = []

	for (row, col, val) in indices:
		output.append((row, col, rm1[row][col], rm2[row][col]))

	return output

