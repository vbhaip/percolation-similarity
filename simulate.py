# 
from random import random
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import argparse
import os
import sys
from ast import literal_eval
from datetime import datetime

from helpers import percolation_similarity_integration, percolation_similarity_threshold
import graph_generation as test_graphs

available_metrics = ["jaccard", "simrank", "resistance", "ps_integration", "ps_threshold"]

metric_to_format = {"jaccard": "Jaccard Index", "simrank": "SimRank", "resistance": "Resistance distance", 
	"ps_integration": "Percolation Similarity by Integration", "ps_threshold": "Percolation Similarity by Threshold"}

SAMPLE_SIZE = 100
P_INTERVAL = 0.01

def similarity(graph, metric):
	nodes = list(graph)
	total_nodes = len(nodes)
	similarities = np.zeros((total_nodes, total_nodes))

	mask = np.zeros_like(similarities)
	mask[np.tril_indices_from(mask, k=0)] = True

	if metric == "ps_threshold":
		similarities = percolation_similarity_threshold(graph, SAMPLE_SIZE, p_interval=P_INTERVAL)
	elif metric == "ps_integration":
		similarities = percolation_similarity_threshold(graph, SAMPLE_SIZE, p_interval=P_INTERVAL)
	else:
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

	return similarities, mask

	# print(get_k_largest_idx(similarities, 5))
	# ax = sns.heatmap(similarities, linewidth=0.5, cmap="YlGn", mask=mask)
	# plt.title(metric)
	# plt.show()


def simulate(graph, metrics, output_dir):
	for metric in metrics:
		similarities, mask = similarity(graph, metric)

		ax = sns.heatmap(similarities, linewidth=0.5, cmap="YlGn", mask=mask, vmin=0, vmax=1)
		plt.title(f"{metric_to_format[metric]} for {graph.name} Graph")
		plt.savefig(os.path.join(output_dir, f"{metric}-heatmap"))
		plt.clf()


def __main__():
	parser = argparse.ArgumentParser()

	now_dir = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

	parser.add_argument("-d", "--dir", default=now_dir)

	parser.add_argument("-m", "--metrics", dest="metrics")
	parser.add_argument("-g", "--graph", dest="graph", choices=test_graphs.graphs)

	parser.add_argument("--clique_size", type=int, default=0)
	parser.add_argument("--num_cliques", type=int, default=0)

	parser.add_argument("--core_periphery_sizes", dest="cp_sizes")
	parser.add_argument("--core_periphery_probs", dest="cp_probs")

	parser.add_argument("--seed", type=int, default=0)

	args = parser.parse_args()

	if args.metrics = "all":
		metrics = available_metrics
	else:
		metrics = args.metrics.split(",")
		for metric in metrics:
			if metric not in available_metrics:
				raise Exception(f"{metric} not valid: choose something from {available_metrics}")

	graph = None
	if args.graph == "karate":
		graph = test_graphs.karate()
		graph.name = "Zachary's Karate Club"
	if args.graph == "clique":
		assert args.clique_size > 0
		graph = test_graphs.clique(args.clique_size)
		graph.name = f"{args.clique_size}-Clique"
	if args.graph == "connected_cliques":
		assert args.clique_size > 0 and args.num_cliques > 0
		graph = test_graphs.connected_cliques(args.clique_size, args.num_cliques)
		graph.name = f"{args.num_cliques} Connected {args.clique_size}-Cliques"
	if args.graph == "core_periphery":
		cp_sizes = literal_eval(args.cp_sizes)
		cp_probs = literal_eval(args.cp_probs)
		assert len(cp_sizes) == len(cp_probs) and len(cp_sizes[0]) == len(cp_probs[0])
		graph = test_graph.core_periphery(cp_sizes, cp_probs, seed=args.seed)
		graph.name = f"Core Periphery"

	output_dir = os.path.join("./output", now_dir)

	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "command.txt"), "w") as f:
		f.write(f"{sys.argv}")

	simulate(graph, metrics, output_dir)


__main__()


# karate = nx.karate_club_graph()
# # similarity(karate, "simrank")

# # print(perturbed_graph_connectivity(karate, 0.2))

# # ------------
# # percolation_similarity_0_2 = (percolation_similarity(karate, 100, p_interval=0.05))
# # print(get_k_largest_idx(percolation_similarity_0_2, 5))


# # ax = sns.heatmap(percolation_similarity_0_2, linewidth=0.5, cmap="YlGn")
# # plt.show()
# # -----------

# # sizes = [10, 10]
# # probs = [[0.8, 0.5], [0.5, 0.1]]
# # core_periphery = nx.stochastic_block_model(sizes, probs, seed=0)

# # PS_core_periphery = (percolation_similarity_integration(core_periphery, 20, p_interval=0.1))
# # # print(get_k_largest_idx(PS_core_periphery, 5))


# # PS_linear_search = (percolation_similarity_linear_search(core_periphery, 100, p_interval=0.01))


# # mask = np.zeros_like(PS_linear_search)
# # mask[np.tril_indices_from(mask, k=1)] = True
# # ax = sns.heatmap(PS_linear_search, linewidth=0.5, cmap="YlGn", mask=mask)
# # plt.show()

# # similarity(core_periphery, "jaccard")

# # -------

# clique = nx.caveman_graph(1, 2)

# PS_linear_search = (percolation_similarity_threshold(clique, 100, p_interval=0.01))


# mask = np.zeros_like(PS_linear_search)
# mask[np.tril_indices_from(mask, k=1)] = False
# ax = sns.heatmap(PS_linear_search, linewidth=0.5, cmap="YlGn", mask=mask)
# plt.show()
