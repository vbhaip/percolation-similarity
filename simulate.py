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
import time

from helpers import percolation_similarity_integration, percolation_similarity_threshold, get_rank_matrix, zero_out_lower_triangular, find_differences, matrix_map, get_k_largest_idx
import graph_generation as test_graphs

available_metrics = ["jaccard", "simrank", "resistance", "ps_integration", "ps_threshold"]

metric_to_format = {"jaccard": "Jaccard Index", "simrank": "SimRank", "resistance": "Resistance distance", 
	"ps_integration": "PS by Integration", "ps_threshold": "PS by Threshold"}

SAMPLE_SIZE = 100
P_INTERVAL = 0.01


karate_community_indices_1 = [0,1,2,3,4,5,6,7,8,10,11,12,13,16,17,19,21]
karate_community_indices_2 = [9,14,15,18,20,22,23,24,25,26,27,28,29,30,31,32,33]


karate_index_map = {}

count = 0
for i in karate_community_indices_1:
	karate_index_map[count] = i
	count += 1

for i in karate_community_indices_2:
	karate_index_map[count] = i
	count += 1


# returns matrix of similarities for the given metric. all entries between 0 to 1, lower triangle part will be all zeros, and the diagonal is always 1
def similarity(graph, metric):
	nodes = list(graph)
	total_nodes = len(nodes)
	similarities = np.zeros((total_nodes, total_nodes))

	mask = np.zeros_like(similarities)
	mask[np.tril_indices_from(mask, k=0)] = True

	if metric == "ps_threshold":
		similarities = percolation_similarity_threshold(graph, SAMPLE_SIZE, p_interval=P_INTERVAL)
	elif metric == "ps_integration":
		similarities = percolation_similarity_integration(graph, SAMPLE_SIZE, p_interval=P_INTERVAL)
	elif metric == "resistance":
		distances = np.zeros((total_nodes, total_nodes))
		for i, node_i in enumerate(list(graph)):
			for j, node_j in enumerate(list(graph)):
				if i != j:
					distances[i][j] = nx.resistance_distance(graph, nodeA=node_i, nodeB=node_j)
				else:
					# two nodes that are the same must have distance of 0
					distances[i][i] = 0

		# normalize so it's 0 to 1
		highest_distance = np.max(distances)
		similarities = 1-distances/highest_distance

		similarities = zero_out_lower_triangular(similarities)

	else:
		for i, node_i in enumerate(list(graph)):
			for j, node_j in enumerate(list(graph)):
				if metric == "jaccard":

					similarity = nx.jaccard_coefficient(graph, [(node_i,node_j)])
					# print(list(similarity))
					# print(list(similarity)[0][2])
					similarities[i][j] = list(similarity)[0][2]
				if metric == "simrank":
					similarity = nx.simrank_similarity(graph, source=node_i, target=node_j)
					# print(list(similarity)[0][2])
					similarities[i][j] = similarity


	return similarities, mask

	# print(get_k_largest_idx(similarities, 5))
	# ax = sns.heatmap(similarities, linewidth=0.5, cmap="YlGn", mask=mask)
	# plt.title(metric)
	# plt.show()


def simulate_one_plot(graph, metrics, output_dir):
	metric_times = {}
	metric_to_similarity_rank = {}

	fig, axes = plt.subplots(2, len(metrics),figsize=(5*len(metrics), 5*2))
	fig.suptitle(f"{graph.name}", fontsize=20)

	for ind, metric in enumerate(metrics):
		start = time.time()
		similarities, mask = similarity(graph, metric)
		rank_matrix = get_rank_matrix(similarities)

		metric_times[metric] = time.time() - start

		cbar = (ind == len(metrics)-1)

		ticklabels = 'auto'
		if graph.name == "Zachary's Karate Club":
			ticklabels = karate_community_indices_1+karate_community_indices_2
			similarities = matrix_map(similarities, karate_index_map)
			rank_matrix = matrix_map(rank_matrix, karate_index_map)

		ax = sns.heatmap(similarities, linewidth=0.5, cmap="YlGn", mask=mask, vmin=0, vmax=1, xticklabels=ticklabels, yticklabels=ticklabels, ax=axes[0][ind], cbar=cbar, cbar_kws={'label': 'Similarity Metric Value'})
		ax.set_title(f"{metric_to_format[metric]}", fontsize=15)
		ax.tick_params(axis='both', which='major', labelsize=6)


		ax = sns.heatmap(rank_matrix, linewidth=0.5, cmap = "RdYlGn", mask=mask, vmin=0, vmax=(len(rank_matrix)-1)*len(rank_matrix)/2-1, xticklabels=ticklabels, yticklabels=ticklabels, ax=axes[1][ind], cbar=cbar, cbar_kws={'label': 'Relative Ranking'})
		ax.tick_params(axis='both', which='major', labelsize=6)


		metric_to_similarity_rank[metric] = rank_matrix

	plt.savefig(os.path.join(output_dir, f"all-plots"))
	plt.clf()

	for i in range(0, len(metrics)-1):
		for j in range(i+1, len(metrics)):
			m1 = metrics[i]
			m2 = metrics[j]

			diff = find_differences(metric_to_similarity_rank[m1], metric_to_similarity_rank[m2])

			with open(os.path.join(output_dir, "rank_differences.txt"), "a") as f:
				f.write(f"{m1}, {m2}\n")
				f.write("_________\n")
				for (row, col, r1, r2) in diff:
					f.write(f"{row}, {col}: {r1}, {r2}\n")
				f.write("\n\n")



	sorted_metrics = sorted(metrics, key=lambda x: metric_times[x])

	with open(os.path.join(output_dir, "metric_data.txt"), "w") as f:
		for metric in sorted_metrics:
			f.write(f"{metric}:\t {metric_times[metric]}\n")



def simulate(graph, metrics, output_dir):
	metric_times = {}
	metric_to_similarity_rank = {}
	for metric in metrics:
		start = time.time()
		similarities, mask = similarity(graph, metric)
		rank_matrix = get_rank_matrix(similarities)

		metric_times[metric] = time.time() - start

		ticklabels = 'auto'
		if graph.name == "Zachary's Karate Club":
			ticklabels = karate_community_indices_1+karate_community_indices_2
			similarities = matrix_map(similarities, karate_index_map)
			rank_matrix = matrix_map(rank_matrix, karate_index_map)

		ax = sns.heatmap(similarities, linewidth=0.5, cmap="YlGn", mask=mask, vmin=0, vmax=1, xticklabels=ticklabels, yticklabels=ticklabels, cbar_kws={'label': 'Similarity Metric Value'})
		plt.title(f"{metric_to_format[metric]} for {graph.name} Graph")
		plt.savefig(os.path.join(output_dir, f"{metric}-heatmap"))
		plt.clf()


		
		ax = sns.heatmap(rank_matrix, linewidth=0.5, cmap = "RdYlGn", mask=mask, vmin=0, vmax=(len(rank_matrix)-1)*len(rank_matrix)/2-1, xticklabels=ticklabels, yticklabels=ticklabels, cbar_kws={'label': 'Relative Ranking'})
		plt.title(f"{metric_to_format[metric]} for {graph.name} Graph")
		plt.savefig(os.path.join(output_dir, f"{metric}-rank-heatmap"))
		plt.clf()

		metric_to_similarity_rank[metric] = rank_matrix


	print(metrics)
	for i in range(0, len(metrics)-1):
		for j in range(i+1, len(metrics)):
			m1 = metrics[i]
			m2 = metrics[j]

			diff = find_differences(metric_to_similarity_rank[m1], metric_to_similarity_rank[m2])

			with open(os.path.join(output_dir, "rank_differences.txt"), "a") as f:
				f.write(f"{m1}, {m2}\n")
				f.write("_________\n")
				for (row, col, r1, r2) in diff:
					f.write(f"{row}, {col}: {r1}, {r2}\n")
				f.write("\n\n")



	sorted_metrics = sorted(metrics, key=lambda x: metric_times[x])

	with open(os.path.join(output_dir, "metric_data.txt"), "w") as f:
		for metric in sorted_metrics:
			f.write(f"{metric}:\t {metric_times[metric]}\n")


def __main__():
	parser = argparse.ArgumentParser()

	now_dir = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

	parser.add_argument("-d", "--dir", default=now_dir)

	parser.add_argument("-m", "--metrics", dest="metrics")
	parser.add_argument("-g", "--graph", dest="graph", choices=test_graphs.graphs)

	parser.add_argument("--clique_size", type=int, default=0)
	parser.add_argument("--num_cliques", type=int, default=0)

	parser.add_argument("--core_periphery_sizes", "--cp_sizes", dest="cp_sizes")
	parser.add_argument("--core_periphery_probs", "--cp_probs", dest="cp_probs")

	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--one_plot", action='store_true')

	args = parser.parse_args()

	if args.metrics == "all":
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
		assert len(cp_sizes) == len(cp_probs) and len(cp_sizes) == len(cp_probs[0])
		graph = test_graphs.core_periphery(cp_sizes, cp_probs, seed=args.seed)
		graph.name = f"Core Periphery"
	if args.graph == "clique_with_outsider":
		assert args.clique_size > 0
		graph = test_graphs.clique_with_outsider(size=args.clique_size)
		graph.name = f"{args.clique_size}-Clique with Outsider"

	if args.graph == "chain":
		assert args.clique_size > 0
		graph = test_graphs.chain(args.clique_size)
		graph.name = f"{args.clique_size} Chain"

	if args.graph == "florentine":
		graph = test_graphs.florentine()
		graph.name = "Florentine Families"

	if args.graph == "clique_missing_edge":
		assert args.clique_size > 0
		graph = test_graphs.clique_missing_edge(args.clique_size)
		graph.name = f"{args.clique_size}-Clique Missing One Edge"

	output_dir = os.path.join("./output", now_dir)

	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "command.txt"), "w") as f:
		f.write(f"{sys.argv}")

	if args.one_plot:
		simulate_one_plot(graph, metrics, output_dir)
	else:
		simulate(graph, metrics, output_dir)


__main__()
