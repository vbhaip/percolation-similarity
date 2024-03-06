#vbhaip
from random import random
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

graphs = ["karate", "clique", "core_periphery", "connected_cliques", "clique_with_outsider", "clique_missing_edge", "chain", "florentine"]

def core_periphery(sizes, probs, seed=0):
	return nx.stochastic_block_model(sizes, probs, seed=seed)

def clique(size):
	return nx.caveman_graph(1, size)

def connected_cliques(size, number_of_cliques):
	return nx.connected_caveman_graph(number_of_cliques, size)

def clique_with_outsider(size):
	graph = nx.caveman_graph(1, size)
	# graph.add_node(size)
	graph.add_edge(size, size-1)
	return graph

def clique_missing_edge(size):
	graph = nx.caveman_graph(1, size)
	graph.remove_edge(size-1, size-2)
	return graph

def karate():
	return nx.karate_club_graph()

def chain(size):
	return nx.path_graph(size)

def florentine():
	return nx.florentine_families_graph()

def visualize(graph):
	nx.draw_kamada_kawai(graph)
	plt.show()
