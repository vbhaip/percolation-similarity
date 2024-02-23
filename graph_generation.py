from random import random
import networkx as nx
import numpy as np


graphs = ["karate", "clique", "core_periphery", "connected_cliques"]

def core_periphery(sizes, probs, seed=0):
	return nx.stochastic_block_model(sizes, probs, seed=seed)

def clique(size):
	return nx.caveman_graph(1, size)

def connected_cliques(size, number_of_cliques):
	return nx.connected_caveman_graph(number_of_cliques, size)

def karate():
	return nx.karate_club_graph()