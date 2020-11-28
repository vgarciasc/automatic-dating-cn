import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import json

from os import listdir
from os.path import join
from collections import Counter
from csv import QUOTE_ALL

STOP_WORD_GROUPS = ["PRP", "VIRG", "SENT", "PR", "CONJ", "DET", "QUOTE", "PRP+DET", "CARD"]
STOP_WORD_LEMMAS = ["<unknown>"]

def add_node(graph, word):
	if not graph.has_node(word):
		graph.add_node(word)

def add_edge(graph, last_word, word):
	if last_word is not None:
		if graph.has_edge(last_word, word):
			graph[last_word][word]['weight'] += 1
		else:
			graph.add_edge(last_word, word, weight=1)

def read_colonia_file(filename):
	file = open(filename, 'r', encoding="utf-8") 

	G = nx.DiGraph()
	word_counter = Counter()
	last_word = None

	for line in file.readlines() : 
		# Text lines have the format 'word \t syntactic group \t lemma'
		# Other lines indicate sentences, paragraphs, etc. These are ignored
		parsed = line.strip().split("\t")

		if len(parsed) == 3 \
			and parsed[1] not in STOP_WORD_GROUPS \
			and parsed[2] not in STOP_WORD_LEMMAS:

			word = parsed[2].lower()
			word_counter[word] += 1

			add_node(G, word)
			add_edge(G, last_word, word)
			
			last_word = word

	return G, word_counter
	
if __name__ == '__main__':
	metadata_df = pd.read_csv("colonia_metadata.csv")
	path = "data/txt_colonia/"

	should_write_graph = False
	should_draw_graph = False
	should_collect_metrics = False

	data = {
		"number_of_nodes": [],
		"number_of_edges": [],
		"density": [],
		"assortativity_coefficient": [],
		"mean_degree": [],
		"mean_degree_centrality": [],
		"max_degree_centrality": [],
		"mean_clustering": [],
		"max_clustering": [],
		"average_shortest_path_length": []
	}

	for index, row in metadata_df.iterrows():
		filename = join(path, row['filename'])
		G, word_counter = read_colonia_file(filename)

		print("--------")
		print("filename", filename)

		if should_write_graph:
			graph_filename = "data/gml_colonia/" + row['filename'].split(".")[0] + ".graphml"
			nx.write_graphml(G, graph_filename)

		if should_collect_metrics:
			data["number_of_nodes"].append(G.number_of_nodes())
			data["number_of_edges"].append(G.number_of_edges())
			data["density"].append(nx.density(G))
			data["assortativity_coefficient"].append(nx.degree_assortativity_coefficient(G, weight='weight'))
			data["mean_degree"].append(np.mean([val for key, val in G.degree(weight='weight')]))
			degree_centrality = [val for key, val in nx.degree_centrality(G).items()]
			data["mean_degree_centrality"].append(np.mean(degree_centrality))
			data["max_degree_centrality"].append(np.max(degree_centrality))
			# betweenness_centrality = [val for key, val in nx.betweenness_centrality(G).items()]
			# data["mean_betweenness_centrality"].append(np.mean(betweenness_centrality))
			# data["max_betweenness_centrality"].append(np.max(betweenness_centrality))
			clustering = [val for key, val in nx.clustering(G, weight='weight').items()]
			data["mean_clustering"].append(np.mean(clustering))
			data["max_clustering"].append(np.max(clustering))
			data["average_shortest_path_length"].append(nx.average_shortest_path_length(G, weight='weight'))
			# data["average_betwenness"].append(np.mean([val for key, val in nx.betweenness_centrality(G, weight='weight').items()]))
		
		if should_draw_graph:
			nx.draw(G, with_labels=True)
			plt.show()

	if should_collect_metrics:
		for key, val in data.items():
			metadata_df[key] = val
		metadata_df.to_csv('network_metrics.csv', sep=';', quoting=QUOTE_ALL)