import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import sys

from os import listdir
from os.path import join
from collections import Counter
from csv import QUOTE_ALL

import reader as r
import century_similarities_compiler as csc

# Generates a 'baseline_dataset' CSV file, which contains features extracted only
# by simple statistical analysis in the texts (number of words, similarity between most
# frequent words, etc)

def generate_baseline_file(in_data_filename, out_data_filename, data_path, verbose=True):
	data_df = pd.read_csv(in_data_filename, sep=";", quotechar='"')

	data = {
		"lemma_quantity": [],
		"normalized_lemma_quantity": []
	}

	for index, row in data_df.iterrows():
		filename = join(data_path, row['filename'])
		G, word_counter = r.read_colonia_file(filename)

		if verbose:
			print("\t--------")
			print("\t", (index + 1), "/", len(data_df), ": filename", filename)

		data["lemma_quantity"].append(G.number_of_nodes())
		data["normalized_lemma_quantity"].append(G.number_of_nodes() / row['words'])

	# Generates the CSV file
	for key, val in data.items():
		data_df[key] = val
	data_df.to_csv(out_data_filename, sep=';', quoting=QUOTE_ALL)

def generate_network_metrics_file(in_data_filename, out_data_filename, data_path, verbose=True):
	metadata_df = pd.read_csv(in_data_filename, sep=";", quotechar='"')

	data = {
		"number_of_nodes": [],
		"normalized_number_of_nodes": [],
		"number_of_edges": [],
		"normalized_number_of_edges": [],
		"density": [],
		"assortativity_coefficient": [],
		"average_shortest_path_length": [],
		"diameter": [],
		"transitivity": [],
		"mean_degree": [],
		"mean_clustering": [],
		"max_clustering": [],
		"mean_betweenness_centrality": [],
		"max_betweenness_centrality": [],
	}

	if verbose:
		print("generating metrics file...")

	for index, row in metadata_df.iterrows():
		filename = join(data_path, row['filename'])
		G, word_counter = r.read_colonia_file(filename)

		if verbose:
			print("\t--------")
			print("\t", (index + 1), "/", len(metadata_df), ": filename", filename)

		data["number_of_nodes"].append(G.number_of_nodes())
		data["normalized_number_of_nodes"].append(G.number_of_nodes() / row['words'])
		data["number_of_edges"].append(G.number_of_edges())
		data["normalized_number_of_edges"].append(G.number_of_edges() / (row['words'] * row['words'] / 2))
		data["density"].append(nx.density(G))
		data["assortativity_coefficient"].append(nx.degree_assortativity_coefficient(G, weight='weight'))
		data["average_shortest_path_length"].append(nx.average_shortest_path_length(G, weight='weight'))
		data["diameter"].append(nx.diameter(G))
		data["transitivity"].append(nx.transitivity(G))
		data["mean_degree"].append(np.mean([val for key, val in G.degree(weight='weight')]))

		clustering = [val for key, val in nx.clustering(G, weight='weight').items()]
		data["mean_clustering"].append(np.mean(clustering))
		data["max_clustering"].append(np.max(clustering))

		betweenness_centrality = [val for key, val in nx.betweenness_centrality(G).items()]
		data["mean_betweenness_centrality"].append(np.mean(betweenness_centrality))
		data["max_betweenness_centrality"].append(np.max(betweenness_centrality))

	for key, val in data.items():
		if len(val) > 0:
			metadata_df[key] = val
	metadata_df.to_csv(out_data_filename, sep=';', quoting=QUOTE_ALL)

if __name__ == '__main__':
	flag_1 = None
	flag_2 = None

	if len(sys.argv) > 1:
		flag_1 = sys.argv[1]
	if len(sys.argv) > 2:
		flag_2 = sys.argv[2]

	if flag_1 == "--baseline":
		metadata_filename = "colonia_metadata.csv"
		lemmarank_data_filename = "lemmaranks.csv"
		similarity_data_filename = "century_similarities.csv"
		baseline_data_filename = "baseline_data.csv"

		data_path = "data/txt_colonia/"
		rank_len = int(flag_2)

		csc.generate_lemmarank_file(metadata_filename, lemmarank_data_filename, data_path, rank_len, csc.extract_most_freq)
		csc.generate_similarity_file(metadata_filename, lemmarank_data_filename, similarity_data_filename, data_path, \
			[], rank_len, csc.jaccard_similarity)
		generate_baseline_file(similarity_data_filename, baseline_data_filename, data_path)

	elif flag_1 == "--network":
		metadata_filename = "colonia_metadata.csv"
		lemmarank_data_filename = "lemmaranks.csv"
		similarity_data_filename = "century_similarities.csv"
		network_metrics_filename = "network_metrics.csv"

		data_path = "data/txt_colonia/"
		rank_len = int(flag_2)

		csc.generate_lemmarank_file(metadata_filename, lemmarank_data_filename, data_path, rank_len, csc.extract_most_closeness)
		csc.generate_similarity_file(metadata_filename, lemmarank_data_filename, similarity_data_filename, data_path, \
			[], rank_len, csc.jaccard_similarity)
		generate_network_metrics_file(similarity_data_filename, network_metrics_filename, data_path)

	else:
		print("Incorrect usage detected. Use one of the following patterns:")
		print("\t main.py --baseline [rank length]")
		print("\t main.py --network [rank length]")
		sys.exit(0)