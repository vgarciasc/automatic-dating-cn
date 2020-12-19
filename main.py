# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import sys
import datetime

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

def generate_graphs_file(in_data_filename, data_path, graph_path, read_fn=r.read_colonia_file, verbose=True):
	data_df = pd.read_csv(in_data_filename, sep=";", quotechar='"')

	if verbose:
		print("generating graphs file...")

	for index, row in data_df.iterrows():
		filename = join(data_path, row['filename'])
		G, word_counter = read_fn(filename)

		if verbose:
			print("\t--------")
			print("\t", (index + 1), "/", len(data_df), ": filename", filename)

		nx.write_gml(G, join(graph_path, row['filename'].split(".")[0] + ".gml"))

def generate_network_metrics_file(in_data_filename, out_data_filename, data_path, graph_path, verbose=True):
	metadata_df = pd.read_csv(in_data_filename, sep=";", quotechar='"')
	data = {
		"number_of_nodes": [],
		"normalized_number_of_nodes": [],
		# "number_of_edges": [],
		# "normalized_number_of_edges": [],
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
		total_time = datetime.datetime.now()

	for index, row in metadata_df.iterrows():
		filename = join(graph_path, row['filename'].split(".")[0] + ".gml")
		graph = ig.Graph.Read_GML(filename)
		start = datetime.datetime.now()

		if verbose:
			print("\t--------")
			print("\t", (index + 1), "/", len(metadata_df), ": filename", filename)

		data["number_of_nodes"].append(graph.vcount())
		data["normalized_number_of_nodes"].append(graph.vcount() / row['words'])
		# data["number_of_edges"].append(graph.ecount())
		# data["normalized_number_of_edges"].append(graph.ecount() / row['words'])
		data["density"].append(graph.density())
		data["assortativity_coefficient"].append(graph.assortativity_degree(graph.strength()))
		data["average_shortest_path_length"].append(graph.average_path_length())
		data["diameter"].append(graph.diameter())
		data["transitivity"].append(graph.transitivity_undirected())
		data["mean_degree"].append(np.mean([d for d in graph.strength()]))

		clustering = graph.transitivity_local_undirected(weights = graph.es["weight"])
		data["mean_clustering"].append(np.mean(clustering))
		data["max_clustering"].append(np.max(clustering))

		betweenness_centrality = graph.betweenness(weights = graph.es["weight"])
		data["mean_betweenness_centrality"].append(np.mean(betweenness_centrality))
		data["max_betweenness_centrality"].append(np.max(betweenness_centrality))
		
		end = datetime.datetime.now()
		if verbose:
			print("\ttime elapsed:", end - start)

	for key, val in data.items():
		if len(val) > 0:
			metadata_df[key] = val
	metadata_df.to_csv(out_data_filename, sep=';', quoting=QUOTE_ALL)

	if verbose:
		print("total_time:", (datetime.datetime.now()- total_time))

if __name__ == '__main__':
	colonia_metadata_filename = "data/colonia_metadata.csv"
	tychobrahe_metadata_filename = "data/tychobrahe_metadata.csv"
	lemmarank_data_filename = "data/lemmaranks.csv"
	similarity_data_filename = "data/century_similarities.csv"
	network_metrics_filename = "data/network/network_metrics_punctuated.csv"
	baseline_data_filename = "data/baseline/baseline_data.csv"

	graph_path = "data/graphs/"

	flag_1 = None
	flag_2 = None

	if len(sys.argv) > 1:
		flag_1 = sys.argv[1]
	if len(sys.argv) > 2:
		flag_2 = sys.argv[2]

	if flag_1 == "--graphs":
		generate_graphs_file(tychobrahe_metadata_filename, "data/txt_tychobrahe", graph_path, read_fn = r.read_tychobrahe_file)
		generate_graphs_file(colonia_metadata_filename, "data/txt_colonia", graph_path, read_fn = r.read_colonia_file)

	elif flag_1 == "--baseline":
		rank_len = int(flag_2)

		csc.generate_lemmarank_file(colonia_metadata_filename, "data/baseline/lemmaranks.csv", "data/txt_colonia", rank_len, csc.extract_most_freq)
		generate_baseline_file(colonia_metadata_filename, baseline_data_filename, "data/txt_colonia")

	elif flag_1 == "--network":
		rank_len = int(flag_2)

		csc.generate_lemmarank_file(colonia_metadata_filename, "data/network/lemmaranks_punctuated.csv", "data/txt_colonia", rank_len, csc.extract_most_closeness)
		generate_network_metrics_file(colonia_metadata_filename, network_metrics_filename, "data/txt_colonia", graph_path)

	else:
		print("Incorrect usage detected. Use one of the following patterns:")
		print("\t main.py --baseline [rank length]")
		print("\t main.py --network [rank length]")
		sys.exit(0)