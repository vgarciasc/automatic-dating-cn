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

def generate_graphs_file(in_data_filename, data_path, graph_path, model_str, read_fn=r.read_colonia_file, verbose=True):
	data_df = pd.read_csv(in_data_filename, sep=";", quotechar='"')

	if verbose:
		print("generating graphs file...")

	for index, row in data_df.iterrows():
		filename = join(data_path, row['filename'])

		if model_str == "mod1":
			G, word_counter = read_fn(filename, should_remove_stopwords=False, should_separate_sentences=False)
		elif model_str == "mod2":
			G, word_counter = read_fn(filename, should_remove_stopwords=True, should_separate_sentences=False)
		elif model_str == "mod3":
			G, word_counter = read_fn(filename, should_remove_stopwords=True, should_separate_sentences=True)

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
		"size_gcc": [],
		"strongly_connected_components": []
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

		components = graph.components()
		gcc = graph.subgraph(components[np.argmax([len(c) for c in components])])

		data["number_of_nodes"].append(graph.vcount())
		data["normalized_number_of_nodes"].append(graph.vcount() / row['words'])
		# data["number_of_edges"].append(graph.ecount())
		# data["normalized_number_of_edges"].append(graph.ecount() / row['words'])
		data["density"].append(graph.density())
		data["assortativity_coefficient"].append(graph.assortativity_degree([1/s if s != 0 else 0 for s in graph.strength()]))

		# average_shortest_path_length = total / (gcc.vcount() * (gcc.vcount() - 1))
		average_shortest_path_length = gcc.average_path_length()
		data["average_shortest_path_length"].append(average_shortest_path_length)

		data["diameter"].append(graph.diameter(weights=[1/w for w in graph.es["weight"]]))
		data["transitivity"].append(graph.transitivity_undirected(mode = "zero"))
		data["mean_degree"].append(np.mean(graph.strength(weights = "weight")))

		# clustering = gcc.transitivity_local_undirected(mode = "zero")
		# data["mean_clustering"].append(np.mean(clustering))
		# data["max_clustering"].append(np.max(clustering))

		# betweenness_centrality = graph.betweenness(weights = "weight")
		# data["mean_betweenness_centrality"].append(np.mean(betweenness_centrality))
		# data["max_betweenness_centrality"].append(np.max(betweenness_centrality))

		data["size_gcc"].append(gcc.vcount() / graph.vcount())
		data["strongly_connected_components"].append(len(components))
		
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
	network_metrics_filename = "data/network/network_metrics.csv"
	baseline_data_filename = "data/baseline/baseline_data.csv"

	graph_path = "data/graphs/"
	data_path = "data/txt_colonia"

	flag_1 = None
	flag_2 = None
	flag_3 = None
	flag_4 = None

	if len(sys.argv) > 1:
		flag_1 = sys.argv[1]
	if len(sys.argv) > 2:
		flag_2 = sys.argv[2]
	if len(sys.argv) > 3:
		flag_3 = sys.argv[3]
	if len(sys.argv) > 4:
		flag_4 = sys.argv[4]

	if flag_1 == "--graphs":
		model_str = flag_2
		if model_str == "mod1":
			graph_path += "mod1/"
		elif model_str == "mod2":
			graph_path += "mod2/"
		elif model_str == "mod3":
			graph_path += "mod3/"
		else:
			print("Wrong model received. Possibilities: mod1, mod2, mod3")
			sys.exit(1)

		# generate_graphs_file(tychobrahe_metadata_filename, "data/txt_tychobrahe", graph_path, model_str, read_fn = r.read_tychobrahe_file)
		generate_graphs_file(colonia_metadata_filename, data_path, graph_path, model_str, read_fn = r.read_colonia_file)

	elif flag_1 == "--baseline":
		rank_len = int(flag_2)

		csc.generate_lemmarank_file(colonia_metadata_filename, "data/baseline/lemmaranks.csv", data_path, rank_len, csc.extract_most_freq)
		generate_baseline_file(colonia_metadata_filename, baseline_data_filename, data_path)

	elif flag_1 == "--network":
		rank_len = int(flag_2)

		model_str = flag_3
		if model_str == "mod1":
			graph_path += "mod1/"
		elif model_str == "mod2":
			graph_path += "mod2/"
		elif model_str == "mod3":
			graph_path += "mod3/"
		else:
			print("Wrong model received. Possibilities: mod1, mod2, mod3")
			sys.exit(1)

		network_metrics_filename = network_metrics_filename.split(".")[0] + "_" + model_str + ".csv"

		# csc.generate_lemmarank_file(colonia_metadata_filename, "data/network/lemmaranks.csv", data_path, rank_len, csc.extract_most_closeness)
		generate_network_metrics_file(colonia_metadata_filename, network_metrics_filename, data_path, graph_path)

	else:
		print("Incorrect usage detected. Use one of the following patterns:")
		print("\t main.py --baseline [rank length]")
		print("\t main.py --network [rank length]")
		sys.exit(0)