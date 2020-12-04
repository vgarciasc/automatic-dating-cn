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

import scipy.sparse
import scipy.sparse.csgraph

import reader

def generate_lemmarank_file(in_data_filename, out_data_filename, data_path, rank_len, rank_fn, verbose=True):
	""" Generates a 'lemmarank' file, that is, a CSV file with a series of columns called 'rank_i_lemma' and 
		'rank_i_score', with 'i' going from 0 to (rank_len - 1). These correspond to the top 'rank_len' lemmas 
		in the books, according to the metric defined by the function 'rank_fn'. """

	metadata_df = pd.read_csv(in_data_filename, sep=";", quotechar='"')

	# Creates 'rank' columns in DataFrame
	data = {}
	for i in range(0, rank_len):
		data["rank_" + str(i) + "_lemma"] = []
		data["rank_" + str(i) + "_score"] = []

	if verbose:
		print("generating lemmarank file...")

	# Fill out the columns for each book
	for index, row in metadata_df.iterrows():
		filename = join(data_path, row['filename'])
		G, word_counter = reader.read_colonia_file(filename)

		ranked_lemmas = rank_fn(G, word_counter, rank_len)

		if verbose:
			print("\t--------")
			print("\t", (index + 1), "/", len(metadata_df), ": filename", filename)

		for i in range(0, len(ranked_lemmas)):
			word, score = ranked_lemmas[i]
			data["rank_" + str(i) + "_lemma"].append(word)
			data["rank_" + str(i) + "_score"].append(score)

	# Outputs the file
	for key, val in data.items():
		metadata_df[key] = val
	metadata_df.to_csv(out_data_filename, sep=';', quoting=QUOTE_ALL)

def generate_similarity_file(metadata_filename, lemmarank_filename, out_data_filename, data_path, \
	test_filenames, rank_len, similarity_fn, verbose=True):
	""" Generates a 'similarity' file, that is, a CSV file with a column for each century (16th~20th).
		Each column corresponds to the mean similarity between the current row book and the books from
		that column century. The similarity between two books is done via 'lemmarank' file, and 
		specified by the 'similarity_fn' parameter (e.g.: Jaccard Similarity).

		WARNING: the parameter 'test_filenames' should include all the books contained in the test set.
		Failure in doing so will result in the similarity score taking those test books into account,
		consequently contaminating the results. """

	metadata_df = pd.read_csv(metadata_filename, sep=";", quotechar='"')
	lemmarank_df = pd.read_csv(lemmarank_filename, sep=";", quotechar='"')

	data = {
		"16th_century_mean_similarity": [],
		"17th_century_mean_similarity": [],
		"18th_century_mean_similarity": [],
		"19th_century_mean_similarity": [],
		"20th_century_mean_similarity": [],
	}

	if verbose:
		print("generating similarity file...")

	for i, row in lemmarank_df.iterrows():
		filename = join(data_path, row['filename'])
		G, word_counter = reader.read_colonia_file(filename)

		if verbose:
			print("\t--------")
			print("\t", (i + 1), "/", len(lemmarank_df), ": filename", filename)

		similarities_16th_cent = []
		similarities_17th_cent = []
		similarities_18th_cent = []
		similarities_19th_cent = []
		similarities_20th_cent = []

		ranked_lemmas = row["rank_0_lemma":("rank_" + str(rank_len - 1) + "_score")]

		# Fills out similarities between this book and other books in other centuries.
		# WARNING: should exclude the books in the test set!
		for i2, row2 in lemmarank_df.iterrows():
			if row2["filename"] == row["filename"] or row2["filename"] in test_filenames:
				continue

			ranked_lemmas_2 = row2["rank_0_lemma":("rank_" + str(rank_len - 1) + "_score")]
			similarity = similarity_fn(ranked_lemmas, ranked_lemmas_2)

			if row2["century"] == "16th Century":
				similarities_16th_cent.append(similarity)
			elif row2["century"] == "17th Century":
				similarities_17th_cent.append(similarity)
			elif row2["century"] == "18th Century":
				similarities_18th_cent.append(similarity)
			elif row2["century"] == "19th Century":
				similarities_19th_cent.append(similarity)
			elif row2["century"] == "20th Century":
				similarities_20th_cent.append(similarity)

		data["16th_century_mean_similarity"].append(np.mean(similarities_16th_cent))
		data["17th_century_mean_similarity"].append(np.mean(similarities_17th_cent))
		data["18th_century_mean_similarity"].append(np.mean(similarities_18th_cent))
		data["19th_century_mean_similarity"].append(np.mean(similarities_19th_cent))
		data["20th_century_mean_similarity"].append(np.mean(similarities_20th_cent))

	# Generates the CSV file
	for key, val in data.items():
		metadata_df[key] = val
	metadata_df.to_csv(out_data_filename, sep=';', quoting=QUOTE_ALL)		

def extract_most_freq(G, word_counter, rank_len):
	""" Ranks lemmas by frequency. """
	return word_counter.most_common(rank_len)

def extract_most_closeness(G, word_counter, rank_len):
	""" Ranks lemmas by closeness. """
	ranked = [(k, v) for k, v in nx.closeness_centrality(G).items()]
	ranked = sorted(ranked, key=lambda tup:tup[1], reverse=True)[:rank_len]
	return [(key, val) for key, val in ranked]

def jaccard_similarity(ranked_lemmas_1, ranked_lemmas_2):
	""" Defines similarity using the Jaccard Similarity metric. """
	set1 = set([k for k in ranked_lemmas_1 if isinstance(k, str)])
	set2 = set([k for k in ranked_lemmas_2 if isinstance(k, str)])
	return len(set1.intersection(set2)) / len(set1.union(set2))

if __name__ == '__main__':
	metadata_filename = "colonia_metadata.csv"
	lemmarank_data_filename = "book_wordfreq.csv"
	similarity_data_filename = "century_similarities.csv"
	data_path = "data/txt_colonia/"

	rank_len = 10

	generate_lemmarank_file(metadata_filename, lemmarank_data_filename, data_path, rank_len, extract_most_freq)
	generate_similarity_file(metadata_filename, lemmarank_data_filename, similarity_data_filename, data_path, \
		[], rank_len, jaccard_similarity)