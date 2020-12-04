import networkx as nx
import numpy as np
import pandas as pd
import pdb

from os import listdir
from os.path import join
from collections import Counter
from csv import QUOTE_ALL

# no stopword removal
# STOP_WORD_GROUPS = ["QUOTE"]
# STOP_WORD_LEMMAS = []

# aggressive stopword removal
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

			lemma = parsed[2].lower()
			if parsed[2] == "<unknown>":
				lemma = parsed[0].lower()
			
			word = lemma
			word_counter[word] += 1

			add_node(G, word)
			add_edge(G, last_word, word)
			
			last_word = word

	return G, word_counter